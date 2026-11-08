"""Val Loss로 고른 증강 checkpoint를 고정하고 PBR Val/실사에 사전 고정 평가 수행."""
from pathlib import Path
import os,sys,json,csv,time,subprocess,hashlib,traceback
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D
D.ROOT=ROOT

def guard():
 if time.time()>=1793714609+75*60:raise RuntimeError('시간 예산: 평가 중단 후 보고')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def model(path):
 m=D.S.explicit_module();net=D.S.build_network(m,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda();ck=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict(ck['model_state_dict'],strict=True);net.eval();return net,ck

def main():
 info=json.loads((ROOT/'checks/rotation_training.json').read_text());assert info['status']=='PASS';path=Path(info['best_val_loss']['checkpoint']);assert path.is_file();checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest();guard();torch.set_num_threads(2);cv2.setNumThreads(1)
 net,ck=model(path);assert ck['epoch']==info['best_val_loss']['epoch'];samples,train,val=D.S.dataset();raw={}
 def hook(m,a,out):raw['code']=out[1].detach().clone()
 h=net.net.register_forward_hook(hook);xyzrows=[];posrows=[]
 for split,ids in [('train',train),('val',val)]:
  for at in range(0,len(ids),4):
   guard();batch=ids[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in batch]).cuda();bbox=torch.stack([samples[i]['bbox'] for i in batch]).cuda()
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pred=net.inference_batch(x,bbox)
   hard=(torch.sigmoid(raw['code'])>.5).cpu().numpy().transpose(0,2,3,1)
   for j,i in enumerate(batch):
    guard();s=samples[i];box=s['bbox'].numpy();mask=pred['pred_mask'][j].cpu().numpy()>0;gtmask=s['tensors'][1].numpy()>.5;uv=pred['coord_2d_image'][j].float().cpu().numpy()
    ff=cv2.imread(str(BASE/'labels/front'/f'{i:06d}_000000.png'));bb=cv2.imread(str(BASE/'labels/back'/f'{i:06d}_000000.png'));valid=D.CROP((ff.any(-1)&bb.any(-1)).astype(np.uint8),box,128,cv2.INTER_NEAREST)>0;ma=gtmask&valid;miou=float(np.count_nonzero(mask&gtmask)/max(np.count_nonzero(mask|gtmask),1))
    for branch,full,key,off in [('front',ff,'pred_front_code_obj',0),('back',bb,'pred_back_code_obj',24)]:
     q=D.CROP(full,box,128,cv2.INTER_NEAREST);truth=q.astype(float)/255*D.SIZE+D.LO;p=pred[key][j].float().cpu().numpy();delta=p[ma]-truth[ma];target=D.ENC(None,q)>.5;z0=float(np.mean(hard[j,...,off+16][ma]!=target[...,16][ma]));row={'split':split,'frame':i,'branch':branch,'n':int(ma.sum()),'mask_iou':miou,'xyz_l2':float(np.linalg.norm(delta,axis=-1).mean()),'x_MAE':float(abs(delta[:,0]).mean()),'y_MAE':float(abs(delta[:,1]).mean()),'z_MAE':float(abs(delta[:,2]).mean()),'Z0_hard_error':z0};xyzrows.append(row)
     pr=D.pose('rotation_'+branch,i,mask,p,uv,s['K'],s['R'],s['t'].reshape(3));pr['split']=split;pr['branch']=branch;posrows.append(pr)
 table(ROOT/'stage3/pbr_xyz_per_frame.csv',xyzrows);table(ROOT/'stage3/pbr_pose_per_frame.csv',posrows)
 oldxyz=list(csv.DictReader((BASE/'diagnostics_train_val_v1/train_val_xyz_comparison.csv').open()));oldplanes=list(csv.DictReader((BASE/'diagnostics_train_val_v1/hcce_plane_error.csv').open()));oldpose=list(csv.DictReader((BASE/'diagnostics_train_val_v1/train_val_pose_metrics.csv').open()));summary=[]
 for split in ['train','val']:
  for branch in ['front','back']:
   xx=[r for r in xyzrows if r['split']==split and r['branch']==branch];pp=[r for r in posrows if r['split']==split and r['branch']==branch];good=[r for r in pp if r['valid_pose']];ox=next(r for r in oldxyz if r['split']==split and r['branch']==branch and r['scope']=='GTmask');op=next(r for r in oldpose if r['split']==split and r['branch']==branch);oz=next(r for r in oldplanes if r['split']==split and r['branch']==branch and r['axis']=='z' and r['plane']=='0')
   for label,xr,pr,z in [('baseline',ox,op,float(oz['hard_bit_error'])),('rotation',None,None,float(np.mean([r['Z0_hard_error'] for r in xx])))]:
    row={'model':label,'split':split,'branch':branch,'frames':len(xx),'mask_iou':float(xr['mask_iou']) if xr else float(np.mean([r['mask_iou'] for r in xx])),'xyz_l2':float(xr['xyz_l2']) if xr else float(np.mean([r['xyz_l2'] for r in xx])),'x_MAE':float(xr['x_MAE']) if xr else float(np.mean([r['x_MAE'] for r in xx])),'y_MAE':float(xr['y_MAE']) if xr else float(np.mean([r['y_MAE'] for r in xx])),'z_MAE':float(xr['z_MAE']) if xr else float(np.mean([r['z_MAE'] for r in xx])),'Z0_hard_error':z,'PnP_valid':int(pr['PnP_valid']) if pr else len(good),'ADD_pass':int(pr['ADD_pass']) if pr else sum(r['add_pass'] for r in pp),'ADD_rate':float(pr['ADD_rate']) if pr else sum(r['add_pass'] for r in pp)/len(pp),'add_mean_valid':float(pr['add_mean']) if pr else (float(np.mean([r['add'] for r in good])) if good else None),'rotation_deg_mean_valid':float(pr['rotation_deg_mean']) if pr else (float(np.mean([r['rotation_deg'] for r in good])) if good else None),'translation_mean_valid':float(pr['translation_mean']) if pr else (float(np.mean([r['translation'] for r in good])) if good else None)};summary.append(row)
 table(ROOT/'stage3/pbr_comparison.csv',summary);write(ROOT/'stage3/pbr_comparison.json',{'checkpoint_fixed_before_real_evaluation':True,'epoch':ck['epoch'],'sha256':checkpoint_hash,'rows':summary,'condition':'GT visible BBox, same batch4/AMP and evaluation scope; baseline epoch155 vs augmented ValLoss-selected checkpoint; train fit not independent'})
 h.remove();del net;torch.cuda.empty_cache()
 # Real evaluation happens only after model selection and PBR metrics are saved; no real GT is used for checkpoint selection.
 guard();net,ck=model(path);source=json.loads((ROOT/'stage1/pose_results.json').read_text());old=source['rows'];ids=[r['frame'] for r in old];assert len(ids)==27;byid={r['frame']:r for r in old};params=json.loads((Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')/'camera_params.json').read_text());K=np.array(params['camera_matrix']);gt=json.loads((Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')/'scene_gt_obj04.json').read_text());mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None];images=[]
 for frame in ids:
  im=cv2.imread(str(ROOT/'stage1/controlled_rgb'/f'{frame:06d}.png'));assert im is not None;box=np.array(json.loads(byid[frame]['bbox_padded_xywh']));rgb=D.CROP(im,box,256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255;images.append(((x-mean)/std,box,im))
 outdir=ROOT/'stage3/real_overlays';outdir.mkdir();real=[];D.RECS.clear()
 for at in range(0,len(ids),4):
  guard();part=ids[at:at+4];x=torch.stack([images[at+j][0] for j in range(len(part))]).cuda();boxes=torch.tensor(np.array([images[at+j][1] for j in range(len(part))]),dtype=torch.float32,device='cuda')
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pred=net.inference_batch(x,boxes)
  for j,frame in enumerate(part):
   guard();r=np.array(gt[str(frame)][0]['cam_R_m2c']).reshape(3,3);t=np.array(gt[str(frame)][0]['cam_t_m2c']);mask=pred['pred_mask'][j].cpu().numpy()>0;uv=pred['coord_2d_image'][j].float().cpu().numpy();p=pred['pred_front_code_obj'][j].float().cpu().numpy();row=D.pose('rotation_real',frame,mask,p,uv,K,r,t);base=byid[frame];row['baseline_add']=base['add'];row['baseline_add_pass']=base['add_pass'];row['baseline_valid']=base['valid_pose'];row['baseline_inlier_ratio']=base['inlier_ratio_actual'];real.append(row);im=images[at+j][2].copy()
   for color,(rr,tt) in [((0,255,0),(r,t))]+([((255,0,255),(np.array(D.RECS[('rotation_real',frame)]['R']),np.array(D.RECS[('rotation_real',frame)]['t'])))] if row['valid_pose'] else []):
    proj=D.project(D.V,rr,tt,K);poly=[np.clip(proj[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in D.F];cv2.polylines(im,poly,True,color,1,cv2.LINE_AA)
   caption=f'rotation frame {frame:06d}, GT green / PnP magenta, ADD '+(f'{row["add"]:.2f}' if row['valid_pose'] else 'FAIL');cv2.putText(im,caption,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);assert cv2.imwrite(str(outdir/f'{frame:06d}.png'),im)
 table(ROOT/'stage3/real_comparison_per_frame.csv',real);good=[r for r in real if r['valid_pose']];result={'frames':27,'PnP_valid':len(good),'ADD_pass':sum(r['add_pass'] for r in real),'ADD_rate':sum(r['add_pass'] for r in real)/27,'mean_ADD_valid':float(np.mean([r['add'] for r in good])) if good else None,'mean_RE_valid':float(np.mean([r['rotation_deg'] for r in good])) if good else None,'mean_TE_valid':float(np.mean([r['translation'] for r in good])) if good else None,'mean_inlier_ratio_all':float(np.mean([r['inlier_ratio_actual'] for r in real])),'failed_pnp_frames':[r['frame'] for r in real if not r['valid_pose']],'baseline_ADD_pass':sum(r['baseline_add_pass'] for r in real),'model_selected_only_by_PBR_Val_loss':True,'real_data_used_for_training_or_model_selection':False,'real_development_only':True,'checkpoint_epoch':ck['epoch'],'checkpoint_sha256':checkpoint_hash}
 assert len(real)==27 and [r['frame'] for r in real]==ids and hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash
 write(ROOT/'stage3/real_comparison.json',result);write(ROOT/'stage3/returned_poses.json',[{'frame':f,**p} for (tag,f),p in D.RECS.items()]);write(ROOT/'checks/stage3_gate.json',{'status':'PASS','PBR_train_val_measured':True,'real_27_measured_after_selection':True,'checkpoint_epoch':ck['epoch']});print(json.dumps({'pbr':summary,'real':result},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'checks/stage3_evaluation_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc()});raise
