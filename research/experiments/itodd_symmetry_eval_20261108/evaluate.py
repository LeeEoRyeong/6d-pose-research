"""Two-model evaluation: controlled ROI/K, fixed PnP, no training."""
import csv,hashlib,itertools,json,sys,datetime,traceback,subprocess
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
import torchvision.ops.misc as misc
if not hasattr(misc,'ConvNormActivation'):misc.ConvNormActivation=misc.Conv2dNormActivation
import symmetry
ROOT=Path(__file__).resolve().parent
EXP=Path('/RESEARCH_DATA/HCCEPose_experiments');BASE=EXP/'obj04_seminar_v1';RUN=BASE/'runs/baseline_seed42_20261103_214353/trained_only'
SCENE=EXP/'realistic_final100/train_pbr/000000'; REAL=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer'); AUD=EXP/'itodd_official_audit_20261108'
sys.path.insert(0,str(RUN/'scripts'));from support import explicit_module,build_network,original_function

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def json_read(p):return json.loads(Path(p).read_text())
def write(p,value):
 with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def png(p,im):
 if p.exists():raise FileExistsError(p)
 assert cv2.imwrite(str(p),im)
def project(v,R,t,K):
 c=v@R.T+np.asarray(t).reshape(1,3);uv=c@K.T;return uv[:,:2]/uv[:,2:]
def sample(id,img,K,R,t,box,domain,crop):
 im=crop(img,np.asarray(box),256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/255
 x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
 return {'id':id,'image':img,'crop':im,'input':x,'K':K,'R':R,'t':t,'box':box,'domain':domain}
def get_data(v):
 crop=original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});pad=original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np})
 split=json_read(RUN/'checks/split.json');assert len(split['val_ids'])==20 and not set(split['train_ids'])&set(split['val_ids'])
 gt=json_read(SCENE/'scene_gt.json');cam=json_read(SCENE/'scene_camera.json');gi=json_read(SCENE/'scene_gt_info.json');data={'pbr':[],'real':[]};files=[]
 for i in split['val_ids']:
  a=gt[str(i)][0];R=np.array(a['cam_R_m2c']).reshape(3,3);t=np.array(a['cam_t_m2c']);K=np.array(cam[str(i)]['cam_K']).reshape(3,3);box=pad(np.array(gi[str(i)][0]['bbox_visib'],dtype=float),1.5)
  p=SCENE/'rgb'/f'{i:06d}.jpg';im=cv2.imread(str(p));assert im is not None;files.append(p);data['pbr'].append(sample(i,im,K,R,t,box,'pbr',crop))
 gt=json_read(REAL/'scene_gt_obj04.json');cameras=json_read(REAL/'scene_camera_obj04.json');params=json_read(REAL/'camera_params.json');K=np.array(params['camera_matrix']);dist=np.array(params['distortion_coefficients']);mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,K,(1280,720),cv2.CV_32FC1)
 for i in sorted(map(int,gt)):
  a=gt[str(i)][0];R=np.array(a['cam_R_m2c']).reshape(3,3);t=np.array(a['cam_t_m2c']);assert np.allclose(K,np.array(cameras[str(i)]['cam_K']).reshape(3,3),atol=1e-9)
  uv=project(v,R,t,K);low=uv.min(0);high=uv.max(0);box=pad(np.r_[low,high-low],1.5);path=REAL/'image'/f'{i:06d}.jpg';im=cv2.remap(cv2.imread(str(path)),mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  known=BASE/'agent_run_v1/stage1/controlled_rgb'/f'{i:06d}.png';ref=cv2.imread(str(known));assert ref is not None and np.array_equal(im,ref),f'controlled undistortion mismatch {i}'
  files.extend([path,known]);data['real'].append(sample(i,im,K,R,t,box,'real',crop))
 assert len(data['pbr'])==20 and len(data['real'])==27
 return data,files

def visualize(folder,s,mask,front,back,pose,v,faces,lo,size):
 frame=s['id'];name=f'{frame:06d}';png(folder/f'{name}_mask.png',mask.astype(np.uint8)*255)
 p=folder/f'{name}_xyz.npz'
 with p.open('xb') as f:np.savez_compressed(f,front_xyz=front.astype('float32'),back_xyz=back.astype('float32'),predicted_mask=mask.astype('uint8'),bbox=np.array(s['box']),cam_K=s['K'])
 im=s['image'].copy()
 def wire(R,t,color):
  uv=project(v,R,t,s['K']);polys=[np.clip(uv[face],-50000,50000).astype(np.int32).reshape(-1,1,2) for face in faces];cv2.polylines(im,polys,True,color,1,cv2.LINE_AA)
 wire(s['R'],s['t'],(0,255,0))
 if pose is not None:wire(pose[0],pose[1],(255,0,255))
 cv2.putText(im,f'{name} GT green / PnP magenta' if pose is not None else f'{name} GT green / PnP FAILED',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);png(folder/f'{name}_pose_overlay.png',im)
 def xyz_img(x):
  q=np.clip((x-lo)/size,0,1);q=(q*255).astype('uint8');q[~mask]=0;return q
 imgs=[cv2.cvtColor(s['crop'],cv2.COLOR_BGR2RGB),np.repeat((mask*255).astype('uint8')[:,:,None],3,axis=2),xyz_img(front),xyz_img(back)]
 panel=Image.new('RGB',(1024,285),(20,20,20));d=ImageDraw.Draw(panel)
 for j,(a,title) in enumerate(zip(imgs,['RGB ROI','Predicted mask','Front XYZ: R=X G=Y B=Z','Back XYZ: R=X G=Y B=Z'])):
  pic=Image.fromarray(a).resize((256,256),Image.Resampling.NEAREST);panel.paste(pic,(256*j,29));d.text((256*j+3,8),title,fill='white')
 path=folder/f'{name}_prediction_panel.png'
 with path.open('xb') as f:panel.save(f,format='PNG')

def run_one(tag,net,ck,data,syms,v,faces,info,lo,size,run_dir,metrics_enabled):
 net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval();torch.cuda.reset_peak_memory_stats();solve=original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});rows=[]
 for domain,items in data:
  folder=run_dir/tag/domain;folder.mkdir(parents=True,exist_ok=False)
  for start in range(0,len(items),4):
   batch=items[start:start+4];x=torch.stack([s['input'] for s in batch]).cuda();box=torch.tensor(np.array([s['box'] for s in batch]),dtype=torch.float32,device='cuda')
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
   for j,s in enumerate(batch):
    mask=out['pred_mask'][j].cpu().numpy()>0;uv=out['coord_2d_image'][j].float().cpu().numpy();front=out['pred_front_code_obj'][j].float().cpu().numpy();back=out['pred_back_code_obj'][j].float().cpu().numpy()
    r={'model':tag,'domain':domain,'frame':s['id'],'mask_pixels':int(mask.sum()),'pnp_valid':False,'failure':None,'ADD':None,'ADD_S':None,'symmetry_ADD':None,'symmetry_index':None,'ADD_pass':False,'ADD_S_pass':False,'symmetry_ADD_pass':False,'pred_R':None,'pred_t':None}
    pose=None
    try:
     if mask.sum()<=4:raise ValueError('predicted mask contains at most 4 pixels')
     cv2.setRNGSeed(42+s['id']);pnp=solve((mask,front,uv,s['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
     if not pnp['success']:raise ValueError('solve_PnP returned success=False')
     R=np.asarray(pnp['rot'],dtype=float);t=np.asarray(pnp['tvecs'],dtype=float).reshape(3)
     if not(np.isfinite(R).all() and np.isfinite(t).all()):raise ValueError('non-finite pose')
     if np.min((v@R.T+t)[:,2])<=0:raise ValueError('non-positive mesh depth')
     pose=(R,t);r.update(pnp_valid=True,pred_R=R.tolist(),pred_t=t.tolist())
     if metrics_enabled:
      a=symmetry.add(v,R,t,s['R'],s['t']);asym=symmetry.add_s(v,R,t,s['R'],s['t']);ea,idx=symmetry.symmetry_add(v,R,t,s['R'],s['t'],syms);th=.1*info['diameter']
      r.update(ADD=a,ADD_S=asym,symmetry_ADD=ea,symmetry_index=idx,ADD_pass=a<th,ADD_S_pass=asym<th,symmetry_ADD_pass=ea<th)
    except Exception as e:r['failure']=type(e).__name__+': '+str(e)
    visualize(folder,s,mask,front,back,pose,v,faces,lo,size);rows.append(r)
    print(tag,domain,s['id'],'pnp',r['pnp_valid'],'ADD',r['ADD'],'ADD-S',r['ADD_S'],'sym',r['symmetry_ADD'],flush=True)
 peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
 return rows,peak

def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 if apps:raise RuntimeError('GPU compute already active: '+apps)
 validation=json_read(ROOT/'symmetry_validation.json');passed=validation['status']=='PASS'
 v,faces,syms,info=symmetry.load();data,files=get_data(v)
 checkpoints={'official':AUD/'itodd_obj04_1_0000step50000.pt','baseline':RUN/'checkpoints/best_val_loss.pt'}
 files.extend([*checkpoints.values(),AUD/'official_models_info.json',AUD/'official_obj_000004.ply',RUN/'checks/split.json',SCENE/'scene_gt.json',SCENE/'scene_camera.json',SCENE/'scene_gt_info.json',REAL/'scene_gt_obj04.json',REAL/'scene_camera_obj04.json',REAL/'camera_params.json'])
 before={str(p):sha(p) for p in files};module=explicit_module();localinfo=json_read(BASE/'data/models/models_info.json')['4'];lo=np.array([localinfo['min_'+a] for a in 'xyz']);size=np.array([localinfo['size_'+a] for a in 'xyz']);minimum=torch.tensor(lo,dtype=torch.float32,device='cuda');scale=torch.tensor(size,dtype=torch.float32,device='cuda')
 outdir=ROOT/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'));outdir.mkdir(exist_ok=False)
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):official=torch.load(checkpoints['official'],map_location='cpu',weights_only=True)
 baseline=torch.load(checkpoints['baseline'],map_location='cpu',weights_only=True);net=module.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=minimum,size_xyz=scale)
 # Five selected real frames first; then remaining 22 and all PBR only after symmetry PASS.
 chosen={26,51,78,99,126};first=[s for s in data['real'] if s['id'] in chosen];assert len(first)==5
 tasks=[('real_first5',first)]
 if passed:tasks += [('real_remaining',[s for s in data['real'] if s['id'] not in chosen]),('pbr',data['pbr'])]
 rows,peak1=run_one('official',net,official,tasks,syms,v,faces,info,lo,size,outdir,passed);del net;torch.cuda.empty_cache()
 if passed:
  net=build_network(module,minimum,scale);extra,peak2=run_one('baseline',net,baseline,[('real',data['real']),('pbr',data['pbr'])],syms,v,faces,info,lo,size,outdir,True);rows+=extra;del net;torch.cuda.empty_cache()
 else:peak2=None
 after={str(p):sha(p) for p in files};assert after==before
 keys=['model','domain','frame','mask_pixels','pnp_valid','failure','ADD','ADD_S','symmetry_ADD','symmetry_index','ADD_pass','ADD_S_pass','symmetry_ADD_pass','pred_R','pred_t']
 with (outdir/'per_frame.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(r[k]) if k in ('pred_R','pred_t') else r[k] for k in keys} for r in rows])
 summaries={}
 for tag in set(r['model'] for r in rows):
  for domain in ('real','pbr'):
   a=[r for r in rows if r['model']==tag and (r['domain'].startswith('real') if domain=='real' else r['domain']=='pbr')]
   if not a:continue
   ok=[r for r in a if r['pnp_valid']];summaries[tag+'_'+domain]={'n':len(a),'pnp_valid':len(ok),'failure_count':len(a)-len(ok),'ADD_pass':sum(r['ADD_pass'] for r in a),'ADD_S_pass':sum(r['ADD_S_pass'] for r in a),'symmetry_ADD_pass':sum(r['symmetry_ADD_pass'] for r in a),'valid_mean_ADD':float(np.mean([r['ADD'] for r in ok])) if passed and ok else None,'valid_mean_ADD_S':float(np.mean([r['ADD_S'] for r in ok])) if passed and ok else None,'valid_mean_symmetry_ADD':float(np.mean([r['symmetry_ADD'] for r in ok])) if passed and ok else None}
 result={'symmetry_validation':validation,'metrics_enabled':passed,'unit':'BOP CAD mm; real physical GT scale assumed','threshold':.1*info['diameter'],'pnp':'front-only original solve_PnP pnp_op=2, 1.5px, 150 iterations, seed=42+frame','pbr_roi':'GT visible bbox pad1.5','real_roi':'GT CAD projection bbox pad1.5 oracle','real_undistort':'raw remap K,D,newK=K; all 27 pixel-equal previous controlled PNG','loss_training':'none; inference only','roi_limit':'oracle, not end-to-end','pbr_overlap':'unverified vs official BOP training, no independent claim','real_limit':'developmental 27; five frames prior ambiguous','input_files_hash_count':len(before),'input_hashes_unchanged':True,'official_peak':peak1,'baseline_peak':peak2,'summary':summaries,'input_hashes':before}
 write(outdir/'summary.json',result);print('RESULT',outdir,flush=True);print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  path=ROOT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json');write(path,{'error':repr(e),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
