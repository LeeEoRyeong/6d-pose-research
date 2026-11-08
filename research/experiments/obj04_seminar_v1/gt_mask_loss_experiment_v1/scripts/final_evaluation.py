"""PBR 선택을 고정한 뒤 공식 checkpoint만 동일 조건으로 실사27 한 번 평가."""
from pathlib import Path
import json,csv,math,hashlib,traceback,time,torch,numpy as np,cv2
from common_experiment import ROOT,BASE,D,guard,write,load_module,model
from train_gt_mask import evaluate,make_truth

def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid_result(r):
 if r['valid_pose'] and not all(r[k] is not None and math.isfinite(float(r[k])) for k in ('add','rotation_deg','translation')):r.update(valid_pose=False,add_pass=False,add=None,rotation_deg=None,translation=None,failure='비유한 Pose 평가량')
 return r

def main():
 s=json.loads((ROOT/'checks/training_summary.json').read_text());assert s['status']=='PASS';official=Path(s['official']['checkpoint']);secondary=Path(s['secondary']['checkpoint']);assert official.is_file() and secondary.is_file();official_hash=sha(official)
 # Select and lock solely from PBR before reading any real image or real pose.
 frozen={'status':'FROZEN','selection_source':'PBR Val total loss epoch>=1','official_epoch':s['official']['epoch'],'official_checkpoint':str(official),'official_sha256':official_hash,'secondary_epoch':s['secondary']['epoch'],'secondary_rule':'PBR Val GT-valid Front/Back XYZ L2 mean, supplementary only','same_selected_epoch':s['same_selected_epoch'],'real27_used_for_selection':False,'real_model_evaluations_authorized':1}
 write(ROOT/'checks/selection_frozen.json',frozen);guard();torch.set_num_threads(2);cv2.setNumThreads(1);module=load_module();net=model(module,'cuda');ck=torch.load(official,map_location='cpu',weights_only=True);assert ck['epoch']==frozen['official_epoch'];net.load_state_dict(ck['model_state_dict'],strict=True);net.eval();samples,train,val=D.S.dataset();truth=make_truth(samples,val);v=evaluate(net,module,samples,val,truth)
 recorded=s['official']['metrics']['val'];diffs={k:abs(v[k]-recorded[k]) for k in ['total_loss','mask_iou','front_XYZ_L2','back_XYZ_L2','front_Z0_error']};assert max(diffs.values())<.001,diffs
 baseline=list(csv.DictReader((BASE/'diagnostics_train_val_v1/train_val_xyz_comparison.csv').open()));plane=list(csv.DictReader((BASE/'diagnostics_train_val_v1/hcce_plane_error.csv').open()));pose=list(csv.DictReader((BASE/'diagnostics_train_val_v1/train_val_pose_metrics.csv').open()));rows=[]
 for kind in ['front','back']:
  x=next(r for r in baseline if r['split']=='val' and r['branch']==kind and r['scope']=='GTmask');z=next(r for r in plane if r['split']=='val' and r['branch']==kind and r['axis']=='z' and r['plane']=='0');p=next(r for r in pose if r['split']=='val' and r['branch']==kind)
  rows.append({'model':'baseline','epoch':155,'branch':kind,'n':20,'mask_iou':float(x['mask_iou']),'XYZ_L2':float(x['xyz_l2']),'x_MAE':float(x['x_MAE']),'y_MAE':float(x['y_MAE']),'z_MAE':float(x['z_MAE']),'Z0_error':float(z['hard_bit_error']),'PnP_valid':int(p['PnP_valid']),'ADD_pass':int(p['ADD_pass']),'ADD_rate':float(p['ADD_rate']),'mean_ADD_valid':float(p['add_mean']),'mean_RE_valid':float(p['rotation_deg_mean']),'mean_TE_valid':float(p['translation_mean'])})
  rows.append({'model':'GT_mask_loss','epoch':frozen['official_epoch'],'branch':kind,'n':20,'mask_iou':v['mask_iou'],'XYZ_L2':v[kind+'_XYZ_L2'],'x_MAE':v[kind+'_x_MAE'],'y_MAE':v[kind+'_y_MAE'],'z_MAE':v[kind+'_z_MAE'],'Z0_error':v[kind+'_Z0_error'],'PnP_valid':v[kind+'_PnP_valid'],'ADD_pass':v[kind+'_ADD_pass'],'ADD_rate':v[kind+'_ADD_rate'],'mean_ADD_valid':v[kind+'_add_mean'],'mean_RE_valid':v[kind+'_rotation_deg_mean'],'mean_TE_valid':v[kind+'_translation_mean']})
 csvout(ROOT/'tables/pbr_validation_metrics.csv',rows);write(ROOT/'checks/pbr_final_reproduction.json',{'status':'PASS','recorded_vs_recomputed_abs_diff':diffs,'official_checkpoint_epoch':ck['epoch'],'PBR_metrics':v,'official_and_secondary_distinct':not s['same_selected_epoch']})
 # After the PBR selection is frozen, read the real data and perform exactly one model pass over 27 frames.
 old=json.loads((BASE/'agent_run_v1/stage1/pose_results.json').read_text());baseline_rows=old['rows'];ids=[x['frame'] for x in baseline_rows];assert len(ids)==len(set(ids))==27;ref={x['frame']:x for x in baseline_rows};data=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer');gt=json.loads((data/'scene_gt_obj04.json').read_text());K=np.array(json.loads((data/'camera_params.json').read_text())['camera_matrix']);mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None];imset=[]
 for i in ids:
  png=BASE/'agent_run_v1/stage1/controlled_rgb'/f'{i:06d}.png';im=cv2.imread(str(png));assert im is not None;box=np.array(json.loads(ref[i]['bbox_padded_xywh']));crop=D.CROP(im,box,256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(crop.transpose(2,0,1)),dtype=torch.float32)/255;imset.append(((x-mean)/std,box,im))
 images=ROOT/'figures/real27_overlays';images.mkdir();real=[];D.RECS.clear()
 for at in range(0,len(ids),4):
  guard();part=ids[at:at+4];x=torch.stack([imset[at+j][0] for j in range(len(part))]).cuda();bb=torch.tensor(np.array([imset[at+j][1] for j in range(len(part))]),dtype=torch.float32,device='cuda')
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pred=net.inference_batch(x,bb)
  for j,i in enumerate(part):
   guard();rr=np.array(gt[str(i)][0]['cam_R_m2c']).reshape(3,3);tt=np.array(gt[str(i)][0]['cam_t_m2c']);pm=pred['pred_mask'][j].cpu().numpy()>0;xyz=pred['pred_front_code_obj'][j].float().cpu().numpy();uv=pred['coord_2d_image'][j].float().cpu().numpy();result=valid_result(D.pose('GTmaskLoss_real',i,pm,xyz,uv,K,rr,tt));old=ref[i];result.update(baseline_add=old['add'],baseline_valid=old['valid_pose'],baseline_pass=old['add_pass'],GT_quality_prior=old['quality_prior']);real.append(result)
   im=imset[at+j][2].copy();transforms=[((0,255,0),(rr,tt))]
   if result['valid_pose']:
    p=D.RECS[('GTmaskLoss_real',i)];transforms.append(((255,0,255),(np.array(p['R']),np.array(p['t']))))
   for color,(rot,t) in transforms:
    proj=D.project(D.V,rot,t,K);polys=[np.clip(proj[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in D.F];cv2.polylines(im,polys,True,color,1,cv2.LINE_AA)
   caption=f'frame {i:06d} GT green / prediction magenta | ADD '+(f'{result["add"]:.2f}' if result['valid_pose'] else 'PnP failed');cv2.putText(im,caption,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);assert cv2.imwrite(str(images/f'{i:06d}.png'),im)
 assert len(real)==27 and [r['frame'] for r in real]==ids and sha(official)==official_hash
 csvout(ROOT/'tables/real27_metrics.csv',real);good=[r for r in real if r['valid_pose']];summary={'status':'PASS','n':27,'PnP_valid':len(good),'ADD_pass':sum(r['add_pass'] for r in real),'ADD_rate':sum(r['add_pass'] for r in real)/27,'mean_inlier_ratio_all27':float(np.mean([r['inlier_ratio_actual'] for r in real])),'PnP_failure_frames':[r['frame'] for r in real if not r['valid_pose']],'mean_ADD_valid':float(np.mean([r['add'] for r in good])) if good else None,'median_ADD_valid':float(np.median([r['add'] for r in good])) if good else None,'mean_RE_valid':float(np.mean([r['rotation_deg'] for r in good])) if good else None,'median_RE_valid':float(np.median([r['rotation_deg'] for r in good])) if good else None,'mean_TE_valid':float(np.mean([r['translation'] for r in good])) if good else None,'median_TE_valid':float(np.median([r['translation'] for r in good])) if good else None,'checkpoint_epoch':ck['epoch'],'checkpoint_sha256':official_hash,'real27_inference_runs':1,'real_used_for_selection':False,'ROI':'same GT CAD projection and controlled undistorted PNG as agent_run_v1/stage1','physical_unit':'original CAD scale, not independently certified mm'}
 write(ROOT/'checks/real27_summary.json',summary);write(ROOT/'tables/real27_returned_poses.json',[{'frame':i,**r} for (tag,i),r in D.RECS.items()]);print(json.dumps({'PBR':rows,'real27':summary},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'checks/final_evaluation_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc(),'do_not_change_selection_or_retry_real':True});raise
