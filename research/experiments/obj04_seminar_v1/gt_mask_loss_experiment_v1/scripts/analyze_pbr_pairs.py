"""이미 고정된 두 모델의 PBR Val 프레임별 XYZ와 PnP 결과 비교. 실사 접근 없음."""
import csv,json,math,torch,numpy as np
from common_experiment import ROOT,BASE,D,guard,load_module,model
from train_gt_mask import make_truth

def main():
 freeze=json.loads((ROOT/'checks/selection_frozen.json').read_text());assert freeze['status']=='FROZEN';guard();torch.set_num_threads(2);samples,train,val=D.S.dataset();truth=make_truth(samples,val);net=model(load_module(),'cuda');ck=torch.load(freeze['official_checkpoint'],map_location='cpu',weights_only=True);assert ck['epoch']==freeze['official_epoch'];net.load_state_dict(ck['model_state_dict']);net.eval()
 previous=json.loads((BASE/'diagnostics_v1/matched_batch4/checks/prediction_results.json').read_text());oldpose={r['frame']:r for r in previous['candidates'] if r['stage']=='front'};oldxyz={}
 with (BASE/'diagnostics_train_val_v1/tables/xyz_per_frame.csv').open() as f:
  for r in csv.DictReader(f):
   if r['split']=='val' and r['branch']=='front' and r['scope']=='GTmask':oldxyz[int(r['frame'])]=float(r['xyz_l2'])
 result=[]
 for at in range(0,len(val),4):
  guard();ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();box=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
  for j,i in enumerate(ids):
   guard();pm=out['pred_mask'][j].cpu().numpy()>0;xyz=out['pred_front_code_obj'][j].float().cpu().numpy();uv=out['coord_2d_image'][j].float().cpu().numpy();gm=samples[i]['tensors'][1].numpy()>.5
   import cv2
   valid=cv2.imread(str(ROOT/'support'/f'{i:06d}_front.png'),0)>0;valid2=cv2.imread(str(ROOT/'support'/f'{i:06d}_back.png'),0)>0;ma=gm&valid&valid2
   e=float(np.linalg.norm(xyz[ma]-truth[i]['front'][ma],axis=1).mean());r=D.pose('pbr_pair',i,pm,xyz,uv,samples[i]['K'],samples[i]['R'],samples[i]['t'].reshape(3))
   if r['valid_pose'] and not all(r[k] is not None and math.isfinite(float(r[k])) for k in ('add','rotation_deg','translation')):r.update(valid_pose=False,add_pass=False,add=None,rotation_deg=None,translation=None,failure='비유한 평가량')
   o=oldpose[i];result.append({'frame':i,'baseline_XYZ_L2':oldxyz[i],'new_XYZ_L2':e,'XYZ_L2_difference':e-oldxyz[i],'baseline_PnP_valid':o['valid_pose'],'new_PnP_valid':r['valid_pose'],'baseline_ADD':o['add'],'new_ADD':r['add'],'baseline_ADD_pass':o['add_pass'],'new_ADD_pass':r['add_pass'],'baseline_inlier_ratio':o['inlier_ratio_actual'],'new_inlier_ratio':r['inlier_ratio_actual'],'new_failure':r['failure']})
 assert len(result)==20
 with (ROOT/'tables/pbr_validation_per_frame.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 out={'status':'PASS','frames':20,'baseline_ADD_pass_frames':[r['frame'] for r in result if r['baseline_ADD_pass']],'new_ADD_pass_frames':[r['frame'] for r in result if r['new_ADD_pass']],'shared_ADD_pass_frames':[r['frame'] for r in result if r['baseline_ADD_pass'] and r['new_ADD_pass']],'mean_XYZ_difference_new_minus_baseline':float(np.mean([r['XYZ_L2_difference'] for r in result])),'XYZ_better_frames':sum(r['XYZ_L2_difference']<0 for r in result),'XYZ_worse_frames':sum(r['XYZ_L2_difference']>0 for r in result),'baseline_mean_inlier_ratio':float(np.mean([r['baseline_inlier_ratio'] for r in result])),'new_mean_inlier_ratio':float(np.mean([r['new_inlier_ratio'] for r in result])),'note':'same fixed Val20, no checkpoint selection, paired descriptive result only'}
 with (ROOT/'checks/pbr_frame_analysis.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
