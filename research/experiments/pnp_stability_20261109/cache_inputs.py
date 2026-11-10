"""Cache predictions and exact PnP inputs; reads no GT pose."""
import sys,json,csv,hashlib,itertools,os,argparse
from pathlib import Path
import cv2,numpy as np,torch
P=Path(__file__).resolve().parent;E=P.parent;DEV=Path('/home/oylee/projects/HCCEPose_obj04_transfer');OLD=E/'itodd_symmetry_eval_20261108';Y=E/'yolo_hcce_rgb_pipeline';ROI=E/'roi_prediction_stability_20261109';sys.path.insert(0,str(OLD));import evaluate as e,symmetry

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def direct(pts3,pts2,K,seed):
 if len(pts3)<=4:return None
 cv2.setRNGSeed(seed)
 try:
  success,rv,tv,rawin=cv2.solvePnPRansac(pts3,pts2,K,None,reprojectionError=1.5,iterationsCount=150,flags=cv2.SOLVEPNP_EPNP)
  if not success or rv is None or tv is None:return None
  return cv2.Rodrigues(rv)[0],tv.reshape(3),rawin
 except cv2.error:return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=P/('smoke' if a.smoke else 'full');out.mkdir(exist_ok=False)
 if not a.smoke:assert json.loads((P/'smoke/cache_check.json').read_text())['status']=='PASS'
 assert not e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU busy';torch.set_num_threads(2);cv2.setNumThreads(1)
 prior=read(ROI/'full/per_frame.csv');assert len(prior)==324;ids=sorted({int(x['frame']) for x in prior});assert len(ids)==27;important={32,66,68,78,92,94,96};chosen=[r for r in prior if r['condition']=='baseline' or int(r['frame']) in important]
 if a.smoke:chosen=[r for r in chosen if (int(r['frame']),r['condition']) in {(26,'baseline'),(66,'baseline'),(66,'scale_1.20'),(66,'scale_0.85'),(68,'baseline'),(94,'scale_1.40')}]
 assert len(chosen)==(6 if a.smoke else 104);cal=json.loads((DEV/'camera_params.json').read_text());K=np.array(cal['camera_matrix'],float);D=np.array(cal['distortion_coefficients'],float);mapx,mapy=cv2.initUndistortRectifyMap(K,D,None,K,(1280,720),cv2.CV_32FC1);oldrefs={int(r['frame']):r for r in read(Y/'pose_eval.csv')};v,faces,syms,info=symmetry.load();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+x] for x in 'xyz']);size=np.array([li['size_'+x] for x in 'xyz']);mn=torch.tensor(lo,dtype=torch.float32,device='cuda');sz=torch.tensor(size,dtype=torch.float32,device='cuda');net=e.explicit_module().HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=mn,size_xyz=sz);cp=e.AUD/'itodd_obj04_1_0000step50000.pt'
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(cp,map_location='cpu',weights_only=True)
 net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval();del ck;crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});sources=[cp,e.AUD/'official_obj_000004.ply',e.AUD/'official_models_info.json',Y/'pose_eval.csv',Y/'detector_eval.csv',ROI/'full/per_frame.csv',DEV/'camera_params.json']+[DEV/'image'/f'{i:06d}.jpg' for i in sorted({int(r['frame']) for r in chosen})];before={str(p):sha(p) for p in sources};cache=out/'cache';cache.mkdir();records=[];tensor_checks=[]
 for r in chosen:
  i=int(r['frame']);name=r['condition'];box=np.array(json.loads(r['bbox_xywh']),float);im=cv2.remap(cv2.imread(str(DEV/'image'/f'{i:06d}.jpg')),mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);refim=cv2.imread(str(e.BASE/'agent_run_v1/stage1/controlled_rgb'/f'{i:06d}.png'));assert np.array_equal(im,refim);cropped=crop(im,box,256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(cropped.transpose(2,0,1))).float()/255;x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):o=net.inference_batch(x[None].cuda(),torch.tensor(box[None],dtype=torch.float32,device='cuda'))
  mask=o['pred_mask'][0].cpu().numpy()>0;xyz=o['pred_front_code_obj'][0].float().cpu().numpy();uv=o['coord_2d_image'][0].float().cpu().numpy();xyzback=o['pred_back_code_obj'][0].float().cpu().numpy();p3=np.ascontiguousarray(xyz[mask,:].astype(np.float32));p2=np.ascontiguousarray(uv[mask,:].astype(np.float32));assert len(p2)==int(r['mask_pixels']) and np.isfinite(p2).all() and np.isfinite(p3).all()
  try:
   cv2.setRNGSeed(42+i);old=solve((mask,xyz,uv,K),pnp_op=2,reprojectionError=1.5,iterationsCount=150);oldok=bool(old['success']);R=np.asarray(old['rot'],float) if oldok else None;t=np.asarray(old['tvecs'],float).reshape(3) if oldok else None;oldin=old['inliers'].reshape(-1) if oldok else np.array([],np.int32)
   if oldok and (not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0):oldok=False
  except Exception:oldok=False;R=None;t=None;oldin=np.array([],np.int32)
  assert oldok==(r['pnp_valid']=='True'),(i,name,'prior pnp mismatch');dc=direct(p3,p2,K,42+i);newok=dc is not None and np.min((v@dc[0].T+dc[1])[:,2])>0;assert oldok==newok,(i,name,'original vs direct mismatch')
  if oldok:
   assert np.allclose(R,dc[0],atol=1e-7) and np.allclose(t,dc[1],atol=1e-6),(i,name,'direct pose mismatch')
   ref=oldrefs[i]
   if name=='baseline':assert np.allclose(R,json.loads(ref['pred_R']),atol=1e-5) and np.allclose(t,json.loads(ref['pred_t']),atol=1e-3)
   elif ref['pnp_valid']=='True':
    rb=np.array(json.loads(ref['pred_R']));tb=np.array(json.loads(ref['pred_t']));assert abs(angle(R,rb)-float(r['rotation_raw_deg']))<.002 and abs(np.linalg.norm(t-tb)-float(r['translation_raw_cad_units']))<.002
   proj=cv2.projectPoints(p3,cv2.Rodrigues(R)[0],t,K,None)[0].reshape(-1,2);err=np.linalg.norm(proj-p2,axis=1);check=np.flatnonzero(err<1.5);assert np.array_equal(oldin,check);reproj_median=float(np.median(err));inratio=len(oldin)/len(p2);rawcount=len(dc[2]) if dc[2] is not None else 0
  else:err=np.array([],np.float32);reproj_median='';inratio='';rawcount=''
  namefile=f'{i:06d}__{name}.npz';np.savez_compressed(cache/namefile,mask=mask,front_xyz=xyz,back_xyz=xyzback,coordinates=uv,points3d=p3,points2d=p2,K=K,bbox=box,crop_bgr=cropped,baseline_R=R if oldok else np.empty((0,)),baseline_t=t if oldok else np.empty((0,)),original_recomputed_inliers=oldin,reprojection_errors=err)
  records.append({'frame':i,'condition':name,'path':namefile,'point_count':len(p2),'points2_finite':bool(np.isfinite(p2).all()),'points3_finite':bool(np.isfinite(p3).all()),'bbox_xywh':json.dumps(box.tolist()),'seed_original':42+i,'baseline_pnp_valid':oldok,'ransac_raw_inlier_count':rawcount,'original_recomputed_inlier_count':len(oldin) if oldok else '', 'original_recomputed_inlier_ratio':inratio,'median_reprojection_error_all_px':reproj_median})
  print('CACHE',i,name,len(p2),'success',oldok,flush=True)
 with (out/'cache_inventory.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 assert before=={x:sha(Path(x)) for x in before};json.dump({'status':'PASS','samples':len(records),'prior_roi_reproduction':'PASS all selected conditions','direct_solver_equivalence':'PASS success/R/t and original reprojected inliers; direct EPNP RANSAC matching op2','pixel_equal_undistortion':True,'finite_correspondences':True,'source_hashes_unchanged':True,'source_sha256':before,'scope':'27 baseline plus all 12 conditions for 7 anomalous frames; no GT','input_weights_note':'YOLO checkpoint identified only, saved predicted bbox reused'},(out/'cache_check.json').open('x'),indent=2);print('CACHE PASS',len(records),flush=True)
if __name__=='__main__':main()
