"""BB100 raw provenance, controlled undistortion and conservative grouping audit."""
import csv,json,hashlib,sys,collections,datetime
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;SRC=Path('/RESEARCH_DATA/qwen_smt/data/BB_rgb_obj04/run_20260813_164421');EXP=Path('/RESEARCH_DATA/HCCEPose_experiments');BB=EXP/'real_bb100_20261108';DEV=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer');PRIOR=Path('/RESEARCH_DATA/HCCEPose/bb_visible_audit_20261108_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cv2.setNumThreads(2);gt=json.load((BB/'BB_obj04/scene_gt_obj04.json').open());cam=json.load((BB/'BB_obj04/scene_camera_obj04.json').open());cal=json.load((DEV/'camera_params.json').open());K=np.array(cal['camera_matrix']);D=np.array(cal['distortion_coefficients']);ids=sorted(map(int,gt));assert len(ids)==100 and set(gt)==set(cam)
 def index(root):
  out={}
  for p in root.rglob('*.jpg'):
   i=int(p.stem)
   if i in out:raise ValueError(f'duplicate frame {i}')
   out[i]=p
  return out
 raw=index(SRC/'rgb');und=index(BB/'rgb_undistorted');assert set(ids)<=raw.keys() and set(ids)<=und.keys()
 with (SRC/'frame_opc_mapping.csv').open(encoding='utf-8-sig',newline='') as f:logs={int(Path(r['frame_name']).stem):r for r in csv.DictReader(f)}
 assert set(ids)<=logs.keys()
 oldimgs=sorted((DEV/'image').glob('*.jpg'));oldgt=json.load((DEV/'scene_gt_obj04.json').open()); assert len(oldimgs)==27
 oldhash={int(p.stem):sha(p) for p in oldimgs};oldthumb={int(p.stem):cv2.resize(cv2.imread(str(p)),(160,90),interpolation=cv2.INTER_AREA).astype(np.float32) for p in oldimgs}
 prev_dir=HERE/'bb100_undistorted_K_original';prev_dir.mkdir(exist_ok=False);ovdir=HERE/'overlays/bb100_projection';ovdir.mkdir(parents=True,exist_ok=False)
 Kmap=cv2.initUndistortRectifyMap(K,D,None,K,(1280,720),cv2.CV_32FC1);rows=[];poses={};stats={};representatives={31,43,67,75,99,221,235,247,340,379}
 sys.path.insert(0,str(EXP/'bb_weak_supervision_20261108'));from pipeline import load_renderer
 renderer=load_renderer();v,faces=renderer.mesh()
 for i in ids:
  a=gt[str(i)][0];R=np.array(a['cam_R_m2c']).reshape(3,3);t=np.array(a['cam_t_m2c']);poses[i]=(R,t);assert np.max(abs(np.array(cam[str(i)]['cam_K']).reshape(3,3)-K))<1e-8
  pose=np.load(BB/'BB_obj04/poses_obj04'/f'pose_{i:06d}.npy',allow_pickle=False);pose_delta=float(max(np.max(abs(R-pose[:3,:3])),np.max(abs(t-pose[:3,3]))));assert pose_delta<1e-8
  source=cv2.imread(str(raw[i]));prior=cv2.imread(str(und[i]));assert source.shape==prior.shape==(720,1280,3)
  remap=cv2.remap(source,*Kmap,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);path=prev_dir/f'{i:06d}.png';assert not path.exists();assert cv2.imwrite(str(path),remap)
  delta=np.abs(remap.astype(np.int16)-prior.astype(np.int16));mae=float(delta.mean());stat=logs[i];thumb=cv2.resize(source,(160,90),interpolation=cv2.INTER_AREA).astype(np.float32);near=min(oldthumb,key=lambda j:float(np.mean(np.abs(thumb-oldthumb[j]))));m=float(np.mean(np.abs(thumb-oldthumb[near])));rawhash=sha(raw[i]);dups=[j for j,h in oldhash.items() if h==rawhash]
  xc=v@R.T+t;uv=xc@K.T;uv=uv[:,:2]/uv[:,2:];bbox=np.r_[uv.min(0),uv.max(0)-uv.min(0)];inview=bool((xc[:,2]>0).all() and uv[:,0].min()>=0 and uv[:,1].min()>=0 and uv[:,0].max()<1280 and uv[:,1].max()<720)
  nearest_pose=None
  for j in map(int,oldgt):
   z=oldgt[str(j)][0];R2=np.array(z['cam_R_m2c']).reshape(3,3);t2=np.array(z['cam_t_m2c']);re=float(np.degrees(np.arccos(np.clip((np.trace(R@R2.T)-1)/2,-1,1))));te=float(np.linalg.norm(t-t2));score=te+re
   if nearest_pose is None or score<nearest_pose[0]:nearest_pose=(score,j,re,te)
  row={'frame':i,'state':raw[i].parent.name,'source_session':'run_20260813_164421','frame_time':stat['frame_read_wall_time'],'frame_mono':stat['frame_read_mono_time'],'robot_step':stat['current_step'],'robot_step_name':stat['current_step_name'],'match_valid':stat['match_valid'],'raw_sha256':rawhash,'raw_exact_matches_dev27':','.join(map(str,dups)),'nearest_dev_raw_thumbnail_mae_frame':near,'nearest_dev_raw_thumbnail_mae':m,'nearest_dev_pose_by_translation_plus_rotation_frame':nearest_pose[1],'nearest_dev_pose_rotation_deg':nearest_pose[2],'nearest_dev_pose_translation_native_units':nearest_pose[3],'undistort_existing_mae_8bit':mae,'existing_undistort_history':'not documented; pixel comparison only','K_D_source':'dev27 camera_params: BB K matches; D for BB is unverified assumption','output_newK':'input K, no ROI/resize','pose_npy_json_max_abs':pose_delta,'projected_cad_amodal_bbox_xywh':json.dumps(bbox.tolist()),'projection_in_screen':inview,'pose_quality':'not independently verified','group_id':''}
  rows.append(row)
  if i in representatives:
   mask=cv2.imread(str(PRIOR/'pseudo_masks'/f'{i:06d}_amodal.png'),0);assert mask is not None;blend=remap.copy();p=mask>0;blend[p]=(blend[p]*.55+np.array([0,200,0])*.45).astype(np.uint8);cont,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(blend,cont,-1,(0,255,255),1);cv2.putText(blend,f'{i:06d} CAD amodal / new undistort; not GT mask',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);assert cv2.imwrite(str(ovdir/f'{i:06d}.png'),blend)
  if i%50==0:print('BB audit frame',i,flush=True)
 # Connected components; a close time, or similar pose, or identical pose makes an edge.
 parent={i:i for i in ids}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for ai,i in enumerate(ids):
  R,t=poses[i];t0=float(logs[i]['frame_read_mono_time'])
  for j in ids[ai+1:]:
   R2,t2=poses[j];dt=abs(float(logs[j]['frame_read_mono_time'])-t0);te=float(np.linalg.norm(t-t2));re=float(np.degrees(np.arccos(np.clip((np.trace(R@R2.T)-1)/2,-1,1))))
   if dt<=1.0 or (te<=5.0 and re<=5.0):union(i,j)
 for row in rows:row['group_id']=find(row['frame'])
 with (HERE/'bb100_audit.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 group=collections.defaultdict(list)
 for r in rows:group[r['group_id']].append(r['frame'])
 result={'status':'PASS_GENERATION_CONDITIONAL_CALIBRATION','raw_source':str(SRC),'raw_frames_available':len(raw),'bb_frames':100,'raw_exact_duplicates_dev27':sum(bool(r['raw_exact_matches_dev27']) for r in rows),'undistort_existing_mean_mae':float(np.mean([r['undistort_existing_mae_8bit'] for r in rows])),'undistort_existing_max_mae':max(r['undistort_existing_mae_8bit'] for r in rows),'all_projection_in_view':all(r['projection_in_screen'] for r in rows),'reference_K_D':'dev27 calibration, BB K JSON matches exactly, BB D provenance not independently verified','group_rules':'same session, undirected edge if |frame time|<=1 s OR translation<=5 CAD units AND rotation<=5 degrees; transitive components. exploratory thresholds','groups':{str(k):v for k,v in group.items()},'source_session_count':1,'pose_ground_truth':'not independently validated','raw_hashes_unchanged':all(sha(raw[r['frame']])==r['raw_sha256'] for r in rows)}
 with (HERE/'bb100_summary.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print('BB SUMMARY',json.dumps({k:v for k,v in result.items() if k!='groups'},ensure_ascii=False), 'group_sizes',sorted((len(x) for x in group.values()),reverse=True),flush=True)
if __name__=='__main__':main()
