"""실사 GT·카메라 선행조건 검증 및 통제 보정 RGB의 oracle ROI 기반 단일 checkpoint 평가."""
from pathlib import Path
import json,hashlib,sys,os,subprocess,time,traceback,csv
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;DATA=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer');PREV=Path('/RESEARCH_DATA/HCCEPose/validation_obj04_transfer_20261103');SOURCE=BASE/'runs/baseline_seed42_20261103_214353/trained_only';DEADLINE=1793714609+75*60
sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D
D.ROOT=ROOT

def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def guard():
 if time.time()>=DEADLINE:raise RuntimeError('75분 경과: 계산을 중단하고 최종 보고 우선')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[v for v in q.splitlines() if v.strip() and int(v.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 사용자의 GPU compute 작업 감지: '+str(other))
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 guard();torch.set_num_threads(2);cv2.setNumThreads(1)
 reference=json.loads((PREV/'numeric_results.json').read_text());manifest=reference['source_manifest'];assert len(manifest)==85
 bad=[v['path'] for v in manifest if not (DATA/v['path']).is_file() or hashfile(DATA/v['path'])!=v['sha256']];assert not bad,('원본 해시 변경',bad)
 params=json.loads((DATA/'camera_params.json').read_text());K=np.array(params['camera_matrix'],dtype=np.float64);dist=np.array(params['distortion_coefficients'],dtype=np.float64);gt=json.loads((DATA/'scene_gt_obj04.json').read_text());cam=json.loads((DATA/'scene_camera_obj04.json').read_text());ids=sorted(int(x) for x in gt);images=sorted(int(x.stem) for x in (DATA/'image').glob('*.jpg'));und=sorted(int(x.stem) for x in (DATA/'image_undistorted').glob('*.jpg'));npys=sorted(int(x.stem.split('_')[-1]) for x in (DATA/'poses_obj04').glob('*.npy'));assert len(ids)==27 and ids==images==und==npys==sorted(map(int,cam))
 sourcecad=DATA/'obj_000004.ply';bascad=BASE/'data/models/obj_000004.ply';assert hashfile(sourcecad)==hashfile(bascad)
 uvall=[];orient=[]
 for frame in ids:
  assert len(gt[str(frame)])==1 and gt[str(frame)][0]['obj_id']==4
  entry=gt[str(frame)][0];r=np.array(entry['cam_R_m2c']).reshape(3,3);t=np.array(entry['cam_t_m2c']);matrix=np.load(DATA/'poses_obj04'/f'pose_{frame:06d}.npy',allow_pickle=False);assert matrix.shape==(4,4) and np.isfinite(matrix).all()
  assert np.max(np.abs(r-matrix[:3,:3]))<1e-10 and np.max(np.abs(t-matrix[:3,3]))<1e-10 and np.max(np.abs(matrix[3]-[0,0,0,1]))<1e-10
  assert np.max(abs([REDACTED_EMAIL](3)))<1e-9 and abs(np.linalg.det(r)-1)<1e-9 and np.max(abs(np.array(cam[str(frame)]['cam_K']).reshape(3,3)-K))<1e-9
  xc=D.V@r.T+t;assert np.min(xc[:,2])>0
  uv=D.project(D.V,r,t,K);low=uv.min(axis=0);high=uv.max(axis=0);assert np.isfinite(uv).all() and (low>=0).all() and high[0]<1280 and high[1]<720
  bbox=np.r_[low,high-low];box=D.PAD(bbox,1.5);uvall.append({'frame':frame,'bbox_gt_cad_xywh':bbox.tolist(),'bbox_padded_xywh':box.tolist(),'min_camera_z':float(xc[:,2].min()),'R':r,'t':t});orient.append(frame)
 previous=BASE/'runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt';cksha=hashfile(previous)
 camera=json.loads((ROOT/'checks/camera_check.json').read_text());assert len(camera['frames'])==27 and max(f['remap_vs_existing_und_MAE'] for f in camera['frames'])<1.0
 write(ROOT/'checks/stage1_preflight.json',{'status':'PASS','frames':ids,'source_manifest_files_checked':85,'image_to_json_pose_frame_sets_equal':True,'GT_equals_NPY':True,'K_scene_equals_raw_calibration':True,'CAD_sha_matches_training_copy':True,'positive_z_and_projected_bbox_on_screen':True,'controlled_undistortion_K':K.tolist(),'D':dist.tolist(),'controlled_output_explanation':'기존 undistorted 영상의 newK 이력 확정 대신 RAW로부터 K,D,newK=K를 명시해 생성','existing_undistorted_comparison':'mean MAE {:.6f}, max frame MAE {:.6f}, 8-bit intensity (not geometric certification)'.format(camera['remap_mae_mean'],max(f['remap_vs_existing_und_MAE'] for f in camera['frames'])),'GT_geometric_precision_limit':'이전 육안 coarse 정합성; 066/068/070/072/073 판단불가, 독립 정밀 검증은 미완료','physical_unit':'CAD와 t는 동일 수치 단위로 가정; 재투영으로 절대 물리 단위 확정 불가','ROI':'GT CAD vertex projection AABB then square pad 1.5; fixed deterministic oracle','checkpoint_sha256':cksha})
 imdir=ROOT/'stage1/controlled_rgb';ovdir=ROOT/'stage1/overlays';imdir.mkdir();ovdir.mkdir()
 mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,K,(1280,720),cv2.CV_32FC1)
 info=D.INFO;minn=torch.tensor(D.LO,dtype=torch.float32,device='cuda');size=torch.tensor(D.SIZE,dtype=torch.float32,device='cuda');mod=D.S.explicit_module();net=D.S.build_network(mod,minn,size).cuda();ck=torch.load(previous,map_location='cpu',weights_only=True);assert ck['epoch']==155;net.load_state_dict(ck['model_state_dict'],strict=True);net.eval();assert hashfile(previous)==cksha
 mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None];samples=[]
 for item in uvall:
  frame=item['frame'];im=cv2.imread(str(DATA/'image'/f'{frame:06d}.jpg'));assert im is not None
  im=cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);assert cv2.imwrite(str(imdir/f'{frame:06d}.png'),im)
  rgb=D.CROP(im,np.array(item['bbox_padded_xywh']),256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255;samples.append(((x-mean)/std,item,im))
 allrows=[];D.RECS.clear()
 for at in range(0,len(samples),4):
  guard();group=samples[at:at+4];x=torch.stack([a[0] for a in group]).cuda();box=torch.tensor([a[1]['bbox_padded_xywh'] for a in group],dtype=torch.float32,device='cuda')
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
  for j,(_,item,im) in enumerate(group):
   guard();frame=item['frame'];mask=out['pred_mask'][j].cpu().numpy()>0;xyz=out['pred_front_code_obj'][j].float().cpu().numpy();uv=out['coord_2d_image'][j].float().cpu().numpy();row=D.pose('baseline_real',frame,mask,xyz,uv,K,item['R'],item['t']);row['quality_prior']='시각 판단불가' if frame in (66,68,70,72,73) else '시각 거시정합';row['bbox_gt_cad_xywh']=json.dumps(item['bbox_gt_cad_xywh']);row['bbox_padded_xywh']=json.dumps(item['bbox_padded_xywh']);allrows.append(row)
   for color,(rr,tt) in [((0,255,0),(item['R'],item['t']))]+([((255,0,255),(np.array(D.RECS[('baseline_real',frame)]['R']),np.array(D.RECS[('baseline_real',frame)]['t'])))] if row['valid_pose'] else []):
    pp=D.project(D.V,rr,tt,K);polys=[np.clip(pp[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in D.F];cv2.polylines(im,polys,True,color,1,cv2.LINE_AA)
   cap=f'frame {frame:06d}: GT green / predicted magenta | ADD '+(f'{row["add"]:.2f}' if row['valid_pose'] else 'PNP FAILED');cv2.putText(im,cap,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);assert cv2.imwrite(str(ovdir/f'{frame:06d}.png'),im)
   print(json.dumps({'frame':frame,'PnP':row['valid_pose'],'ADD':row['add'],'inlier_ratio':row['inlier_ratio_actual']},ensure_ascii=False),flush=True)
 with (ROOT/'stage1/real_pose_per_frame.csv').open('x',newline='') as f:
  keys=list(dict.fromkeys(k for r in allrows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
 assert [v['frame'] for v in allrows]==ids and len(allrows)==27 and len(list(ovdir.glob('*.png')))==27
 good=[v for v in allrows if v['valid_pose']];n=len(good)
 summary={'status':'PASS','checkpoint_epoch':155,'frames':27,'PnP_valid':n,'PnP_valid_rate':n/27,'ADD_pass':sum(v['add_pass'] for v in allrows),'ADD_rate':sum(v['add_pass'] for v in allrows)/27,'failure_frames':[v['frame'] for v in allrows if not v['valid_pose']],'ADD_fail_frames':[v['frame'] for v in allrows if not v['add_pass']],'inlier_ratio_mean_all_frames':float(np.mean([v['inlier_ratio_actual'] for v in allrows])),'valid_pose_error_stats':{k:{'n':n,'mean':float(np.mean([v[k] for v in good])) if n else None,'median':float(np.median([v[k] for v in good])) if n else None} for k in ['add','rotation_deg','translation']},'model_input':'RAW converted with calibrated K,D to explicit K, no crop/resize before oracle BBox crop','GT_ROI':'oracle CAD projected bbox pad1.5, not YOLO/E2E','real_accuracy_limitation':'GT is developmental and visual-coarse; 5 frames uncertain; absolute unit unverified; no exclusion','time_seconds':time.time(),'checkpoint_sha256':cksha}
 write(ROOT/'stage1/pose_results.json',{'summary':summary,'rows':allrows,'returned_poses':[{'frame':f,**p} for (tag,f),p in D.RECS.items()]});write(ROOT/'checks/stage1_gate.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'checks/stage1_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc()});raise
