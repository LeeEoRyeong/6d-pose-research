from pathlib import Path
import cv2,numpy as np,json,hashlib,datetime
R=Path(__file__).resolve().parents[1];D=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer');B=R.parent
params=json.loads((D/'camera_params.json').read_text());K=np.array(params['camera_matrix'],dtype=np.float64);dist=np.array(params['distortion_coefficients'],dtype=np.float64);size=(1280,720)
mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,K,size,cv2.CV_32FC1)
# Real raw→controlled undistortion uses explicitly selected same K, no crop/resize.
valid=(mapx>=1)&(mapx<size[0]-2)&(mapy>=1)&(mapy<size[1]-2)
rows=[]
for p in sorted((D/'image').glob('*.jpg')):
 frame=int(p.stem);raw=cv2.imread(str(p));u=cv2.imread(str(D/'image_undistorted'/p.name));assert raw is not None and u is not None and raw.shape==u.shape==(720,1280,3)
 remap=cv2.remap(raw,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);dif=np.abs(remap.astype(np.float32)-u.astype(np.float32));baseline=np.abs(raw.astype(np.float32)-u.astype(np.float32));
 rows.append({'frame':frame,'valid_pixels':int(valid.sum()),'remap_vs_existing_und_MAE':float(dif[valid].mean()),'raw_vs_existing_und_MAE':float(baseline[valid].mean()),'remap_vs_existing_und_p95':float(np.quantile(dif[valid],.95))})
out={'K':K.tolist(),'D':dist.tolist(),'D_model_assumption':'OpenCV Brown-Conrady k1,k2,p1,p2,k3','output_K_explicit':K.tolist(),'output_size':size,'output_crop':False,'output_resize':False,'method':'raw JPEG -> cv2.initUndistortRectifyMap(K,D,None,K) -> cv2.remap INTER_LINEAR','existing_undistorted_provenance':'not claimed; pixels compared separately','frames':rows,'remap_mae_mean':float(np.mean([v['remap_vs_existing_und_MAE'] for v in rows])),'raw_mae_mean':float(np.mean([v['raw_vs_existing_und_MAE'] for v in rows]))}
with (R/'checks/camera_check.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps({k:v for k,v in out.items() if k!='frames'},ensure_ascii=False,indent=2))
