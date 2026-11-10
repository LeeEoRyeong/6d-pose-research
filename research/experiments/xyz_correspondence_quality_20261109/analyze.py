"""Audit existing PnP correspondences and predicted XYZ without GT."""
import os,sys,json,csv,hashlib,argparse,collections
from pathlib import Path
import cv2,numpy as np
P=Path(__file__).resolve().parent;E=P.parent;SRC=E/'pnp_stability_20261109/full';ROI=E/'roi_prediction_stability_20261109/full';sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def eig(x,dimensions):
 if len(x)<2:return [0.]*dimensions,0,None
 x=x.astype(np.float64);cov=(x-x.mean(0)).T@(x-x.mean(0))/len(x);v=np.maximum(np.linalg.eigvalsh(cov)[::-1],0);rank=int(np.sum(v>v[0]*1e-6)) if v[0]>1e-20 else 0;cond=float(v[0]/v[-1]) if v[-1]>v[0]*1e-12 and v[0]>1e-20 else None;return v.tolist(),rank,cond
def med(x):return float(np.median(x)) if len(x) else ''
def p95(x):return float(np.quantile(x,.95)) if len(x) else ''
def gate(row,rule):
 if row['pnp_valid']!=True:return False
 count=row['point_count'];ins=row['final_inliers'];ratio=row['final_inlier_ratio'];err=row['reproj_median_all_px'];r2=row['cov2_rank'];r3=row['cov3_rank']
 return {'none':True,'points32':count>=32,'inliers20':ins>=20,'ratio10pct':ratio>=.1,'median3px':err<=3.,'spatial_rank':r2>=2 and r3>=2,'combined':count>=32 and ins>=20 and ratio>=.1 and err<=3. and r2>=2 and r3>=2}[rule]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=P/('smoke' if a.smoke else 'full');out.mkdir(exist_ok=False)
 if not a.smoke:assert json.loads((P/'smoke/smoke_check.json').read_text())['status']=='PASS'
 assert json.loads((SRC/'cache_check.json').read_text())['status']=='PASS';inv=read(SRC/'cache_inventory.csv');assert len(inv)==104;prior={(r['frame'],r['condition']):r for r in read(ROI/'per_frame.csv')};selected=[r for r in inv if not a.smoke or (int(r['frame']),r['condition']) in {(32,'baseline'),(66,'baseline'),(66,'scale_0.85'),(68,'baseline')}];assert len(selected)==(4 if a.smoke else 104);paths=[SRC/'cache'/r['path'] for r in inv]+[SRC/'cache_inventory.csv',ROI/'per_frame.csv'];before={str(p):sha(p) for p in paths};v,faces,syms,info=symmetry.load();rows=[];(out/'spatial').mkdir();cv2.setNumThreads(1)
 for r in selected:
  i=int(r['frame']);condition=r['condition'];z=np.load(SRC/'cache'/r['path'],allow_pickle=False);base=np.load(SRC/'cache'/f'{i:06d}__baseline.npz',allow_pickle=False);mask=z['mask'];xyz=z['front_xyz'];uv=z['coordinates'];pts3=z['points3d'];pts2=z['points2d'];K=z['K'];box=z['bbox'];p=prior[(str(i),condition)];assert np.array_equal(pts3,xyz[mask].astype(np.float32)) and np.array_equal(pts2,uv[mask].astype(np.float32));assert len(pts3)==int(r['point_count'])==int(p['mask_pixels']) and np.isfinite(pts2).all() and np.isfinite(pts3).all() and np.allclose(box,np.array(json.loads(p['bbox_xywh'])),atol=1e-10)
  norm2=(pts2-box[:2])/box[2:] if len(pts2) else np.empty((0,2));e2,rank2,cond2=eig(norm2,2);e3,rank3,cond3=eig(pts3,3);uniq3=len(np.unique(np.round(pts3.astype(float),5),axis=0)) if len(pts3) else 0;uniq2=len(np.unique(np.round(pts2.astype(float),5),axis=0)) if len(pts2) else 0;hull2=float(cv2.contourArea(cv2.convexHull(norm2.astype(np.float32)))) if len(norm2)>=3 else 0.
  raw=np.array([],np.int32);final=np.array([],np.int32);errors=np.array([],np.float32);R=None;t=None;ok=False
  if len(pts3)>4:
   cv2.setRNGSeed(42+i)
   try:
    success,rv,tv,ri=cv2.solvePnPRansac(pts3,pts2,K,None,reprojectionError=1.5,iterationsCount=150,flags=cv2.SOLVEPNP_EPNP)
    if success and rv is not None and tv is not None:
     R=cv2.Rodrigues(rv)[0];t=tv.reshape(3);ok=bool(np.isfinite(R).all() and np.isfinite(t).all() and np.min((v@R.T+t)[:,2])>0)
     if ok:
      raw=ri.reshape(-1).astype(np.int32) if ri is not None else raw;pred=cv2.projectPoints(pts3,rv,tv,K,None)[0].reshape(-1,2);errors=np.linalg.norm(pred-pts2,axis=1);final=np.flatnonzero(errors<1.5).astype(np.int32);assert np.array_equal(final,z['original_recomputed_inliers']);assert np.allclose(R,z['baseline_R'],atol=1e-7) and np.allclose(t,z['baseline_t'],atol=1e-6);assert len(raw)==int(r['ransac_raw_inlier_count'])
   except cv2.error:pass
  assert ok==(r['baseline_pnp_valid']=='True'),(i,condition,'solver discrepancy')
  m0=base['mask'];b0=base['bbox'];yx=np.indices(m0.shape,dtype=np.float32);xp=b0[0]+yx[1]*b0[2]/m0.shape[1];yp=b0[1]+yx[0]*b0[3]/m0.shape[0];mx=((xp-box[0])*mask.shape[1]/box[2]).astype(np.float32);my=((yp-box[1])*mask.shape[0]/box[3]).astype(np.float32);ma=cv2.remap(mask.astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;xx=cv2.remap(xyz,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);common=m0&ma;delta=np.full(m0.shape,np.nan,np.float32)
  if common.any():delta[common]=np.linalg.norm(base['front_xyz'][common]-xx[common],axis=1)
  old=p['xyz_median_cad_units'];xm=med(delta[common]);assert (old=='' and xm=='') or (old!='' and xm!='' and abs(float(old)-xm)<1e-5)
  np.savez_compressed(out/'spatial'/r['path'],delta_xyz=delta,common_mask=common,raw_inlier_indices=raw,final_inlier_indices=final,reprojection_error_px=errors)
  span=np.ptp(pts3,axis=0) if len(pts3) else np.full(3,np.nan);ratio=mask.sum()/max(1,int(m0.sum()));row={'frame':i,'condition':condition,'point_count':len(pts3),'baseline_mask_points':int(m0.sum()),'mask_size_ratio_baseline_grid':float(ratio),'mask_shrunk_below_half':bool(ratio<.5),'mask_near_disappearance_le4':bool(len(pts3)<=4),'pnp_valid':ok,'xyz_span_x':float(span[0]) if len(pts3) else '', 'xyz_span_y':float(span[1]) if len(pts3) else '', 'xyz_span_z':float(span[2]) if len(pts3) else '', 'duplicate_2d_fraction_round_1e5':1-uniq2/len(pts2) if len(pts2) else '', 'duplicate_3d_fraction_round_1e5':1-uniq3/len(pts3) if len(pts3) else '', 'cov2_eig1_roi2':e2[0],'cov2_eig2_roi2':e2[1],'cov2_rank':rank2,'cov2_condition':cond2 if cond2 is not None else '', 'cov2_hull_area_fraction':hull2,'cov3_eig1_cad2':e3[0],'cov3_eig2_cad2':e3[1],'cov3_eig3_cad2':e3[2],'cov3_rank':rank3,'cov3_condition':cond3 if cond3 is not None else '', 'cov3_planarity_ratio_e3_e1':e3[2]/e3[0] if e3[0]>1e-20 else '', 'cov3_linearity_ratio_e2_e1':e3[1]/e3[0] if e3[0]>1e-20 else '', 'raw_ransac_inliers':len(raw) if ok else '', 'raw_ransac_ratio':len(raw)/len(pts3) if ok else '', 'final_inliers':len(final) if ok else '', 'final_inlier_ratio':len(final)/len(pts3) if ok else '', 'reproj_median_all_px':med(errors),'reproj_p95_all_px':p95(errors),'reproj_median_final_inlier_px':med(errors[final]) if ok else '', 'common_pixels':int(common.sum()),'common_fraction_baseline':float(common.sum()/max(1,int(m0.sum()))),'xyz_delta_median_cad':xm,'xyz_delta_p95_cad':p95(delta[common]),'cache_path':r['path']}
  for rule in ['none','points32','inliers20','ratio10pct','median3px','spatial_rank','combined']:row['gate_'+rule]=gate(row,rule)
  rows.append(row);print('AUDIT',i,condition,'points',len(pts3),'pnp',ok,flush=True)
 write(out/'quality_per_case.csv',rows);assert before=={p:sha(Path(p)) for p in before};json.dump({'status':'PASS','count':len(rows),'source_file_count':len(before),'source_sha256':before,'checks':['npz integrity and finite arrays','mask-indexed points exact','prior ROI bbox and mask count','same original PnP seed, return, R/t, raw count, final inlier indices','XYZ overlap median equals prior','cache hashes before and after'],'caveat':'hash stability during this run; no prior NPZ hash available, full prior model inference not repeated','uses_ground_truth':False},(out/'smoke_check.json' if a.smoke else out/'manifest.json').open('x'),indent=2);print('PASS',len(rows))
if __name__=='__main__':main()
