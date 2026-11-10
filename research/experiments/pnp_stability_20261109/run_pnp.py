"""Repeat OpenCV EPNP on fixed cached correspondences, independent of GT."""
import sys,json,csv,time,argparse,itertools,concurrent.futures,hashlib
from pathlib import Path
import numpy as np,cv2
P=Path(__file__).resolve().parent;E=P.parent;sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry

def angle(R,Q):return float(np.degrees(np.arccos(np.clip((np.trace(R@Q.T)-1)/2,-1,1))))
def drift(R,t,Rref,tref,syms):
 angles=[angle(R,Rref@S[:3,:3]) for S in syms];idx=int(np.argmin(angles));_,tt=symmetry.compose(Rref,tref,syms[idx]);return angle(R,Rref),angles[idx],float(np.linalg.norm(t-tref)),float(np.linalg.norm(t-tt)),idx

def task(arg):
 path,phase,smoke=arg;cv2.setNumThreads(1);v,_,syms,_=symmetry.load();z=np.load(path,allow_pickle=False);p3=z['points3d'];p2=z['points2d'];K=z['K'];refR=z['baseline_R'];reft=z['baseline_t'];refok=refR.shape==(3,3);frame=int(path.name.split('__')[0]);cond=path.name.split('__',1)[1][:-4]
 if smoke:combos=list(itertools.product([1.5,3.0],[150,500],range(3)))
 elif phase=='seed':combos=[(1.5,150,i) for i in range(20)]
 else:combos=list(itertools.product([1.,1.5,2.,3.,5.],[150,500,1000],range(20)))
 rows=[]
 for threshold,iters,seed in combos:
  rr={'frame':frame,'condition':cond,'threshold_px':threshold,'iterations':iters,'seed':seed,'points':len(p3),'success':False,'failure':'','ransac_inliers':'','ransac_inlier_ratio':'','reproject_inliers_setting':'','strict_1p5_inliers':'','strict_1p5_ratio':'','reproj_median_all_px':'','reproj_median_ransac_inlier_px':'','reproj_p95_all_px':'','time_ms':'','R':'','t':'','vs_original_rotation_raw_deg':'','vs_original_rotation_sym_deg':'','vs_original_translation_raw':'','vs_original_translation_sym':'','vs_original_symmetry_index':'','vs_seedref_rotation_raw_deg':'','vs_seedref_rotation_sym_deg':'','vs_seedref_translation_raw':'','vs_seedref_translation_sym':'','seed_reference':''}
  if len(p3)<=4:rr['failure']='insufficient_correspondences';rr['time_ms']=0.;rows.append(rr);continue
  cv2.setRNGSeed(seed);start=time.perf_counter()
  try:
   ok,rv,tv,ins=cv2.solvePnPRansac(p3,p2,K,None,reprojectionError=float(threshold),iterationsCount=int(iters),flags=cv2.SOLVEPNP_EPNP);rr['time_ms']=(time.perf_counter()-start)*1000
   if not ok or rv is None or tv is None:rr['failure']='ransac_returned_failure';rows.append(rr);continue
   R=cv2.Rodrigues(rv)[0];t=tv.reshape(3)
   if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:rr['failure']='invalid_pose_or_negative_depth';rows.append(rr);continue
   uv=cv2.projectPoints(p3,rv,tv,K,None)[0].reshape(-1,2);err=np.linalg.norm(uv-p2,axis=1);ix=ins.reshape(-1) if ins is not None else np.array([],np.int32);rr.update(success=True,ransac_inliers=len(ix),ransac_inlier_ratio=len(ix)/len(p3),reproject_inliers_setting=int((err<threshold).sum()),strict_1p5_inliers=int((err<1.5).sum()),strict_1p5_ratio=float(np.mean(err<1.5)),reproj_median_all_px=float(np.median(err)),reproj_median_ransac_inlier_px=float(np.median(err[ix])) if len(ix) else '',reproj_p95_all_px=float(np.quantile(err,.95)),R=json.dumps(R.tolist()),t=json.dumps(t.tolist()))
   if refok:
    d=drift(R,t,refR,reft,syms);rr.update(vs_original_rotation_raw_deg=d[0],vs_original_rotation_sym_deg=d[1],vs_original_translation_raw=d[2],vs_original_translation_sym=d[3],vs_original_symmetry_index=d[4])
  except cv2.error as ex:rr['time_ms']=(time.perf_counter()-start)*1000;rr['failure']='opencv_error'
  rows.append(rr)
 # Within each setting use first successful seed as reference, report reference seed.
 for thresh,it in sorted(set((x['threshold_px'],x['iterations']) for x in rows)):
  group=[x for x in rows if x['threshold_px']==thresh and x['iterations']==it];first=next((x for x in group if x['success']),None)
  if first:
   rr=np.array(json.loads(first['R']));tt=np.array(json.loads(first['t']))
   for x in group:
    if x['success']:
     d=drift(np.array(json.loads(x['R'])),np.array(json.loads(x['t'])),rr,tt,syms);x.update(vs_seedref_rotation_raw_deg=d[0],vs_seedref_rotation_sym_deg=d[1],vs_seedref_translation_raw=d[2],vs_seedref_translation_sym=d[3],seed_reference=first['seed'])
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['seed','grid_baseline','grid_roi'],default='seed');ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();folder=P/('smoke' if a.smoke else 'full');assert json.loads((folder/'cache_check.json').read_text())['status']=='PASS'
 if not a.smoke:
  assert json.loads((P/'smoke/pnp_check.json').read_text())['status']=='PASS'
  if a.phase=='grid_baseline':assert (P/'full/seed.csv').exists()
  if a.phase=='grid_roi':assert (P/'full/grid_baseline.csv').exists()
 inv=list(csv.DictReader((folder/'cache_inventory.csv').open()));inv=[r for r in inv if a.smoke or a.phase=='seed' or (a.phase=='grid_baseline' and r['condition']=='baseline') or (a.phase=='grid_roi' and r['condition']!='baseline')];paths=[folder/'cache'/r['path'] for r in inv];allrows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  futures={ex.submit(task,(p,a.phase,a.smoke)):p for p in paths}
  for fut in concurrent.futures.as_completed(futures):
   ret=fut.result();allrows.extend(ret);print('DONE',futures[fut].name,'calls',len(ret),flush=True)
 allrows.sort(key=lambda x:(x['frame'],x['condition'],x['threshold_px'],x['iterations'],x['seed']));outfile=folder/('pnp_smoke.csv' if a.smoke else a.phase+'.csv')
 with outfile.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 if a.smoke:
  assert len(allrows)==6*12;assert any(x['failure']=='insufficient_correspondences' for x in allrows);assert any(x['success'] for x in allrows)
  same=task((paths[0],a.phase,True));same2=task((paths[0],a.phase,True));assert all(x['success']==y['success'] and x['R']==y['R'] and x['t']==y['t'] for x,y in zip(same,same2));json.dump({'status':'PASS','calls':len(allrows),'insufficient_input_cases_preserved':True,'fixed_seed_repeat':'identical successes and poses','cached_original_solver_equivalence':'cache_check.json'},(folder/'pnp_check.json').open('x'),indent=2)
 print('PHASE DONE',a.phase,'calls',len(allrows),'successful',sum(x['success'] for x in allrows),flush=True)
if __name__=='__main__':main()
