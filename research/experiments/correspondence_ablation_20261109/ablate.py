"""GT-free controlled 2D/3D selection; original 3D values never modified."""
import sys,json,csv,hashlib,time,argparse,concurrent.futures
from pathlib import Path
import numpy as np,cv2
P=Path(__file__).resolve().parent;E=P.parent;C=E/'pnp_stability_20261109/full';Q=E/'xyz_correspondence_quality_20261109/full';sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def angle(R,Q):return float(np.degrees(np.arccos(np.clip((np.trace(R@Q.T)-1)/2,-1,1))))
def choose_unique(xyz,decimals):
 if len(xyz)==0:return np.empty(0,dtype=np.int64)
 _,idx=np.unique(np.round(xyz.astype(np.float64),decimals),axis=0,return_index=True);return np.sort(idx)
def choose_grid(p2,box,indices,n):
 if len(indices)==0:return indices
 pos=(p2[indices]-box[:2])/box[2:];cell=np.floor(pos*n).astype(int);cell=np.clip(cell,0,n-1);flat=cell[:,1]*n+cell[:,0];centers=(cell+.5)/n;dist=np.sum((pos-centers)**2,axis=1);order=np.lexsort((indices,dist,flat));seen=set();keep=[]
 for j in order:
  key=int(flat[j])
  if key not in seen:seen.add(key);keep.append(int(indices[j]))
 return np.array(sorted(keep),dtype=np.int64)
def rngseed(frame,cond,name,seed):return int.from_bytes(hashlib.sha256(f'{frame}|{cond}|{name}|{seed}'.encode()).digest()[:8],'little')
def task(spec):
 path,smoke=spec;cv2.setNumThreads(1);z=np.load(path,allow_pickle=False);p2=z['points2d'];p3=z['points3d'];K=z['K'];box=z['bbox'];baseR=z['baseline_R'];baset=z['baseline_t'];frame=int(path.name[:6]);cond=path.name.split('__',1)[1][:-4];v,_,syms,_=symmetry.load();N=len(p2);assert np.array_equal(p2,z['coordinates'][z['mask']].astype('float32')) and np.array_equal(p3,z['front_xyz'][z['mask']].astype('float32')) and np.isfinite(p2).all() and np.isfinite(p3).all();raw=np.arange(N,dtype=np.int64)
 methods={'A_all':raw,'B_dedup3':choose_unique(p3,3),'B_dedup5':choose_unique(p3,5),'B_dedup7':choose_unique(p3,7)}
 for n in [4,8,16]:methods[f'C_grid{n}']=choose_grid(p2,box,raw,n);methods[f'D_dedup5_grid{n}']=choose_grid(p2,box,methods['B_dedup5'],n)
 outputs=[]
 def run(method,target,indices,mode,trial):
  indices=np.asarray(indices,dtype=np.int64);assert len(np.unique(indices))==len(indices) and ((indices>=0)&(indices<N)).all();n=len(indices);rr={'frame':frame,'condition':cond,'scope':'baseline' if cond=='baseline' else 'changed_roi','method':method,'target_method':target,'mode':mode,'trial':trial,'input_total':N,'selected':n,'selected_fraction':n/N if N else 0.,'success':False,'failure':'','raw_ransac_inliers':'','selected_final_inliers':'','selected_final_ratio':'','selected_median_error_px':'','selected_p95_error_px':'','full_final_inliers':'','full_final_ratio':'','full_median_error_px':'','full_p95_error_px':'','rotation_raw_vs_A_deg':'','rotation_sym_vs_A_deg':'','translation_raw_vs_A':'','translation_sym_vs_A':'','time_ms':'','R':'','t':''}
  if n<=4:rr['failure']='insufficient_points_original_solver_requires_gt4';rr['time_ms']=0.;return rr
  p2s=np.ascontiguousarray(p2[indices].astype(np.float32));p3s=np.ascontiguousarray(p3[indices].astype(np.float32));cv2.setRNGSeed(42+frame);start=time.perf_counter()
  try:
   success,rv,tv,rawins=cv2.solvePnPRansac(p3s,p2s,K,None,reprojectionError=1.5,iterationsCount=150,flags=cv2.SOLVEPNP_EPNP);rr['time_ms']=(time.perf_counter()-start)*1000
   if not success or rv is None or tv is None:rr['failure']='ransac_failure';return rr
   R=cv2.Rodrigues(rv)[0];t=tv.reshape(3)
   if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:rr['failure']='invalid_or_behind_camera';return rr
   alluv=cv2.projectPoints(p3,rv,tv,K,None)[0].reshape(-1,2);dist=np.linalg.norm(alluv-p2,axis=1);ds=dist[indices];rr.update(success=True,raw_ransac_inliers=0 if rawins is None else len(rawins),selected_final_inliers=int((ds<1.5).sum()),selected_final_ratio=float(np.mean(ds<1.5)),selected_median_error_px=float(np.median(ds)),selected_p95_error_px=float(np.quantile(ds,.95)),full_final_inliers=int((dist<1.5).sum()),full_final_ratio=float(np.mean(dist<1.5)),full_median_error_px=float(np.median(dist)),full_p95_error_px=float(np.quantile(dist,.95)),R=json.dumps(R.tolist()),t=json.dumps(t.tolist()))
   if baseR.shape==(3,3):
    angles=[angle(R,baseR@S[:3,:3]) for S in syms];idx=int(np.argmin(angles));_,tb=symmetry.compose(baseR,baset,syms[idx]);rr.update(rotation_raw_vs_A_deg=angle(R,baseR),rotation_sym_vs_A_deg=angles[idx],translation_raw_vs_A=float(np.linalg.norm(t-baset)),translation_sym_vs_A=float(np.linalg.norm(t-tb)))
  except cv2.error:rr['failure']='opencv_error';rr['time_ms']=(time.perf_counter()-start)*1000
  return rr
 for name,idx in methods.items():
  x=run(name,name,idx,'original_order',0)
  if name=='A_all':
   assert x['success']==(baseR.shape==(3,3)),(frame,cond,'baseline success mismatch')
   if x['success']:assert np.allclose(np.array(json.loads(x['R'])),baseR,atol=1e-7) and np.allclose(np.array(json.loads(x['t'])),baset,atol=1e-6) and x['full_final_inliers']==len(z['original_recomputed_inliers']),(frame,cond,'baseline pose/inlier mismatch')
  outputs.append(x)
  for rep in range(3):
   order=np.random.default_rng(rngseed(frame,cond,name,rep)).permutation(idx);outputs.append(run(name,name,order,'shuffled_order',rep))
 for target in ['B_dedup5','C_grid4','C_grid8','C_grid16','D_dedup5_grid4','D_dedup5_grid8','D_dedup5_grid16']:
  k=len(methods[target])
  for rep in range(5):
   idx=np.sort(np.random.default_rng(rngseed(frame,cond,target,rep+100)).choice(N,size=k,replace=False));assert len(idx)==k;outputs.append(run('E_random_matched',target,idx,'random_subset',rep))
 return outputs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();out=P/('smoke' if a.smoke else 'full');out.mkdir(exist_ok=False)
 if not a.smoke:assert json.loads((P/'smoke/check.json').read_text())['status']=='PASS'
 manifest=json.loads((Q/'manifest.json').read_text());assert manifest['status']=='PASS';allhash=manifest['source_sha256'];assert all(sha(Path(n))==v for n,v in allhash.items()),'Phase8 source hash mismatch';inv=rows(C/'cache_inventory.csv');assert len(inv)==104
 if a.smoke:inv=[x for x in inv if (int(x['frame']),x['condition']) in {(32,'baseline'),(66,'baseline'),(68,'baseline'),(66,'scale_0.85')}]
 assert len(inv)==(4 if a.smoke else 104);files=[C/'cache'/x['path'] for x in inv];result=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs={ex.submit(task,(p,a.smoke)):p for p in files}
  for f in concurrent.futures.as_completed(fs):result.extend(f.result());print('DONE',fs[f].name,flush=True)
 result.sort(key=lambda x:(x['frame'],x['condition'],x['method'],x['target_method'],x['mode'],x['trial']));assert len(result)==len(inv)*75
 with (out/'per_trial.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 assert all(sha(Path(n))==v for n,v in allhash.items()),'source mutated';check={'status':'PASS','cases':len(inv),'trials':len(result),'baseline_old_pnp':'PASS success, R/t and final inliers','cache_hashes_match_phase8':True,'source_hashes_unchanged':True,'sampling':'dedup first original index; grid closest cell centre tie original index; random independent deterministic numpy subsets, retain original order','pnp':'OpenCV EPNP 1.5 px 150 iterations; setRNGSeed(42+frame); no claim OpenCV uses caller RNG','ordering':'3 deterministic per-method shuffles; seed depends on frame, ROI, method, trial','repeats':'5 random matched subsets per target per case, not independent images','no_GT':True};json.dump(check,(out/'check.json').open('x'),indent=2);print('PASS',len(inv),len(result),flush=True)
if __name__=='__main__':main()
