import os,csv,json,collections,statistics
from pathlib import Path
import numpy as np,cv2
P=Path(__file__).resolve().parent;O=P/'full';E=P.parent;os.environ['MPLCONFIGDIR']=str(P/'.mpl')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
seed=read(O/'seed.csv');base=read(O/'grid_baseline.csv');roi=read(O/'grid_roi.csv');assert len(seed)==2080 and len(base)==8100 and len(roi)==23100;allgrid=base+roi;groups=collections.defaultdict(list)
for r in allgrid:groups[(r['frame'],r['condition'],r['threshold_px'],r['iterations'])].append(r)
assert len(groups)==104*15 and all(len(v)==20 for v in groups.values());seedgroup=collections.defaultdict(list)
for r in seed:seedgroup[(r['frame'],r['condition'])].append(r)
assert len(seedgroup)==104
repeat_mismatch=[];changed_seed_groups=[]
for (f,c,th,it),z in groups.items():
 if len({(r['success'],r['R'],r['t']) for r in z})!=1:changed_seed_groups.append((f,c,th,it))
 if float(th)==1.5 and int(it)==150:
  old={r['seed']:(r['success'],r['R'],r['t']) for r in seedgroup[(f,c)]}
  if any(old[r['seed']]!=(r['success'],r['R'],r['t']) for r in z):repeat_mismatch.append((f,c))
assert not repeat_mismatch,'original setting repeat mismatch';print('seed varying groups',len(changed_seed_groups),'of',len(groups),'seed-vs-grid mismatch',len(repeat_mismatch),flush=True)
# Scores aggregate once per frame-condition, not 20 seeds as independent samples. Time is median of repeated calls.
metrics=[]
for scope in ['baseline','changed_roi']:
 for th in [1.,1.5,2.,3.,5.]:
  for it in [150,500,1000]:
   gg=[(k,z) for k,z in groups.items() if float(k[2])==th and int(k[3])==it and ((k[1]=='baseline')==(scope=='baseline'))];case=[z[0] for _,z in gg];ok=[r for r in case if r['success']=='True'];orig=[r for r in ok if r['vs_original_rotation_sym_deg']!=''];
   def med(k,rows):
    v=[float(r[k]) for r in rows if r[k]!=''];return float(np.median(v)) if v else ''
   def p95(k,rows):
    v=[float(r[k]) for r in rows if r[k]!=''];return float(np.quantile(v,.95)) if v else ''
   metrics.append({'scope':scope,'threshold_px':th,'iterations':it,'unique_inputs':len(case),'returned':len(ok),'return_rate':len(ok)/len(case),'median_raw_inlier_count':med('ransac_inliers',ok),'median_raw_inlier_ratio':med('ransac_inlier_ratio',ok),'median_strict_1p5_ratio':med('strict_1p5_ratio',ok),'median_all_reprojection_px':med('reproj_median_all_px',ok),'median_inlier_reprojection_px':med('reproj_median_ransac_inlier_px',ok),'matched_original_valid':len(orig),'median_sym_rotation_vs_original_deg':med('vs_original_rotation_sym_deg',orig),'p95_sym_rotation_vs_original_deg':p95('vs_original_rotation_sym_deg',orig),'median_sym_translation_vs_original':med('vs_original_translation_sym',orig),'p95_sym_translation_vs_original':p95('vs_original_translation_sym',orig),'median_pnp_ms_per_input':float(np.median([np.median([float(r['time_ms']) for r in z]) for _,z in gg if z[0]['failure']!='insufficient_correspondences']))})
write(O/'setting_summary.csv',metrics)
# crop and XYZ comparisons are prediction-to-prediction, pixel-aligned same convention as ROI analysis.
inv=read(O/'cache_inventory.csv');prior={(r['frame'],r['condition']):r for r in read(E/'roi_prediction_stability_20261109/full/per_frame.csv')};diagnostics=[]
for r in inv:
 f=r['frame'];name=r['condition'];z=np.load(O/'cache'/r['path']);b=np.load(O/'cache'/f'{int(f):06d}__baseline.npz');m0=b['mask'];m1=z['mask'];box0=b['bbox'];box=z['bbox'];grid=np.indices(m0.shape,dtype=np.float32);xp=box0[0]+grid[1]*box0[2]/m0.shape[1];yp=box0[1]+grid[0]*box0[3]/m0.shape[0];mx=((xp-box[0])*m1.shape[1]/box[2]).astype(np.float32);my=((yp-box[1])*m1.shape[0]/box[3]).astype(np.float32);mm=cv2.remap(m1.astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;xyz=cv2.remap(z['front_xyz'],mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);valid=m0&mm;cnt=int(valid.sum());med=float(np.median(np.linalg.norm(b['front_xyz'][valid]-xyz[valid],axis=1))) if cnt else '';old=prior[(f,name)]['xyz_median_cad_units'];assert (med=='' and old=='') or (med!='' and old!='' and abs(med-float(old))<1e-5),(f,name,med,old)
 cropdiff=float(np.mean(np.abs(z['crop_bgr'].astype('float32')-b['crop_bgr'].astype('float32'))));diagnostics.append({'frame':f,'condition':name,'points':r['point_count'],'baseline_points':int(b['mask'].sum()),'crop_mean_absolute_pixel_change_0_255':cropdiff,'xyz_common_points':cnt,'xyz_common_fraction_of_baseline_mask':cnt/max(1,int(m0.sum())),'xyz_change_median_cad_units':med,'original_pnp_valid':r['baseline_pnp_valid'],'original_reprojected_inliers_1p5px':r['original_recomputed_inlier_count'],'original_reproj_median_all_px':r['median_reprojection_error_all_px'],'interpretation':'crop image difference is coordinate-dependent; no GT accuracy'})
write(O/'input_diagnostics.csv',diagnostics)
seedstats=[]
for (f,c),z in seedgroup.items():
 ok=[r for r in z if r['success']=='True'];rot=[float(r['vs_seedref_rotation_sym_deg']) for r in ok];trans=[float(r['vs_seedref_translation_sym']) for r in ok];seedstats.append({'frame':int(f),'condition':c,'attempts':20,'successful':len(ok),'unique_success_pose_tuples':len({(r['R'],r['t']) for r in ok}),'seed_reference':ok[0]['seed_reference'] if ok else '','max_rotation_sym_vs_first_success_deg':max(rot) if rot else '', 'max_translation_sym_vs_first_success':max(trans) if trans else '', 'seed_effect_observed':len({(r['success'],r['R'],r['t']) for r in z})>1})
write(O/'seed_summary.csv',seedstats)
# Quantified plot: unique input counts by config, no seed pseudo-replication.
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,scope in zip(axes,['baseline','changed_roi']):
 for it in [150,500,1000]:
  z=[x for x in metrics if x['scope']==scope and x['iterations']==it];ax.plot([x['threshold_px'] for x in z],[x['return_rate'] for x in z],marker='o',label=f'{it} iters')
 ax.set_title(scope+' (independent input units; not accuracy)');ax.set_xlabel('RANSAC reprojection threshold (px)');ax.set_ylabel('PnP valid return rate');ax.set_ylim(0,1.05);ax.grid(alpha=.3);ax.legend()
fig.savefig(O/'ransac_return_rate.png',dpi=150);plt.close(fig)
# Inference/correspondence masks and normalized predicted XYZ, no GT.
fig,axs=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
li=json.loads((E/'obj04_seminar_v1/data/models/models_info.json').read_text())['4'];lo=np.array([li['min_'+a] for a in 'xyz']);sz=np.array([li['size_'+a] for a in 'xyz'])
for j,(f,c) in enumerate([(66,'baseline'),(66,'scale_0.85'),(66,'scale_1.20')]):
 z=np.load(O/'cache'/f'{f:06d}__{c}.npz');mask=z['mask'];xyz=np.clip((z['front_xyz']-lo)/sz,0,1);xyz[~mask]=0;axs[j,0].imshow(cv2.cvtColor(z['crop_bgr'],cv2.COLOR_BGR2RGB));axs[j,1].imshow(mask,cmap='gray');axs[j,2].imshow(xyz);axs[j,0].set_ylabel(f'{f} {c}\n{mask.sum()} points');
 for x in axs[j]:x.set_xticks([]);x.set_yticks([])
for j,t in enumerate(['ROI crop','Predicted mask','Predicted front XYZ (normalized)']):axs[0,j].set_title(t)
fig.savefig(O/'frame66_crop_mask_xyz.png',dpi=140);plt.close(fig)
meta={'seed_groups':len(seedgroup),'original_seed_groups_unchanged':sum(len({(r['success'],r['R'],r['t']) for r in v})==1 for v in seedgroup.values()),'all_setting_seed_groups':len(groups),'setting_seed_groups_with_any_change':len(changed_seed_groups),'setting_1p5_150_vs_seed_phase_mismatch':len(repeat_mismatch),'unique_input_counts':{'baseline':27,'changed_roi':77},'calls_seed':len(seed),'calls_grid':len(allgrid),'prediction_xyz_cache_matches_prior_median':True,'causal_note':'20 nominal seeds produced identical outputs here; independence of RANSAC samples is not established','timing_note':'4 CPU workers concurrently; solvePnPRansac wall time only, excludes reprojection; not isolated benchmark'}
json.dump(meta,(O/'summary_meta.json').open('x'),indent=2);print(json.dumps(meta,indent=2),flush=True)
