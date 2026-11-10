import sys,os,csv,json,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P/'full';E=P.parent;sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry
os.environ['MPLCONFIGDIR']=str(P/'.mpl');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def angle(R,Q):return float(np.degrees(np.arccos(np.clip((np.trace(R@Q.T)-1)/2,-1,1))))
data=read(O/'per_trial.csv');assert len(data)==7800;_,_,syms,_=symmetry.load();original=[r for r in data if r['mode']=='original_order'];random=[r for r in data if r['mode']=='random_subset'];order=[r for r in data if r['mode']=='shuffled_order'];assert len(original)==1040 and len(random)==3640 and len(order)==3120
origmap={(r['frame'],r['condition'],r['method']):r for r in original};rngroups=collections.defaultdict(list)
for r in random:rngroups[(r['frame'],r['condition'],r['target_method'])].append(r)
assert all(len(z)==5 for z in rngroups.values());percase=[]
for r in original:
 percase.append({'frame':r['frame'],'condition':r['condition'],'scope':r['scope'],'method':r['method'],'kind':'deterministic','selected':int(r['selected']),'success_fraction':float(r['success']=='True'),'successful_trials':int(r['success']=='True'),'trials':1,'mean_full_final_ratio':float(r['full_final_ratio']) if r['success']=='True' else '', 'mean_full_median_error_px':float(r['full_median_error_px']) if r['success']=='True' else '', 'mean_selected_median_error_px':float(r['selected_median_error_px']) if r['success']=='True' else '', 'mean_time_ms':float(r['time_ms']),'mean_sym_rotation_vs_A_deg':float(r['rotation_sym_vs_A_deg']) if r['rotation_sym_vs_A_deg'] else ''})
for key,z in rngroups.items():
 good=[x for x in z if x['success']=='True'];assert all(x['selected']==origmap[key]['selected'] for x in z);mean=lambda k:float(np.mean([float(x[k]) for x in good if x[k]!=''])) if any(x[k]!='' for x in good) else ''
 percase.append({'frame':key[0],'condition':key[1],'scope':z[0]['scope'],'method':key[2],'kind':'random_matched','selected':int(z[0]['selected']),'success_fraction':len(good)/5,'successful_trials':len(good),'trials':5,'mean_full_final_ratio':mean('full_final_ratio'),'mean_full_median_error_px':mean('full_median_error_px'),'mean_selected_median_error_px':mean('selected_median_error_px'),'mean_time_ms':float(np.mean([float(x['time_ms']) for x in z])),'mean_sym_rotation_vs_A_deg':mean('rotation_sym_vs_A_deg')})
write(O/'per_case_summary.csv',percase)
summary=[]
for scope in ['all','baseline','changed_roi']:
 for method in sorted({r['method'] for r in percase}):
  for kind in ['deterministic','random_matched']:
   z=[r for r in percase if r['method']==method and r['kind']==kind and (scope=='all' or r['scope']==scope)]
   if not z:continue
   good=[r for r in z if r['success_fraction']>0];med=lambda k:float(np.median([float(x[k]) for x in good if x[k]!=''])) if any(x[k]!='' for x in good) else ''
   summary.append({'scope':scope,'method':method,'kind':kind,'cases':len(z),'cases_with_any_success':len(good),'success_fraction_mean':float(np.mean([x['success_fraction'] for x in z])),'median_selected_count':float(np.median([x['selected'] for x in z])),'median_full_inlier_ratio_success_cases':med('mean_full_final_ratio'),'median_full_error_px_success_cases':med('mean_full_median_error_px'),'median_selected_error_px_success_cases':med('mean_selected_median_error_px'),'median_pose_sym_change_deg_success_pairs':med('mean_sym_rotation_vs_A_deg'),'median_pnp_ms':float(np.median([x['mean_time_ms'] for x in z]))})
write(O/'method_summary.csv',summary)
# Deterministic selection vs count-matched random subsets: case is paired only if all 5 random trials and deterministic trial succeed.
paired=[]
for target in ['B_dedup5','C_grid4','C_grid8','C_grid16','D_dedup5_grid4','D_dedup5_grid8','D_dedup5_grid16']:
 for scope in ['all','baseline','changed_roi']:
  choices=[]
  for key,z in rngroups.items():
   if key[2]!=target or (scope!='all' and z[0]['scope']!=scope):continue
   orig=origmap[key];good=[x for x in z if x['success']=='True']
   if orig['success']=='True' and len(good)==5:
    choices.append({'delta':float(orig['full_median_error_px'])-float(np.mean([float(x['full_median_error_px']) for x in good])),'ratio_delta':float(orig['full_final_ratio'])-float(np.mean([float(x['full_final_ratio']) for x in good]))})
  paired.append({'scope':scope,'target':target,'cases_both_all5_success':len(choices),'median_full_error_delta_selected_minus_random_mean_px':float(np.median([x['delta'] for x in choices])) if choices else '', 'median_full_inlier_ratio_delta_selected_minus_random_mean':float(np.median([x['ratio_delta'] for x in choices])) if choices else '', 'fraction_selected_lower_full_error':float(np.mean([x['delta']<0 for x in choices])) if choices else '', 'note':'paired complete successes only, not independent GT or precision'})
write(O/'matched_random_comparison.csv',paired)
# Effect of permuting an identical selected subset; compare paired output matrices to original-order output of same method.
ordergroups=collections.defaultdict(list)
for r in order:ordergroups[(r['frame'],r['condition'],r['method'])].append(r)
orderresults=[]
for method in sorted({x['method'] for x in original}):
 zz=[(k,v) for k,v in ordergroups.items() if k[2]==method];flip=0;rotation=[];translation=[];any_pose=0
 for key,rs in zz:
  a=origmap[key];assert len(rs)==3 and all(x['selected']==a['selected'] for x in rs);flip+=int(any(x['success']!=a['success'] for x in rs));changes=[]
  if a['success']=='True':
   Ra=np.array(json.loads(a['R']));ta=np.array(json.loads(a['t']))
   for x in rs:
    if x['success']=='True':
     R=np.array(json.loads(x['R']));t=np.array(json.loads(x['t']));angles=[angle(R,Ra@S[:3,:3]) for S in syms];j=int(np.argmin(angles));_,tref=symmetry.compose(Ra,ta,syms[j]);changes.append((min(angles),float(np.linalg.norm(t-tref))))
  if changes:
   rotation.append(max(x[0] for x in changes));translation.append(max(x[1] for x in changes));any_pose+=1
 orderresults.append({'method':method,'cases':len(zz),'success_changed_any_of_3':flip,'valid_pose_comparison_cases':any_pose,'max_sym_rotation_median_deg':float(np.median(rotation)) if rotation else '', 'max_sym_rotation_p95_deg':float(np.quantile(rotation,.95)) if rotation else '', 'max_sym_rotation_max_deg':max(rotation) if rotation else '', 'max_sym_translation_median_cad':float(np.median(translation)) if translation else '', 'order_note':'same points, original vs 3 deterministic order permutations; OpenCV RNG seed fixed'})
write(O/'order_summary.csv',orderresults)
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);sel=['A_all','B_dedup5','C_grid4','C_grid8','C_grid16','D_dedup5_grid4','D_dedup5_grid8','D_dedup5_grid16'];vals=[next(x for x in summary if x['scope']=='all' and x['method']==m and x['kind']=='deterministic') for m in sel];axes[0].bar(sel,[x['success_fraction_mean'] for x in vals]);axes[0].set_ylim(0,1);axes[0].set_ylabel('PnP return fraction (104 inputs)');axes[0].set_title('Not accuracy');axes[0].tick_params(axis='x',rotation=45)
for i,m in enumerate(sel[1:]):
 x=next(a for a in paired if a['scope']=='all' and a['target']==m);axes[1].bar(m,x['median_full_error_delta_selected_minus_random_mean_px'] if x['median_full_error_delta_selected_minus_random_mean_px']!='' else 0)
axes[1].axhline(0,color='black',lw=1);axes[1].set_ylabel('full-point median error difference (px)');axes[1].set_title('Selected − random matched (paired successes only)');axes[1].tick_params(axis='x',rotation=45);fig.savefig(O/'ablation_summary.png',dpi=150);plt.close(fig)
precision=[]
for key in sorted({(r['frame'],r['condition']) for r in original}):
 vals=[int(origmap[(key[0],key[1],f'B_dedup{x}')]['selected']) for x in [3,5,7]];precision.append({'frame':key[0],'condition':key[1],'dedup3':vals[0],'dedup5':vals[1],'dedup7':vals[2],'all_counts_equal':len(set(vals))==1})
write(O/'dedup_precision.csv',precision)
meta={'cases':104,'trials':len(data),'deterministic_methods':10,'random_matched_targets':7,'random_subsets_per_target':5,'order_shuffles_per_method':3,'dedup_equal_count_cases':sum(x['all_counts_equal'] for x in precision),'warnings':['GT-free; PnP return is not accuracy','selected-subset error can be biased; all-original-point reprojection always reported','repeated random trials are not independent frames','timings concurrent 4 CPU workers, PnP only','OpenCV random seed control unproven; input permutations explicitly generated']};json.dump(meta,(O/'summary_meta.json').open('x'),indent=2);print('SUMMARY DONE',json.dumps(meta),flush=True)
