import os,csv,json,collections,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;out=P/'full';os.environ['MPLCONFIGDIR']=str(P/'.mpl')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
with (out/'per_frame.csv').open(newline='') as f:rows=list(csv.DictReader(f))
conditions=['scale_0.85','baseline','scale_1.20','scale_1.40','x_-0.10','x_-0.05','x_+0.05','x_+0.10','y_-0.10','y_-0.05','y_+0.05','y_+0.10'];summary=[]
def stats(z,key):
 vals=[float(x[key]) for x in z if x[key]!=''];return {'n':len(vals),'median':float(np.median(vals)) if vals else None,'p95':float(np.quantile(vals,.95)) if vals else None,'max':max(vals) if vals else None}
for c in conditions:
 z=[r for r in rows if r['condition']==c];s={'condition':c,'n':len(z),'pnp_valid':sum(r['pnp_valid']=='True' for r in z),'pnp_failed_frames':[int(r['frame']) for r in z if r['pnp_valid']!='True'],'baseline_invalid_frames':[int(r['frame']) for r in z if r['baseline_pnp_valid']!='True'],'matched_pose_count':sum(bool(r['rotation_sym_deg']) for r in z),'nonidentity_symmetry_count':sum(r['rotation_selected_symmetry_index']=='1' for r in z)}
 for k in ['rotation_raw_deg','rotation_sym_deg','translation_raw_cad_units','translation_sym_aligned_cad_units','xyz_median_cad_units','xyz_common_fraction_baseline_mask','time_total_ms','time_crop_ms','time_network_decode_ms','time_pnp_ms']:s[k]=stats(z,k)
 summary.append(s)
with (out/'condition_summary.json').open('x') as f:json.dump(summary,f,indent=2)
columns=['condition','n','pnp_valid','matched_pose_count','pnp_failed_frames','baseline_invalid_frames','nonidentity_symmetry_count']+[f'{k}_{v}' for k in ['rotation_raw_deg','rotation_sym_deg','translation_raw_cad_units','translation_sym_aligned_cad_units','xyz_median_cad_units','xyz_common_fraction_baseline_mask','time_total_ms'] for v in ['n','median','p95','max']]
with (out/'condition_summary.csv').open('x',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
 for s in summary:
  r={k:s[k] for k in columns if k in s}
  for k in ['rotation_raw_deg','rotation_sym_deg','translation_raw_cad_units','translation_sym_aligned_cad_units','xyz_median_cad_units','xyz_common_fraction_baseline_mask','time_total_ms']:
   for v in ['n','median','p95','max']:r[f'{k}_{v}']=s[k][v]
  w.writerow(r)
frames=[]
for frame in sorted({int(r['frame']) for r in rows}):
 z=[r for r in rows if int(r['frame'])==frame and r['condition']!='baseline'];valid=[r for r in z if r['rotation_sym_deg']!=''];maxrot=max(valid,key=lambda x:float(x['rotation_sym_deg'])) if valid else None;maxt=max(valid,key=lambda x:float(x['translation_sym_aligned_cad_units'])) if valid else None;xyz=[r for r in z if r['xyz_median_cad_units']!=''];maxxyz=max(xyz,key=lambda x:float(x['xyz_median_cad_units'])) if xyz else None
 frames.append({'frame':frame,'baseline_pnp_valid':z[0]['baseline_pnp_valid'],'conditions_pnp_valid':sum(r['pnp_valid']=='True' for r in z),'max_sym_rotation_deg':maxrot['rotation_sym_deg'] if maxrot else '', 'max_sym_rotation_condition':maxrot['condition'] if maxrot else '', 'max_aligned_translation_cad_units':maxt['translation_sym_aligned_cad_units'] if maxt else '','max_aligned_translation_condition':maxt['condition'] if maxt else '','max_xyz_median_cad_units':maxxyz['xyz_median_cad_units'] if maxxyz else '','max_xyz_condition':maxxyz['condition'] if maxxyz else ''})
with (out/'frame_sensitivity.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
# Distribution plots, all scatter points; missing reference/PnP excluded per condition and n reported in CSV.
fig,axes=plt.subplots(3,1,figsize=(13,12),constrained_layout=True)
for ax,key,title in zip(axes,['rotation_sym_deg','translation_sym_aligned_cad_units','xyz_median_cad_units'],['Symmetry-aware rotation change (deg)','Symmetry-aligned translation change (CAD units)','Median shared-mask front XYZ change (CAD units)']):
 vals=[[float(r[key]) for r in rows if r['condition']==c and r[key]!=''] for c in conditions];ax.boxplot(vals,showfliers=True);ax.set_xticks(range(1,len(conditions)+1),conditions,rotation=35,ha='right');ax.set_title(title+' | prediction-to-prediction, not accuracy');ax.grid(axis='y',alpha=.3)
fig.savefig(out/'stability_distributions.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);ax.bar(conditions,[s['pnp_valid'] for s in summary]);ax.set_ylim(0,27);ax.set_ylabel('valid PnP returns / 27');ax.tick_params(axis='x',rotation=35);ax.set_title('PnP return by ROI condition; not accuracy');fig.savefig(out/'pnp_valid.png',dpi=140);plt.close(fig)
print(json.dumps(summary,indent=2));print('sensitive',sorted(frames,key=lambda x:float(x['max_sym_rotation_deg']) if x['max_sym_rotation_deg'] else -1,reverse=True)[:5])
