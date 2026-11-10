import os,sys,csv,json,collections
from pathlib import Path
import numpy as np,cv2
P=Path(__file__).resolve().parent;O=P/'full';E=P.parent;SRC=E/'pnp_stability_20261109/full';DEV=E/'obj04_seminar_v1/agent_run_v1/stage1/controlled_rgb';os.environ['MPLCONFIGDIR']=str(P/'.mpl');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=read(O/'quality_per_case.csv');assert len(r)==104;pnp=[x for x in r if x['pnp_valid']=='True'];failed=[x for x in r if x['pnp_valid']=='False'];assert len(pnp)==83 and len(failed)==21
rules={'none':'PnP success only','points32':'points ≥32','inliers20':'final inliers ≥20','ratio10pct':'final inlier ratio ≥0.10','median3px':'all-point median reproj ≤3px','spatial_rank':'2D rank≥2 and 3D rank≥2','combined':'all previous gates'};g=[]
for key,label in rules.items():
 accepted=[x for x in r if x['gate_'+key]=='True'];g.append({'gate':key,'rule':label,'total_inputs':len(r),'original_pnp_returned':len(pnp),'accepted':len(accepted),'rejected_among_returned':len(pnp)-len(accepted),'acceptance_all':len(accepted)/len(r),'acceptance_among_returned':len(accepted)/len(pnp),'interpretation':'diagnostic thresholds only; no GT accuracy labels'})
write(O/'gate_summary.csv',g)
case=[]
for x in r:
 if x['pnp_valid']=='True' and (int(x['final_inliers'])<20 or float(x['reproj_median_all_px'])>3):
  case.append({'frame':x['frame'],'condition':x['condition'],'points':x['point_count'],'raw_inliers':x['raw_ransac_inliers'],'final_inliers':x['final_inliers'],'final_ratio':x['final_inlier_ratio'],'reproj_median_px':x['reproj_median_all_px'],'reason':'returned but poor fixed-threshold reprojection consistency; not GT accuracy'})
write(O/'returned_low_consistency.csv',case)
fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
for q,label,col in [(pnp,'PnP returned','tab:blue'),(failed,'PnP failed','tab:red')]:
 axes[0].scatter([int(x['point_count']) for x in q],[float(x['cov2_hull_area_fraction']) for x in q],label=label,c=col,alpha=.6);axes[1].scatter([int(x['point_count']) for x in q],[float(x['duplicate_3d_fraction_round_1e5']) if x['duplicate_3d_fraction_round_1e5'] else 0 for x in q],label=label,c=col,alpha=.6)
axes[0].set(xlabel='correspondence count',ylabel='2D convex hull area / ROI area',title='Image-space spread');axes[1].set(xlabel='correspondence count',ylabel='duplicate predicted XYZ fraction',title='XYZ duplicates (rounded 1e-5 CAD units)')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.savefig(O/'geometry_distribution.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.2),constrained_layout=True);ax.bar([x['gate'] for x in g],[x['accepted'] for x in g]);ax.axhline(len(pnp),color='tab:red',linestyle='--',label='PnP returned: 83');ax.set(ylabel='accepted cases / 104',title='Illustrative diagnostic gates — not accuracy or operational policy');ax.tick_params(axis='x',rotation=25);ax.legend();fig.savefig(O/'gate_counts.png',dpi=150);plt.close(fig)
# Frame-specific views use predicted ROI and mask only; no GT overlay.
li=json.loads((E/'obj04_seminar_v1/data/models/models_info.json').read_text())['4'];lo=np.array([li['min_'+x] for x in 'xyz']);size=np.array([li['size_'+x] for x in 'xyz']);(O/'visualizations').mkdir()
for frame,condition in [(66,'baseline'),(66,'scale_0.85'),(68,'baseline')]:
 name=f'{frame:06d}__{condition}.npz';z=np.load(SRC/'cache'/name);zz=np.load(O/'spatial'/name);im=cv2.imread(str(DEV/f'{frame:06d}.png'));im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);box=z['bbox'];mask=z['mask'];xyz=z['front_xyz'];points=z['points2d'];p3=z['points3d'];raw=zz['raw_inlier_indices'];final=zz['final_inlier_indices'];err=zz['reprojection_error_px'];fig=plt.figure(figsize=(15,9),constrained_layout=True);axes=[fig.add_subplot(2,3,i+1,projection='3d' if i==5 else None) for i in range(6)];axes[0].imshow(im);axes[0].add_patch(plt.Rectangle(box[:2],box[2],box[3],fill=False,color='yellow',lw=2));axes[0].set_title('Original controlled RGB + predicted ROI');axes[1].imshow(mask,cmap='gray');axes[1].set_title(f'Predicted mask: {len(points)} points');scaled=np.clip((xyz-lo)/size,0,1);scaled[~mask]=0;axes[2].imshow(scaled);axes[2].set_title('Predicted XYZ, CAD normalized')
 if len(points):axes[3].scatter(points[:,0],points[:,1],s=3,c='gray',alpha=.3,label='all points')
 if len(raw):axes[3].scatter(points[raw,0],points[raw,1],s=24,facecolors='none',edgecolors='orange',label='RANSAC raw')
 if len(final):axes[3].scatter(points[final,0],points[final,1],s=5,c='cyan',label='final reproj <1.5px')
 axes[3].invert_yaxis();axes[3].set_title(f'2D points; raw={len(raw)}, final={len(final)}' if len(err) else '2D points; inliers N/A (PnP failed)');axes[3].legend(loc='best',fontsize=8)
 if len(err):axes[4].hist(err,bins=40,color='steelblue');axes[4].axvline(1.5,color='red',label='1.5 px');axes[4].legend();axes[4].set_title('Final pose: all-point reprojection errors');axes[4].set_xlabel('px')
 else:axes[4].text(.5,.5,'PnP failed / insufficient points\nno pose, no residuals',ha='center',va='center',transform=axes[4].transAxes);axes[4].set_axis_off()
 if len(p3):axes[5].scatter(p3[:,0],p3[:,1],p3[:,2],s=3,c=p3[:,2],cmap='viridis');axes[5].set_title('Predicted 3D correspondences');axes[5].set_xlabel('X');axes[5].set_ylabel('Y');axes[5].set_zlabel('Z')
 fig.suptitle(f'Frame {frame:03d} / {condition}: predicted input only, no GT');fig.savefig(O/'visualizations'/f'{frame:06d}_{condition}.png',dpi=120);plt.close(fig)
# Pixel-space XYZ change maps, NaN means no shared predicted mask.
for frame,condition in [(66,'scale_1.20'),(68,'scale_1.40'),(32,'scale_1.40'),(92,'y_-0.10'),(94,'scale_1.40'),(96,'y_-0.10')]:
 name=f'{frame:06d}__{condition}.npz';z=np.load(O/'spatial'/name);delta=z['delta_xyz'];fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True);im=ax.imshow(np.ma.masked_invalid(delta),vmin=0,vmax=100,cmap='magma');ax.set_title(f'{frame} {condition}: XYZ difference, shared predicted mask');fig.colorbar(im,ax=ax,label='CAD units; clipped at 100');fig.savefig(O/'visualizations'/f'{frame:06d}_{condition}_xyz_delta.png',dpi=130);plt.close(fig)
meta={'total':len(r),'returned':len(pnp),'failed':len(failed),'failed_more_than_4_points':sum(int(x['point_count'])>4 for x in failed),'failed_at_least_100_points':sum(int(x['point_count'])>=100 for x in failed),'failed_2d_rank2':sum(int(x['cov2_rank'])==2 for x in failed),'failed_3d_rank3':sum(int(x['cov3_rank'])==3 for x in failed),'returned_with_final_inliers_less_than20':sum(int(x['final_inliers'])<20 for x in pnp),'mask_relative_half_cases':sum(x['mask_shrunk_below_half']=='True' for x in r),'mask_le4_cases':sum(x['mask_near_disappearance_le4']=='True' for x in r),'xyz_p95_top_case':max((x for x in r if x['xyz_delta_p95_cad']),key=lambda x:float(x['xyz_delta_p95_cad'])),'selection_caveat':'27 baseline plus 77 selected anomalous-frame ROI cases; non-independent development analysis','gate_caveat':'unvalidated diagnostic acceptance; no pose GT labels'};json.dump(meta,(O/'summary.json').open('x'),indent=2);print('SUMMARY',meta,flush=True)
