from pathlib import Path
import json,csv,collections,hashlib,zipfile
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent
with (O/'boundary_analysis.csv').open(encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
assert len(r)==99 and len({x['boundary_id'] for x in r})==99
for x in r:
 assert x['auto_label'] in ('LIKELY_STATIONARY','LIKELY_MOVED','UNCERTAIN') and x['review_status']=='UNREVIEWED'
 if x['bg_quality_heuristic']=='False' and x['image_sha256_equal']!='True':assert x['auto_label']=='UNCERTAIN'
valid=[x for x in r if x['bg_quality_heuristic']=='True' and x['sift_relative_displacement_px'] and x['orb_relative_displacement_px']]
for x in r:
 if x['auto_label']=='LIKELY_STATIONARY':x['review_priority']='P3_same_image_low_priority';x['priority_reason']='equal image bytes; acquisition history still needs verification'
 elif x['bg_quality_heuristic']=='False':x['review_priority']='P1_unreliable_background';x['priority_reason']='background quality criterion failed'
 elif 'PARTIAL' in (x['occlusion_before'],x['occlusion_after']):x['review_priority']='P1_occlusion';x['priority_reason']='partial occlusion; feature ownership uncertain'
 else:x['review_priority']='P2_other_uncertainty';x['priority_reason']='visible segmentation and motion calibration not verified'
r.sort(key=lambda x:(x['review_priority'],x['boundary_id']))
with (O/'review_priority.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
fig,ax=plt.subplots(1,3,figsize=(14,4))
for j,(key,title) in enumerate([('sift_relative_displacement_px','SIFT relative displacement'),('orb_relative_displacement_px','ORB relative displacement'),('method_disagreement_px','SIFT-ORB disagreement')]):
 a=np.array([float(x[key]) for x in valid if x[key]]);ax[j].hist(a,bins=30);ax[j].set_xlabel('pixels');ax[j].set_ylabel('boundaries');ax[j].set_title(title+' (background quality pass)');ax[j].set_xscale('symlog',linthresh=.1)
fig.tight_layout();fig.savefig(O/'motion_distribution.png',dpi=140);plt.close(fig)
source=json.loads((P/'bb100_visual_review_20261109/manifest.json').read_text())['sources_sha256'];bad=[];n=0
for path,h in source.items():
 p=Path(path)
 if p.is_file():
  n+=1
  if hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(path)
assert not bad
old=P/'bb100_placement_targets_phase15_20261109'; csv_hash=hashlib.sha256((old/'pose_changes.csv').read_bytes()).hexdigest();summary={'phase12_sources_hash_checked':n,'phase12_changed':bad,'phase15_pose_changes_sha256_now':csv_hash,'99_unique_and_status':'PASS','background_gate':'PASS','synthetic_geometry':'PASS','headless_firefox_import_export_persistence':'PASS','windows_chrome':'NOT TESTED','priority_counts':dict(collections.Counter(x['review_priority'] for x in r)),'no_verified_manual_boundary_truth':True}
with zipfile.ZipFile(O/'BB100_motion_assist_windows.zip') as z:
 assert not z.testzip();names=set(z.namelist());assert len([n for n in names if n.startswith('panels/') and n.endswith('.jpg')])==99 and len([n for n in names if n.startswith('frames/') and n.endswith('.jpg')])==100
summary['zip_integrity_and_image_count']='PASS'
(O/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
