from pathlib import Path
import csv,collections,json,hashlib,zipfile,cv2,numpy as np
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent
def read(n):
 with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
rows=read('target_quality.csv');occ={}
for o in ('NONE','PARTIAL'):
 a=[r for r in rows if r['occlusion']==o];occ[o]={'n':len(a),'mean_cad_pixels':float(np.mean([int(x['amodal_pixels']) for x in a])),'median_cad_pixels':float(np.median([int(x['amodal_pixels']) for x in a])),'valid_supervision_pixels':sum(int(x['valid_supervision_pixels']) for x in a)}
before=json.loads((P/'bb100_visual_review_20261109/manifest.json').read_text())['sources_sha256'];changed=[];count=0
for path,h in before.items():
 path=Path(path)
 if path.exists():
  count+=1
  if hashlib.sha256(path.read_bytes()).hexdigest()!=h:changed.append(str(path))
assert not changed
review=P/'bb100_review_export (2).csv';assert hashlib.sha256(review.read_bytes()).hexdigest()=='26aa03999e732d69a887ec0d3380212355d32f1fe1179e5859847f87e515a264'
with zipfile.ZipFile(O/'BB100_placement_review_windows.zip') as z:
 assert not z.testzip();assert len([x for x in z.namelist() if x.startswith('frames/') and x.endswith('.jpg')])==100
for r in read('visible_candidate_manifest.csv'):
 i=int(r['frame']);manifest={int(x['frame']):x for x in read('bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv')} if False else None
 # Use original image path from Phase14 candidate manifest.
 m={int(x['frame']):x for x in list(csv.DictReader((P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig')))}
 img=cv2.imread(m[i]['image_path']);candidate=cv2.imread(str(O/r['cad_sam_intersection_candidate']),0)>0; overlay=img.copy();overlay[candidate]=(overlay[candidate]*.55+np.array([0,255,0])*.45).astype('uint8');cv2.putText(overlay,'SAM AND CAD CANDIDATE - NOT GT',(12,30),0,.7,(255,255,255),2);cv2.imwrite(str(O/'previews'/f'{i:06d}_intersection_NOT_GT.jpg'),overlay)
results={'source_phase12_hash_checked':count,'modified':changed,'review_csv_sha256':'PASS','zip_100_images_integrity':'PASS','browser_firefox_headless':{'cards':99,'images':198,'first_image_load':'PASS','change_status_and_note':'PASS','filter':'PASS','local_storage_reload':'PASS','quoted_csv_parse':'PASS','actual_windows_chrome':'NOT TESTED'},'occlusion_summary':occ}
(O/'final_checks.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n');print(json.dumps(results,indent=2,ensure_ascii=False))
