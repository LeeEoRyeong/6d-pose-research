from pathlib import Path
import cv2,numpy as np,csv,json,hashlib,collections
O=Path(__file__).resolve().parent;P=Path('/home/oylee/projects/HCCEPose_experiments')
with (O/'selected_frames.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
results=[]
for x in rows:
 frame=int(x['frame']);cad=cv2.imread(str(O/f'cad/{frame:06d}_AMODAL_NOT_VISIBLE_GT.png'),0);sam=cv2.imread(str(O/f'sam/{frame:06d}_selected_NOT_GT.png'),0);inter=cv2.imread(str(O/f'intersection/{frame:06d}_CAD_SAM_NOT_GT.png'),0);cont,h=cv2.findContours(cad,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);n=filled=0
 for i,c in enumerate(cont):
  if h[0,i,3]<0 or cv2.contourArea(c)<8:continue
  mask=np.zeros(cad.shape,np.uint8);cv2.drawContours(mask,cont,i,255,-1);mask=(mask>0)&(cad==0)
  if mask.sum()<8:continue
  n+=1;filled+=int((sam[mask]>0).mean()>.5);assert not (inter[mask]>0).any()
 results.append({'frame':frame,'occlusion':x['occlusion'],'cad_enclosed_holes_ge_8px':n,'cad_holes_over_half_sam_foreground':filled,'note':'CAD geometry comparison only; holes may be occluded or misprojected; no ground-truth accuracy'})
with (O/'cad_hole_diagnostic.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
old=json.loads((P/'bb100_visual_review_20261109/manifest.json').read_text())['sources_sha256'];changed=[];count=0
for name,h in old.items():
 p=Path(name)
 if p.is_file():
  count+=1
  if hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(name)
assert not changed
info={'source_hashes_307_phase12':count,'source_changed':changed,'original_checkpoint_sha256':hashlib.sha256((P/'itodd_official_audit_20261108/itodd_obj04_1_0000step50000.pt').read_bytes()).hexdigest(),'sam_holes_check':'PASS structural measurement only','headless_firefox_cards_import_export_reload':'PASS','actual_windows_chrome':'NOT TESTED'}
assert info['original_checkpoint_sha256']=='224c0869e20be8c9825b77925837e98f0e4c31ae1006361cddd1b46e0cb48ffc'
(O/'final_validation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,indent=2))
