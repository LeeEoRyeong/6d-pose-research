from pathlib import Path
import csv,json,cv2,numpy as np,hashlib,html
B=Path('/home/oylee/projects/HCCEPose_experiments');O=B/'bb100_gt_integrity_phase16a5_20261111';A=B/'bb100_visible_gt_phase16a4_20261110';P=B/'bb100_placement_targets_phase15_20261109';(O/'hole_closeups').mkdir(exist_ok=True)
with open(O/'frame_integrity.csv',encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
changed=[]
for a in r:
 fr=a['frame'];rgb=cv2.imread(a['image_path']);m=cv2.imread(a['working_mask'],-1);cad=cv2.imread(str(A/'candidates/A'/f'{fr}.png'),0);prev=cv2.imread(str(P/'amodal_masks'/f"pose_{a['pose_group']}.png"),0)
 if prev is None or not np.array_equal(cad,prev):changed.append(fr)
 contours,h=cv2.findContours(cad,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);inside=[]
 if h is not None:inside=[c for i,c in enumerate(contours) if h[0,i,3]>=0]
 if inside:
  c=max(inside,key=cv2.contourArea);x,y,w,hh=cv2.boundingRect(c);cx=x+w//2;cy=y+hh//2
 else: x,y,w,hh=cv2.boundingRect(cad);cx=x+w//2;cy=y+hh//2
 radius=max(42,min(120,max(w,hh)*2));x1=max(0,cx-radius);x2=min(1280,cx+radius);y1=max(0,cy-radius);y2=min(720,cy+radius)
 overlay=rgb.copy();fg=m==255;ig=m==128;overlay[fg]=(overlay[fg]*.55+np.array([25,35,240])*.45).astype('uint8');overlay[ig]=(overlay[ig]*.45+np.array([0,240,255])*.55).astype('uint8');cv2.drawContours(overlay,contours,-1,(255,255,0),1)
 def zoom(im):return cv2.resize(im[y1:y2,x1:x2],(400,400),interpolation=cv2.INTER_NEAREST)
 board=np.full((430,800,3),25,np.uint8);board[30:,:400]=zoom(rgb);board[30:,400:]=zoom(overlay)
 cv2.putText(board,fr+' RGB hole vicinity',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);cv2.putText(board,'VISIBLE red / IGNORE yellow / CAD cyan',(408,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
 cv2.imwrite(str(O/'hole_closeups'/f'{fr}.jpg'),board)
idx=(O/'index.html').read_text();idx=idx.replace('</a></article>', '</a><br><a href="hole_closeups/PLACEHOLDER.jpg">구멍 확대</a></article>') if False else idx
for a in r:
 fr=a['frame'];idx=idx.replace(f'<a href="visualizations/{fr}.jpg"><img loading="lazy" src="visualizations/{fr}.jpg" alt="{fr}"></a>',f'<a href="visualizations/{fr}.jpg"><img loading="lazy" src="visualizations/{fr}.jpg" alt="{fr}"></a><br><a href="hole_closeups/{fr}.jpg">CAD 구멍 주변 확대 비교</a>')
(O/'index.html').write_text(idx)
priority={}
for field in ['outside_cad_fg_ratio','visible_fg_in_cad_holes_px','cad_inside_background_px']:
 for a in sorted(r,key=lambda x:float(x[field]),reverse=True)[:10]:priority.setdefault(a['frame'],[]).append(field+' 상위 10')
with open(O/'priority_review.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['frame','reason','occlusion','outside_cad_fg_ratio','visible_fg_in_cad_holes_px','cad_inside_background_px','note']);w.writeheader()
 for a in r:
  if a['frame'] in priority:w.writerow({**{k:a[k] for k in ['frame','occlusion','outside_cad_fg_ratio','visible_fg_in_cad_holes_px','cad_inside_background_px']},'reason':';'.join(priority[a['frame']]),'note':'순위 기반 확인 대상. 가림 및 투영 차이로 발생할 수 있으며 오류 판정 아님'})
res={'cad_phase15_pixel_identical':100-len(changed),'cad_mismatch':changed,'hole_closeups':len(list((O/'hole_closeups').glob('*.jpg'))),'priority_frames':len(priority),'top_outside_ratio':[(x['frame'],round(float(x['outside_cad_fg_ratio'])*100,2)) for x in sorted(r,key=lambda x:float(x['outside_cad_fg_ratio']),reverse=True)[:5]],'top_hole':[(x['frame'],int(x['visible_fg_in_cad_holes_px'])) for x in sorted(r,key=lambda x:int(x['visible_fg_in_cad_holes_px']),reverse=True)[:5]]}
(O/'supplement.json').write_text(json.dumps(res,indent=2,ensure_ascii=False));print(json.dumps(res,ensure_ascii=False))
