from pathlib import Path
import csv,json,collections,cv2,numpy as np
O=Path(__file__).resolve().parent;sam=json.loads((O/'sam_generation.json').read_text())['frames'];robot={x['frame']:x for x in json.loads((O/'robot_generation.json').read_text())['frames']};rows=[]
for x in sam:
 i=x['frame'];fr=f'{i:06d}';A=cv2.imread(str(O/'candidates/A'/f'{fr}.png'),0)>0;B=cv2.imread(str(O/'candidates/B'/f'{fr}.png'),0)>0;C=cv2.imread(str(O/'candidates/C'/f'{fr}.png'),0)>0;D=cv2.imread(str(O/'candidates/D'/f'{fr}.png'),0)>0;E=cv2.imread(str(O/'candidates/E'/f'{fr}.png'),0)>0;R=cv2.imread(str(O/'candidates/R'/f'{fr}.png'),0)>0
 assert np.array_equal(C,A&B) and np.array_equal(D,B&~R) and np.array_equal(E,C&~R)
 contours,h=cv2.findContours(A.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);holes=bc=dc=0
 if h is not None:
  for j,c in enumerate(contours):
   if h[0,j,3]<0 or cv2.contourArea(c)<8:continue
   tmp=np.zeros(A.shape,np.uint8);cv2.drawContours(tmp,contours,j,255,-1);ix=(tmp>0)&~A
   if ix.sum()<8:continue
   holes+=1;bc+=int(B[ix].mean()>.5);dc+=int(D[ix].mean()>.5)
 z=robot[i];assert z['B_D_difference']==int(np.count_nonzero(B^D))
 rows.append({'frame':i,'occlusion':x['occlusion'],'pose_group':x['pose_group'],'B_D_diff_pixels':int(np.count_nonzero(B^D)),'B_C_diff_pixels':int(np.count_nonzero(B^C)),'robot_removed_pixels':int((B&R).sum()),'cad_removed_pixels':int((B&~A).sum()),'B_pixels':int(B.sum()),'C_pixels':int(C.sum()),'D_pixels':int(D.sum()),'cad_projected_enclosed_holes_ge8px':holes,'cad_holes_half_filled_B':bc,'cad_holes_half_filled_D':dc,'actual_visible_surface_missing':'UNKNOWN_WITHOUT_ACCEPTED_GT','actual_hole_correctness':'UNKNOWN','note':'CAD projection structure is not visible ground truth'})
with (O/'candidate_analysis.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={'frames':len(rows),'D_identical_to_B_frames':sum(x['B_D_diff_pixels']==0 for x in rows),'D_diff_nonzero_frames':sum(x['B_D_diff_pixels']>0 for x in rows),'B_C_diff_nonzero_frames':sum(x['B_C_diff_pixels']>0 for x in rows),'B_D_diff_median':float(np.median([x['B_D_diff_pixels'] for x in rows])),'B_D_diff_max':max(x['B_D_diff_pixels'] for x in rows),'B_C_diff_median':float(np.median([x['B_C_diff_pixels'] for x in rows])),'note':'hole counts and differences relative to CAD are NOT ground truth; per-frame accepted reference evaluation separate'};(O/'candidate_analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
