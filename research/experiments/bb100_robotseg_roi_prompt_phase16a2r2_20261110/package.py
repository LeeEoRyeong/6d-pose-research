from pathlib import Path
import csv,json,cv2,numpy as np,hashlib,shutil,zipfile
O=Path(__file__).resolve().parent;P=Path('/home/oylee/projects/HCCEPose_experiments');PRE=P/'bb100_robotseg_phase16a2r_resume_20261110';VIS=P/'bb100_visible_mask_phase16a1_20261110';pkg=O/'package';data=[]
for d in ('panels','full','masks'):(pkg/d).mkdir(parents=True)
with (VIS/'selected_frames.csv').open(encoding='utf-8-sig') as f:base={int(x['frame']):x for x in csv.DictReader(f)}
res=json.loads((O/'results.json').read_text())['frames']; assert len(res)==8
settings={75:((320,140,620,410),(465,295),(505,330)),235:((480,235,820,520),(640,385),(690,425))}
for i,(box,positive,negative) in settings.items():
 img=cv2.imread(base[i]['rgb_path']);sam=cv2.imread(str(VIS/'sam'/f'{i:06d}_selected_NOT_GT.png'),0)>0;cad=cv2.imread(str(VIS/'cad'/f'{i:06d}_AMODAL_NOT_VISIBLE_GT.png'),0)>0;conds=['A_full_auto','B_crop_auto','C_full_points','D_crop_points'];masks=[];links=[]
 for cond in conds:
  p=O/'masks'/f'{i:06d}_{cond}_gripper_NOT_GT.png';m=cv2.imread(str(p),0);assert m.shape==(720,1280) and set(np.unique(m)).issubset({0,255});m=m>0;masks.append(m)
  for name in ('gripper','SAM_minus','CADSAM_minus','removed_SAM'):
   path=O/'masks'/f'{i:06d}_{cond}_{name}_NOT_GT.png';shutil.copy2(path,pkg/'masks'/path.name)
  v=next(r for r in res if r['frame']==i and r['condition']==cond);assert int((m&sam).sum())==v['SAM_overlap_removed_pixels'];assert int((m&sam&cad).sum())==v['CADSAM_overlap_removed_pixels']
  if 'crop' in cond:
   x0,y0,x1,y1=v['roi_xyxy'];outside=m.copy();outside[y0:y1,x0:x1]=False;assert not outside.any()
  v['positive_point_in_mask']=bool(m[positive[1],positive[0]]) if 'points' in cond else None;v['negative_point_in_mask']=bool(m[negative[1],negative[0]]) if 'points' in cond else None
  if cond=='A_full_auto':assert v['baseline_matches_previous'] and v['mask_diff_vs_prior_full_pixels']==0
 # close-up is visual-only, different from inference ROI
 x0,y0,x1,y1=box
 def over(m,col):
  z=img.copy();z[m]=(z[m]*.45+np.array(col)*.55).astype('uint8');return z
 def tile(z,title):
  z=z[y0:y1,x0:x1];h,w=z.shape[:2];k=min(500/w,420/h);nh,nw=int(h*k),int(w*k);out=np.zeros((450,500,3),np.uint8);out[27:27+nh,(500-nw)//2:(500-nw)//2+nw]=cv2.resize(z,(nw,nh));cv2.putText(out,title,(5,18),0,.55,(255,255,255),1,cv2.LINE_AA);return out
 marked=img.copy();cv2.drawMarker(marked,positive,(0,255,0),cv2.MARKER_CROSS,12,2);cv2.drawMarker(marked,negative,(0,0,255),cv2.MARKER_CROSS,12,2)
 views=[(marked,'RGB: GREEN POS / RED NEG'),(over(cad,[0,220,220]),'CAD AMODAL'),(over(sam,[255,0,200]),'SAM BB'),(over(masks[0],[0,0,255]),'A FULL AUTO'),(over(masks[1],[0,0,255]),'B CROP AUTO'),(over(masks[2],[0,0,255]),'C FULL GUIDED'),(over(masks[3],[0,0,255]),'D CROP GUIDED'),(over(sam&~masks[3],[0,210,0]),'D SAM MINUS GRIPPER'),(over(sam&masks[3],[0,0,255]),'D REMOVED FROM SAM')];a=[tile(*v) for v in views];full=np.vstack([np.hstack(a[:3]),np.hstack(a[3:6]),np.hstack(a[6:9])]);cv2.imwrite(str(pkg/'panels'/f'{i:06d}_closeup.jpg'),full,[cv2.IMWRITE_JPEG_QUALITY,92]);shutil.copy2(O/'panels'/f'{i:06d}_comparison.jpg',pkg/'panels'/f'{i:06d}_comparison.jpg');shutil.copy2(O/'full'/f'{i:06d}_ROI_PROMPTS.jpg',pkg/'full'/f'{i:06d}_ROI_PROMPTS.jpg');data.append({'frame':i,'full':f'full/{i:06d}_ROI_PROMPTS.jpg','overview':f'panels/{i:06d}_comparison.jpg','closeup':f'panels/{i:06d}_closeup.jpg','results':[r for r in res if r['frame']==i]})
with (O/'results_with_prompt_checks.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=res[0].keys());w.writeheader();w.writerows(res)
page=(O/'template.html').read_text().replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False));(pkg/'index.html').write_text(page,encoding='utf-8');shutil.copy2(O/'results_with_prompt_checks.csv',pkg/'results.csv')
(pkg/'사용법.txt').write_text('전체 ZIP을 풀고 index.html을 엽니다. 초록 십자=수동 양성점, 빨강 십자=수동 음성점, 빨강 overlay=gripper 후보입니다. 전체/접사 영상과 원본 PNG를 검토하세요. 모든 출력은 GT가 아닙니다.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_RobotSeg_ROI_guided_review.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(pkg.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_RobotSeg_ROI_guided_review.zip') as z:
 assert not z.testzip(); names=set(z.namelist());assert all(all(row[k] in names for k in ('full','overview','closeup')) for row in data)
summary={'conditions':8,'baseline_pixel_reproduction':'PASS both','candidate_formula':'PASS','crop_output_outside_roi_zero':'PASS','manual_points_checked':'PASS','full_235_positive_point_in_mask':next(r['positive_point_in_mask'] for r in res if r['frame']==235 and r['condition']=='C_full_points'),'zip_links_crc':'PASS','windows_chrome':'NOT_TESTED'};(O/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
