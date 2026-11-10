from pathlib import Path
import json,csv,cv2,numpy as np,shutil,zipfile,hashlib
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;pkg=O/'package';data=[]
for p in ('panels','full','masks'):(pkg/p).mkdir(parents=True)
rows=json.loads((O/'smoke.json').read_text())['frames']+json.loads((O/'remaining.json').read_text())['frames'];rows.sort(key=lambda r:r['frame']);assert len(rows)==6
for row in rows:
 i=row['frame'];tag='smoke' if i==235 else 'remaining';base=O/tag;maskdir=base/'masks';g=cv2.imread(str(maskdir/f'{i:06d}_robotseg_gripper_NOT_GT.png'),0);a=cv2.imread(str(maskdir/f'{i:06d}_A_sam_NOT_GT.png'),0);b=cv2.imread(str(maskdir/f'{i:06d}_B_cad_sam_NOT_GT.png'),0);c=cv2.imread(str(maskdir/f'{i:06d}_C_sam_minus_gripper_NOT_GT.png'),0);d=cv2.imread(str(maskdir/f'{i:06d}_D_cad_sam_minus_gripper_NOT_GT.png'),0);e=cv2.imread(str(maskdir/f'{i:06d}_E_cad_minus_gripper_NOT_GT.png'),0)
 assert all(m is not None and m.shape==(720,1280) and set(np.unique(m))<={0,255} for m in (g,a,b,c,d,e));assert np.array_equal(c>0,(a>0)&(g==0)) and np.array_equal(d>0,(b>0)&(g==0));assert int(np.count_nonzero(g))==row['robotseg_gripper_pixels'];row['gripper_bbox_xywh']=list(map(int,cv2.boundingRect(g)));row['A_C_identical']=bool(np.array_equal(a,c));row['B_D_identical']=bool(np.array_equal(b,d));assert row['A_C_identical'] and row['B_D_identical']
 panel=f'panels/{i:06d}.jpg';full=f'full/{i:06d}.jpg';shutil.copy2(base/'panels'/f'{i:06d}_comparison.jpg',pkg/panel);shutil.copy2(O/'full_overlays'/f'{i:06d}.jpg',pkg/full)
 for f in maskdir.glob(f'{i:06d}_*.png'):shutil.copy2(f,pkg/'masks'/f.name)
 data.append({**row,'panel':panel,'full':full})
with (O/'frame_results.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
template=(O/'template.html').read_text();(pkg/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False)),encoding='utf-8');shutil.copy2(O/'frame_results.csv',pkg/'frame_results.csv')
(pkg/'사용법.txt').write_text('ZIP 전체를 압축 해제하고 index.html을 열어 전체 화면의 RobotSeg 후보(빨강)와 CAD/SAM 결합 결과를 확인하세요. 자동 출력은 GT가 아닙니다. 좌표별 참조 mask가 없어 정확도를 산출하지 않습니다.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_RobotSeg_6frame_review.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(pkg.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_RobotSeg_6frame_review.zip') as z:
 assert not z.testzip();names=set(z.namelist());assert all(r['full'] in names and r['panel'] in names for r in data)
pre=json.loads((P/'bb100_visual_review_20261109/manifest.json').read_text())['sources_sha256'];bad=[];n=0
for path,h in pre.items():
 path=Path(path)
 if path.is_file():
  n+=1
  if hashlib.sha256(path.read_bytes()).hexdigest()!=h:bad.append(str(path))
assert not bad
report={'frames':6,'mask_format_and_formula':'PASS','A_equals_C_frames':6,'B_equals_D_frames':6,'archive_crc_and_links':'PASS','original_sources_checked':n,'changed':bad,'windows_chrome':'NOT_TESTED','gpu_training':'NOT_RUN'};(O/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
