from pathlib import Path
import csv,json,hashlib,zipfile,shutil,collections
import numpy as np,cv2
O=Path(__file__).resolve().parent;pkg=O/'package'
with (O/'selected_frames.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==10 and collections.Counter(x['occlusion'] for x in rows)=={'NONE':4,'PARTIAL':6} and len({x['pose_group'] for x in rows})==9
for d in ('sam','cad','intersection','uncertainty'):(pkg/d).mkdir()
data=[]
for r in rows:
 frame=int(r['frame']); prefix=f'{frame:06d}';paths={'sam':f'sam/{prefix}_selected_NOT_GT.png','cad':f'cad/{prefix}_AMODAL_NOT_VISIBLE_GT.png','intersection':f'intersection/{prefix}_CAD_SAM_NOT_GT.png','uncertainty':f'uncertainty/{prefix}_unknown_128_no_supervision.png'}
 for path in paths.values():shutil.copy2(O/path,pkg/path)
 a=cv2.imread(str(O/paths['cad']),0);b=cv2.imread(str(O/paths['sam']),0);c=cv2.imread(str(O/paths['intersection']),0);u=cv2.imread(str(O/paths['uncertainty']),0)
 assert all(x.shape==(720,1280) for x in (a,b,c,u)) and all(set(np.unique(x)).issubset({0,255}) for x in (a,b,c));assert set(np.unique(u)).issubset({0,128});assert np.array_equal(c>0,(a>0)&(b>0));assert np.array_equal(u==128,(a>0)|(b>0));assert (pkg/f'frames/{prefix}.jpg').exists() and (pkg/f'frames/{prefix}_RGB.jpg').exists()
 data.append({**r,**paths,'panel':f'frames/{prefix}.jpg','rgb':f'frames/{prefix}_RGB.jpg'})
template=(O/'template.html').read_text();(pkg/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
shutil.copy2(O/'selected_frames.csv',pkg/'selected_frames.csv')
with (pkg/'review_template.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['frame','occlusion','pose_group','method','status','note'])
 for r in rows:w.writerow([r['frame'],r['occlusion'],r['pose_group'],'C_CAD_SAM','UNREVIEWED',''])
(pkg/'사용법.txt').write_text('전체 ZIP을 압축 해제하고 index.html을 Chrome에서 오프라인으로 여세요. 각 프레임은 SAM과 CAD 교집합 등 비교 후보를 제공합니다. 모두 GT가 아닙니다. 검수할 방법을 선택하고 ACCEPT/CORRECT/UNCERTAIN과 메모를 입력한 다음 CSV를 내보내 보관하세요. ACCEPT는 선택된 마스크의 시각 검수이며 파일을 픽셀 단위로 수정하지 않습니다. 직접 수정이 필요하면 별도 라벨링 도구를 사용하세요. D의 128은 불확실, 0은 미지정이며 모두 학습에 사용하지 않습니다.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_visible_mask_review_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(pkg.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_visible_mask_review_windows.zip') as z:
 assert z.testzip() is None;names=set(z.namelist());assert all(all(r[x] in names for x in ['panel','rgb','cad','sam','intersection','uncertainty']) for r in data)
result={'selected':10,'none':4,'partial':6,'pose_groups':9,'mask_shape_binary_and_uncertainty':'PASS','intersection_and_uncertainty_semantics':'PASS','zip_all_links_and_crc':'PASS','supervision_pixels':0,'all_review_status':'UNREVIEWED','browser_windows_chrome':'NOT_TESTED'}
(O/'package_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
