from pathlib import Path
import csv,json,zipfile,shutil,numpy as np,cv2
O=Path(__file__).resolve().parent;pkg=O/'package';data=json.loads((O/'data.json').read_text());assert len(data)==6
for r in data:
 id=r['mask_prefix'];p=O/'flow'/f'{id}_flow_NOT_GT.npz';z=np.load(p);f=z['forward'];b=z['backward'];err=z['fb_error'];valid=z['consistent'];assert f.shape==b.shape==(720,1280,2) and np.isfinite(f).all() and np.isfinite(b).all() and np.isfinite(err).all();assert set(np.unique(valid))<={0,1}
 def m(n):return cv2.imread(str(pkg/'masks'/f'{id}_{n}_NOT_GT.png'),0)>0
 sam=m('sam_before');base=m('cad_sam_before');move=m('flow_move_not_gripper');c=m('sam_minus_motion_exploratory');d=m('cad_sam_minus_motion_exploratory');assert np.array_equal(c,sam&~move) and np.array_equal(d,base&~move) and np.all(~base|sam)
 assert int((sam&move).sum())==r['sam_minus_motion_removed_pixels']
 if not r['flow_suppression_enabled']:assert not move.any()
 assert (pkg/r['panel']).is_file()
template=(O/'template.html').read_text();(pkg/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8');shutil.copy2(O/'pairs.csv',pkg/'pairs.csv')
(pkg/'사용법.txt').write_text('전체 ZIP을 압축 해제하고 index.html을 Chrome에서 오프라인으로 여세요. Flow는 그리퍼 분할이 아닙니다. SAM-minus-motion과 CAD∩SAM-minus-motion은 실패 가능성이 있는 탐색 결과이며 GT나 학습용 마스크가 아닙니다. 노란색은 순·역 flow 불일치 영역으로 가림 확정이 아닙니다. CSV 내보내기로 판단과 메모를 저장하세요.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_liteflownet_review_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(pkg.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_liteflownet_review_windows.zip') as z:
 assert z.testzip() is None;names=set(z.namelist());assert all(r['panel'] in names for r in data)
 for r in data:
  assert all(f"masks/{r['mask_prefix']}_{n}_NOT_GT.png" in names for n in ('sam_before','cad_sam_before','flow_move_not_gripper','sam_minus_motion_exploratory','cad_sam_minus_motion_exploratory','fb_inconsistent'))
result={'pairs':6,'flow_shape_and_finite':'PASS','difference_masks_and_background_gate':'PASS','zip_crc_and_image_mask_links':'PASS','actual_windows_chrome':'NOT_TESTED'};(O/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
