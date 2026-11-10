from pathlib import Path
import shutil,json,zipfile
from app import Store,ROOT,FRAMES
O=ROOT;pkg=O/'review_package';(pkg/'frames').mkdir(parents=True);(pkg/'masks').mkdir();store=Store();items=[]
for fr in FRAMES:
 src=store.entries[fr];panel=O.parent/'bb100_visible_mask_phase16a1_20261110/package/frames'/f'{fr}.jpg';shutil.copy2(panel,pkg/'frames'/panel.name)
 c=[]
 for key,path in src['candidates'].items():
  dst=f'masks/{fr}_{key}.png';shutil.copy2(path,pkg/dst);c.append({'name':key,'path':dst})
 items.append({'frame':fr,'occlusion':src['occlusion'],'panel':f'frames/{fr}.jpg','candidates':c})
data=json.dumps(items,ensure_ascii=False);html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>BB100 후보 오프라인 검수</title><style>body{font:16px sans-serif;max-width:1700px;margin:auto;background:#eef;padding:1rem}article{background:white;padding:1rem;margin:1rem 0}img{max-width:100%}a{margin-right:1rem}.warning{color:#a21}</style><h1>BB100 후보 비교 — 정적 스냅샷</h1><p class="warning">모든 마스크는 미검수 후보이며 GT가 아닙니다. 이 페이지에서는 편집·승인할 수 없습니다. 편집기는 서버에서 실행하고 SSH 포트 포워딩으로 접속하세요. 상태는 실시간 동기화되지 않습니다.</p><main id="list"></main><script>const data=DATA;for(const b of data){let e=document.createElement('article'),h=document.createElement('h2');h.textContent=`${b.frame} ${b.occlusion}`;e.append(h);let im=document.createElement('img');im.src=b.panel;im.loading='lazy';e.append(im);let p=document.createElement('p');for(const c of b.candidates){let a=document.createElement('a');a.href=c.path;a.textContent=c.name;a.target='_blank';p.append(a)}e.append(p);document.querySelector('#list').append(e)}</script></html>'''.replace('DATA',data)
(pkg/'index.html').write_text(html,encoding='utf-8');(pkg/'사용법.txt').write_text('이 ZIP은 후보 비교용 정적 스냅샷입니다. 편집이나 승인 기능은 서버 편집기를 SSH 포트 포워딩하여 사용하세요. 모든 후보는 GT가 아닙니다.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_visible_GT_candidate_review_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(pkg.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_visible_GT_candidate_review_windows.zip') as z:
 assert not z.testzip();names=set(z.namelist());assert all(b['panel'] in names and all(c['path'] in names for c in b['candidates']) for b in items)
(O/'review_package_validation.json').write_text(json.dumps({'frames':10,'references':'PASS','crc':'PASS','snapshot_not_live':True,'editor_in_zip':False},indent=2)+'\n');print('10 frames, ZIP PASS')
