import os,csv,json,hashlib,glob,html,traceback
from pathlib import Path
from collections import Counter
import numpy as np,cv2
B=Path('/home/oylee/projects/HCCEPose_experiments'); O=B/'bb100_gt_integrity_phase16a5_20261111'; A=B/'bb100_visible_gt_phase16a4_20261110'; R=B/'real_bb100_20261108/BB_obj04'; M=B/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv'; CAD=B/'itodd_official_audit_20261108/official_obj_000004.ply'
for d in ['visualizations','contact_sheets']: (O/d).mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 if rows:
  keys=list(dict.fromkeys(k for row in rows for k in row))
  with open(p,'w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def savejpg(p,im):
 assert cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,90])
def holes(binary):
 contours,h=cv2.findContours(binary.astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); out=np.zeros(binary.shape,np.uint8); n=0
 if h is not None:
  for i,c in enumerate(contours):
   if h[0,i,3]>=0:
    t=np.zeros_like(out);cv2.drawContours(t,[c],-1,255,-1);t[binary>0]=0
    if np.any(t): n+=1;out=cv2.bitwise_or(out,t)
 return out,n
manifest=readcsv(M); status=readcsv(A/'annotations/frame_status.csv'); sm={r['frame']:r for r in status}; gt=json.load(open(R/'scene_gt_obj04.json')); cams=json.load(open(R/'scene_camera_obj04.json'))
source=set([M,A/'annotations/frame_status.csv',R/'scene_gt_obj04.json',R/'scene_camera_obj04.json',CAD]);
for m in manifest:
 fr=f"{int(m['frame']):06d}";source.update([Path(m['image_path']),Path(m['pose_path']),A/'candidates/A'/f'{fr}.png'])
 source.update(p for p in (A/'annotations'/fr/'versions').glob('**/*') if p.is_file())
snapshot={str(p):sha(p) if p.is_file() else None for p in sorted(source)}
(O/'source_hashes_before.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False),encoding='utf-8')
rows=[];history=[];bop=[];errors=[]; thumbnails={}
if len(manifest)!=100 or len(status)!=100 or len(sm)!=100 or len({x['frame'] for x in manifest})!=100:errors.append('프레임 manifest/status 개수 또는 중복 오류')
K0=np.array([[984.70009004,0,635.41416344],[0,983.75715611,362.61850196],[0,0,1.]])
for m in manifest:
 fr=f"{int(m['frame']):06d}"; row={'frame':fr,'occlusion':m['occlusion'],'pose_group':m['pose_group'],'image_path':m['image_path'],'pose_path':m['pose_path'],'cad_path':str(CAD)}
 try:
  s=sm[fr];versions=sorted((A/'annotations'/fr/'versions').glob('*')); dirs=[p for p in versions if p.is_dir()]; revs={p.name for p in dirs}; metas=[]
  for v in dirs:
   hm={'frame':fr,'revision':v.name,'metadata_exists':(v/'metadata.json').exists(),'mask_exists':(v/'working_mask.png').exists(),'parent_exists':'','hash_matches':'','status':''}
   if (v/'metadata.json').is_file():
    vmeta=json.loads((v/'metadata.json').read_text());parent=vmeta.get('parent_revision');hm.update(status=vmeta.get('status'),parent_exists=(not parent or parent in revs),hash_matches=((v/'working_mask.png').is_file() and sha(v/'working_mask.png')==vmeta.get('mask_sha256')));metas.append((v,vmeta))
   history.append(hm)
  assert metas,'저장 버전 없음';latest,meta=metas[-1];p=latest/'working_mask.png'; img=cv2.imread(m['image_path'],cv2.IMREAD_COLOR);mask=cv2.imread(str(p),cv2.IMREAD_UNCHANGED);cad=cv2.imread(str(A/'candidates/A'/f'{fr}.png'),cv2.IMREAD_UNCHANGED);pose=np.load(m['pose_path'],allow_pickle=False);k=np.array(cams[str(int(fr))]['cam_K']).reshape(3,3);gj=gt[str(int(fr))][0]
  checks={}
  checks['status_accepted']=s['status']=='ACCEPTED' and s['saved']=='True' and meta['status']=='ACCEPTED' and bool(meta.get('accepted_explicitly')) and bool(meta.get('approved_utc'))
  checks['latest_revision']=latest.name==s['revision']==meta['revision'] and str(p)==s['save_path']==meta['save_path'] and len(dirs)==len(metas)
  checks['frame_match']=str(meta['frame']).zfill(6)==fr and Path(m['image_path']).stem==fr and Path(m['pose_path']).stem.endswith(fr) and gj['obj_id']==4
  checks['image_hash']=sha(m['image_path'])==m['image_sha256']==meta['rgb_sha256'];checks['pose_hash']=sha(m['pose_path'])==m['pose_sha256'];checks['mask_hash']=sha(p)==meta['mask_sha256']
  checks['image_format']=img is not None and img.shape==(720,1280,3)
  checks['mask_format']=mask is not None and mask.shape==(720,1280) and mask.dtype==np.uint8 and set(np.unique(mask)).issubset({0,128,255})
  checks['cad_format']=cad is not None and cad.shape==(720,1280) and set(np.unique(cad)).issubset({0,255}) and CAD.is_file()
  assert checks['image_format'] and checks['mask_format'] and checks['cad_format'],'이미지/마스크 형식 오류'
  fg=mask==255;ignore=mask==128;bg=mask==0;am=cad>0;inter=fg&am;out=fg&~am;missing=am&bg
  rot=pose[:3,:3];orth=float(np.max(abs(rot.T@rot-np.eye(3))));det=float(np.linalg.det(rot));gpose=np.eye(4);gpose[:3,:3]=np.array(gj['cam_R_m2c']).reshape(3,3);gpose[:3,3]=gj['cam_t_m2c']; pdiff=float(np.max(abs(pose-gpose)))
  checks['pose_valid']=pose.shape==(4,4) and bool(np.isfinite(pose).all()) and np.allclose(pose[3],[0,0,0,1],atol=1e-8) and orth<1e-5 and abs(det-1)<1e-5 and pose[2,3]>0
  checks['pose_json_match']=pdiff<1e-8
  checks['camera_valid']=bool(np.isfinite(k).all()) and k[0,0]>0 and k[1,1]>0 and np.allclose(k,K0,atol=1e-8) and np.allclose(k,np.array(json.loads(m['K'])),atol=1e-8)
  fgmask=np.where(fg,255,0).astype(np.uint8);valid=np.where(~ignore,255,0).astype(np.uint8)
  checks['foreground_valid_pair']=np.array_equal(cv2.imread(str(latest/'evaluation_foreground_255.png'),-1),fgmask) and np.array_equal(cv2.imread(str(latest/'evaluation_valid_255.png'),-1),valid)
  checks['metadata_counts']=int(meta['foreground_pixels'])==int(fg.sum()) and int(meta['ignore_pixels'])==int(ignore.sum()) and int(s['foreground_pixels'])==int(fg.sum()) and int(s['ignore_pixels'])==int(ignore.sum())
  checks['nonempty']=bool(fg.any());checks['history_integrity']=all(x['metadata_exists'] and x['mask_exists'] and x['parent_exists'] and x['hash_matches'] for x in history if x['frame']==fr)
  fh,hn=holes(fgmask);ch,chn=holes(cad); cc,lab,stats,_=cv2.connectedComponentsWithStats(fgmask,8); sizes=stats[1:,cv2.CC_STAT_AREA];bbox=cv2.boundingRect(fgmask)
  row.update({**checks,'latest_revision_id':latest.name,'working_mask':str(p),'history_versions':len(dirs),'foreground_px':int(fg.sum()),'foreground_fraction_image':float(fg.mean()),'ignore_px':int(ignore.sum()),'bbox_xywh':json.dumps(bbox),'components_all':int(cc-1),'components_ge10px':int((sizes>=10).sum()),'holes_count':hn,'holes_area_px':int((fh>0).sum()),'holes_ignore_px':int(((fh>0)&ignore).sum()),'cad_area_px':int(am.sum()),'intersection_px':int(inter.sum()),'outside_cad_fg_px':int(out.sum()),'outside_cad_fg_ratio':float(out.sum()/max(fg.sum(),1)),'cad_inside_background_px':int(missing.sum()),'cad_inside_ignore_px':int((am&ignore).sum()),'cad_hole_count':chn,'cad_hole_area_px':int((ch>0).sum()),'visible_fg_in_cad_holes_px':int(((ch>0)&fg).sum()),'cad_fg_in_visible_holes_px':int(((fh>0)&am).sum()),'rotation_orthogonality_error':orth,'rotation_det':det,'pose_json_maxdiff':pdiff,'camera_maxdiff':float(abs(k-K0).max())})
  bop.append({'frame':fr,'rgb':m['image_path'],'pose_npy':m['pose_path'],'scene_gt':str(R/'scene_gt_obj04.json'),'scene_camera':str(R/'scene_camera_obj04.json'),'cad':str(CAD),'working_0_128_255':str(p),'visible_fg_255':str(latest/'evaluation_foreground_255.png'),'valid_255':str(latest/'evaluation_valid_255.png'),'ignore_pixels':int(ignore.sum()),'status':meta['status']})
  panel=img.copy();panel[fg]=(panel[fg]*.55+np.array([25,35,240])*.45).astype(np.uint8);panel[ignore]=(panel[ignore]*.45+np.array([0,240,255])*.55).astype(np.uint8)
  contours,_=cv2.findContours(cad,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(255,255,0),1)
  ys,xs=np.where(fg|am);x1=max(0,int(xs.min())-35);x2=min(1280,int(xs.max())+36);y1=max(0,int(ys.min())-35);y2=min(720,int(ys.max())+36)
  def resize(im,w=480,h=270):return cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA if im.shape[1]>w else cv2.INTER_LINEAR)
  def fit(im,w=480,h=270):
   scale=min(w/im.shape[1],h/im.shape[0]);v=cv2.resize(im,None,fx=scale,fy=scale);canvas=np.full((h,w,3),24,np.uint8);yy=(h-v.shape[0])//2;xx=(w-v.shape[1])//2;canvas[yy:yy+v.shape[0],xx:xx+v.shape[1]]=v;return canvas
  panels=[resize(img),resize(panel),fit(panel[y1:y2,x1:x2])];board=np.full((300,1440,3),28,np.uint8)
  for i,v in enumerate(panels):board[30:,i*480:(i+1)*480]=v
  for i,txt in enumerate([f'{fr} RGB',f'VISIBLE red / IGNORE yellow / CAD cyan',f'ZOOM object + CAD holes']):cv2.putText(board,txt,(i*480+8,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
  savejpg(O/'visualizations'/f'{fr}.jpg',board); thumbnails[fr]=resize(panel,320,180)
 except Exception as e:
  row['error']=f'{type(e).__name__}: {e}';errors.append(f'{fr}: {row["error"]}');traceback.print_exc()
 rows.append(row)
 print(fr,row.get('error','OK'),flush=True)
validrows=[r for r in rows if 'foreground_px' in r]; areas=np.array([r['foreground_px'] for r in validrows]);q1,q3=np.percentile(areas,[25,75]);lo=max(0,q1-1.5*(q3-q1));hi=q3+1.5*(q3-q1)
for r in validrows:
 flags=[]
 if r['foreground_px']<lo or r['foreground_px']>hi:flags.append('면적_IQR_이상')
 if r['outside_cad_fg_ratio']>.05:flags.append('CAD밖_전경_5%초과')
 if r['components_ge10px']>1:flags.append('연결성분_복수')
 if r['ignore_px']>0:flags.append('ignore_존재')
 r['qa_flags']=';'.join(flags)
for i in range(0,len(validrows),25):
 batch=validrows[i:i+25];sheet=np.full((5*210,5*320,3),30,np.uint8)
 for j,r in enumerate(batch):
  y=(j//5)*210;x=(j%5)*320;sheet[y+25:y+205,x:x+320]=thumbnails[r['frame']];cv2.putText(sheet,r['frame']+'  '+str(r['ignore_px'])+' ignore',(x+5,y+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
 savejpg(O/'contact_sheets'/f'sheet_{i//25+1}.jpg',sheet)
writecsv(O/'frame_integrity.csv',rows);writecsv(O/'version_history_audit.csv',history);writecsv(O/'bop_preparation_manifest.csv',bop)
source_after={str(p):sha(p) if p.is_file() else None for p in sorted(source)};changed=[p for p in snapshot if snapshot[p]!=source_after[p]]
# Detect new version files after the initial snapshot as well.
newfiles=[]
for m in manifest:
 fr=f"{int(m['frame']):06d}";newfiles.extend(str(p) for p in (A/'annotations'/fr/'versions').glob('**/*') if p.is_file() and str(p) not in snapshot)
(O/'source_hashes_after.json').write_text(json.dumps(source_after,indent=2,ensure_ascii=False),encoding='utf-8')
checks=[k for k in validrows[0] if k in ('status_accepted','latest_revision','frame_match','image_hash','pose_hash','mask_hash','image_format','mask_format','cad_format','pose_valid','pose_json_match','camera_valid','foreground_valid_pair','metadata_counts','nonempty','history_integrity')] if validrows else []
summary={'manifest_count':len(manifest),'status_count':len(status),'valid_rows':len(validrows),'checks_failed':{k:[r['frame'] for r in validrows if not r[k]] for k in checks},'errors':errors,'changed_sources':changed,'new_version_files':newfiles,'history_count':len(history),'area_min_median_max':[int(areas.min()),float(np.median(areas)),int(areas.max())] if len(areas) else [],'area_iqr_fences':[lo,hi],'frames_with_ignore':sum(r['ignore_px']>0 for r in validrows),'total_ignore_px':sum(r['ignore_px'] for r in validrows),'outside_cad_total':sum(r['outside_cad_fg_px'] for r in validrows),'flags':dict(Counter(f for r in validrows for f in r['qa_flags'].split(';') if f)),'flagged_frames':{f:[r['frame'] for r in validrows if f in r['qa_flags'].split(';')] for f in ['면적_IQR_이상','CAD밖_전경_5%초과','연결성분_복수','ignore_존재']}}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
header='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>BB100 승인 마스크 QA</title><style>body{font:16px sans-serif;background:#20242a;color:white;margin:24px}a{color:#a9e3ff}img{max-width:100%;border:1px solid #777}article{margin:25px 0;border-top:1px solid #666}input{padding:8px}small{color:#ddd}</style><h1>BB100 승인 마스크 QA</h1><p>빨강: visible 전경, 노랑: ignore, 청록: CAD amodal 윤곽. 두 마스크의 차이는 가림·보정·투영 오차 등을 포함하며 라벨 오류를 뜻하지 않습니다.</p><input id="q" placeholder="프레임 또는 표시 검색" oninput="filter()"><p>검사 이미지는 읽기 전용이며 저장되지 않은 브라우저 편집 내용은 포함하지 않습니다.</p>'''
content=[]
for r in rows:
 if 'foreground_px' not in r:continue
 fr=r['frame'];info=f"{fr} / {r['occlusion']} / 전경 {r['foreground_px']} / ignore {r['ignore_px']} / CAD 밖 {r['outside_cad_fg_px']} / {r['qa_flags']}";content.append(f'<article data-text="{html.escape(info)}"><h3>{html.escape(info)}</h3><a href="visualizations/{fr}.jpg"><img loading="lazy" src="visualizations/{fr}.jpg" alt="{fr}"></a></article>')
(O/'index.html').write_text(header+''.join(content)+'''<script>function filter(){let q=document.getElementById('q').value.toLowerCase();document.querySelectorAll('article').forEach(x=>x.style.display=x.dataset.text.toLowerCase().includes(q)?'':'none')}</script></html>''',encoding='utf-8')
print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
