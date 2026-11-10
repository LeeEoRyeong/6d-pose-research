from pathlib import Path
import csv,json,hashlib,collections,math,sys,shutil,zipfile,time,html
import numpy as np,cv2
P=Path('/home/oylee/projects/HCCEPose_experiments'); OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'bb_weak_supervision_20261108'));import pipeline
cv2.setNumThreads(1)
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wc(name,rows):
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def wj(name,o):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False);f.write('\n')
def hashfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def png(p,a):
 assert not p.exists() and cv2.imwrite(str(p),a)
def roi_features(im1,im2,mask):
 gray1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY);gray2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY);select=mask>0
 mae=float(np.mean(np.abs(gray1[select].astype(float)-gray2[select].astype(float))))
 e1=cv2.Canny(gray1,70,150)>0;e2=cv2.Canny(gray2,70,150)>0
 # Feature tracks use amodal support of previous pose and can track occluders.
 pts=cv2.goodFeaturesToTrack(gray1,maxCorners=150,qualityLevel=.01,minDistance=5,mask=mask.astype('uint8'))
 n=0;dx=dy=disp=None
 if pts is not None:
  q,st,_=cv2.calcOpticalFlowPyrLK(gray1,gray2,pts,None,winSize=(21,21),maxLevel=3)
  if q is not None:
   back,st2,_=cv2.calcOpticalFlowPyrLK(gray2,gray1,q,None,winSize=(21,21),maxLevel=3)
   if back is not None:
    ok=(st.ravel()>0)&(st2.ravel()>0)&(np.linalg.norm(back[:,0,:]-pts[:,0,:],axis=1)<1.5)
    diff=(q-pts)[:,0,:][ok];n=len(diff)
    if n:dx,dy=np.median(diff,axis=0).tolist();disp=float(np.median(np.linalg.norm(diff,axis=1)))
 edge_distance=None
 if np.any(e1&select) and np.any(e2&select):
  dist=cv2.distanceTransform((~e2).astype('uint8'),cv2.DIST_L2,3);edge_distance=float(np.median(dist[e1&select]))
 return mae,n,dx,dy,disp,edge_distance

def main():
 start=time.monotonic();source=P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv';rows=readcsv(source);review=readcsv(P/'bb100_review_export (2).csv');audit=readcsv(P/'bb100_gt_camera_verification_20261109/bb100_frame_audit.csv');am={int(x['frame']):x for x in audit}
 assert len(rows)==len(review)==100 and all(r['review_status']=='ACCEPT' for r in rows);assert collections.Counter(x['occlusion'] for x in rows)=={'NONE':39,'PARTIAL':61}
 rows.sort(key=lambda r:(r['capture_time'],int(r['frame'])));assert all(int(a['frame'])<int(b['frame']) for a,b in zip(rows,rows[1:]));assert all(float(am[int(a['frame'])]['frame_mono'])<float(am[int(b['frame'])]['frame_mono']) for a,b in zip(rows,rows[1:]));assert len({x['pose_group'] for x in rows})==36
 for x in ['amodal_masks','xyz_raw_front','xyz_raw_back','visible_candidates','previews','review_package','review_package/frames']:(OUT/x).mkdir()
 byframe={int(r['frame']):r for r in rows}; adapter=pipeline.BBAdapter(byframe); cache={};quality=[];groups={}
 for r in rows:
  frame=int(r['frame']);g=r['pose_group'];groups.setdefault(g,frame)
 for g,frame in groups.items():
  s=adapter.load(frame);m=s['cad'];assert m.any() and np.isfinite(s['near'][m]).all() and np.isfinite(s['far'][m]).all() and np.all(s['near'][m]<=s['far'][m]);assert all(not x.any() for k,x in s['gates'].items() if k in ('u_front','u_back','q_pos','q_neg'))
  mask=m.astype(np.uint8)*255;png(OUT/'amodal_masks'/f'pose_{g}.png',mask);png(OUT/'xyz_raw_front'/f'pose_{g}.png',s['front']);png(OUT/'xyz_raw_back'/f'pose_{g}.png',s['back'])
  ts=adapter.tensors(s);assert tuple(ts['front'].shape)==(24,128,128) and tuple(ts['back'].shape)==(24,128,128) and all(float(ts[k].sum())==0 for k in ('u_front','u_back','q_pos','q_neg'))
  # quantized reprojection using the uncorrected pose, replicating the established CPU renderer convention.
  y,x=np.nonzero(m); info=adapter.info;lo=np.array([info['min_'+a] for a in 'xyz']);scale=np.array([info['size_'+a] for a in 'xyz']);errors=[]
  for nm in ('front','back'):
   xyz=s[nm][m].astype(float)/255*scale+lo;cam=xyz@s['R'].T+s['t'];proj=cam@s['K'].T;uv=proj[:,:2]/proj[:,2,None];errors.append(float(np.quantile(np.linalg.norm(uv-np.c_[x,y],axis=1),.99)))
  cache[g]={'pixels':int(m.sum()),'bbox':s['bbox'].tolist(),'p99_front':errors[0],'p99_back':errors[1]}
  if frame in [43,75,235,247] or len(cache)==1:
   im=s['rgb'].copy();overlay=im.copy();overlay[m]=(overlay[m]*.55+np.array([0,0,255])*.45).astype('uint8');pan=np.hstack([cv2.resize(overlay,(640,360)),cv2.resize(s['front'],(640,360)),cv2.resize(s['back'],(640,360))]);cv2.putText(pan,'AMODAL NOT GT',(5,25),0,.7,(255,255,255),2);cv2.putText(pan,'UNCANONICALIZED FRONT',(645,25),0,.65,(255,255,255),2);cv2.putText(pan,'UNCANONICALIZED BACK',(1285,25),0,.65,(255,255,255),2);assert cv2.imwrite(str(OUT/'previews'/f'{frame:06d}.jpg'),pan)
 for r in rows:
  frame=int(r['frame']);g=r['pose_group'];s=cache[g];quality.append({'frame':frame,'occlusion':r['occlusion'],'pose_group':g,'amodal_mask':f'amodal_masks/pose_{g}.png','front_uncanonicalized':f'xyz_raw_front/pose_{g}.png','back_uncanonicalized':f'xyz_raw_back/pose_{g}.png','amodal_pixels':s['pixels'],'amodal_fraction_of_image':s['pixels']/(1280*720),'valid_supervision_pixels':0,'valid_supervision_fraction':0,'front_reprojection_p99_px':s['p99_front'],'back_reprojection_p99_px':s['p99_back'],'mask_kind':'CAD_amodal_not_visible_GT','quality_status':'REVIEW_REQUIRED','reason':'visibility and symmetry canonicalization not verified'})
 wc('target_quality.csv',quality)
 # 4 existing SAM candidates only; intersections are proposals, not supervision.
 samdir=P/'bb_sam_candidates_20261108/results/run_20261108T140008_86355c19';samstats=[]
 for i in (43,75,235,247):
  candidate=cv2.imread(str(samdir/f'{i:06d}_pseudo_mask_candidate_selected.png'),0);assert candidate is not None and candidate.shape==(720,1280)
  cad=cv2.imread(str(OUT/'amodal_masks'/f"pose_{byframe[i]['pose_group']}.png"),0);inter=((candidate>0)&(cad>0)).astype('uint8')*255
  png(OUT/'visible_candidates'/f'{i:06d}_SAM_INTERSECT_CAD_NOT_GT.png',inter)
  samstats.append({'frame':i,'sam_candidate':str(samdir/f'{i:06d}_pseudo_mask_candidate_selected.png'),'cad_sam_intersection_candidate':f'visible_candidates/{i:06d}_SAM_INTERSECT_CAD_NOT_GT.png','candidate_pixels':int((candidate>0).sum()),'intersection_pixels':int((inter>0).sum()),'review_status':'UNREVIEWED_NOT_GT'})
 wc('visible_candidate_manifest.csv',samstats)
 # Sequence, pose changes, amodal-ROI features and boundary proposals. All 99 boundaries may be reviewed.
 plac=1;prows=[];bound=[];imprev=None;prev=None
 for index,r in enumerate(rows):
  frame=int(r['frame']);im=cv2.imread(r['image_path']);assert im is not None
  shutil.copy2(P/'bb100_visual_review_20261109/comparisons'/f'{frame:06d}.jpg',OUT/'review_package'/'frames'/f'{frame:06d}.jpg')
  T=np.load(r['pose_path'],allow_pickle=False)
  if prev is not None:
   same=bool(np.array_equal(T,prev['T']));
   if not same:plac+=1
   rot=T[:3,:3]@prev['T'][:3,:3].T;angle=float(np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))));trans=float(np.linalg.norm(T[:3,3]-prev['T'][:3,3]));dt=float(am[frame]['frame_mono'])-float(am[prev['frame']]['frame_mono'])
   mask=cv2.imread(str(OUT/'amodal_masks'/f"pose_{prev['row']['pose_group']}.png"),0);mae,n,dx,dy,disp,edge=roi_features(imprev,im,mask)
   def origin(T,K):
    p=K@T[:3,3];return p[:2]/p[2]
   K=np.array(json.loads(r['K']));proj_delta=float(np.linalg.norm(origin(T,K)-origin(prev['T'],K)))
   bound.append({'boundary_id':f"B{index:03d}",'before_frame':prev['frame'],'after_frame':frame,'before_time':prev['row']['capture_time'],'after_time':r['capture_time'],'seconds':dt,'exact_pose_equal':same,'rotation_deg':angle,'translation_cad_units':trans,'gt_origin_projection_change_px':proj_delta,'cad_roi_gray_mae':mae,'klt_tracks':n,'klt_median_dx_px':dx,'klt_median_dy_px':dy,'klt_median_displacement_px':disp,'edge_median_distance_px':edge,'auto_candidate':'POSE_CHANGE' if not same else 'SAME_POSE_CHECK','review_status':'UNREVIEWED','interpretation':'ROI may contain gripper/reflection; diagnostics do not prove object movement'})
  prows.append({'capture_order':index+1,'frame':frame,'capture_time':r['capture_time'],'placement_id_candidate':f'placement_{plac:03d}','pose_group':r['pose_group'],'connected_group':r['connected_group'],'occlusion':r['occlusion'],'review_status':'UNREVIEWED','approved_split':'UNASSIGNED'})
  imprev=im;prev={'T':T,'frame':frame,'row':r}
 wc('placement_candidates.csv',prows);wc('pose_changes.csv',bound)
 # Exact pose repeats, duplicate RGB, and previous conservative connected groups constrain proposals.
 bygroup=collections.defaultdict(set)
 for r in prows:bygroup[r['pose_group']].add(r['placement_id_candidate'])
 repeated={k:sorted(v) for k,v in bygroup.items() if len(v)>1}
 split=[]
 for r in prows:
  role='validation' if r['connected_group']=='31' else 'train'
  split.append({'frame':r['frame'],'placement_id_candidate':r['placement_id_candidate'],'pose_group':r['pose_group'],'connected_group':r['connected_group'],'occlusion':r['occlusion'],'tentative_64_36':role,'approved_split':'UNASSIGNED'})
 wc('split_candidate_not_approved.csv',split)
 train={r['frame'] for r in split if r['tentative_64_36']=='train'};val={r['frame'] for r in split if r['tentative_64_36']=='validation'}
 overlaps={k:len({r[k] for r in split if r['frame'] in train}&{r[k] for r in split if r['frame'] in val}) for k in ('pose_group','connected_group','placement_id_candidate')};assert not any(overlaps.values())
 hashes=collections.defaultdict(set)
 for r in rows:hashes[r['image_sha256']].add('train' if int(r['frame']) in train else 'validation')
 assert not any(len(v)>1 for v in hashes.values())
 wj('split_audit.json',{'status':'REVIEW_REQUIRED','seed':'not applicable: only two preexisting connected components, no random split','candidate_train':len(train),'candidate_val':len(val),'overlaps':overlaps,'exact_image_hash_cross_split':0,'exact_pose_revisits_across_contiguous_runs':repeated,'physical_independence':'NOT VERIFIED','70_30_80_20':'not achievable from two conservative components without splitting one','group_kfold':'with 2 conservative groups only 2 folds; physical independence unverified'})
 # Build portable static HTML: decision semantics mean boundary indicates distinct consecutive placements.
 pkg=OUT/'review_package'; records=[]
 for b in bound:
  a=dict(b);a['before_img']=f"frames/{b['before_frame']:06d}.jpg";a['after_img']=f"frames/{b['after_frame']:06d}.jpg";records.append(a)
 (pkg/'boundaries.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 data=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
 template=(OUT/'review_template.html').read_text(encoding='utf-8');(pkg/'index.html').write_text(template.replace('DATA_PLACEHOLDER',data),encoding='utf-8')
 shutil.copy2(OUT/'pose_changes.csv',pkg/'pose_changes.csv');shutil.copy2(OUT/'placement_candidates.csv',pkg/'placement_candidates.csv')
 (pkg/'사용법.txt').write_text('ZIP 전체를 Windows에서 압축 해제하고 index.html을 Chrome에서 여세요. 후보 진단은 자동 확정이 아닙니다. ACCEPT=두 연속 프레임 사이 실제 객체 재배치 경계 인정, REJECT=경계 아님, UNCERTAIN=판단 불가. 동일 Pose 경계도 검수 가능합니다. CSV 내보내기로 자주 저장하고 이후 CSV 가져오기로 복구하세요. CSV를 서버에 전달해도 수동 검토 전 최종 분할이 자동 확정되지 않습니다. 시각 검수는 정밀 GT 인증이 아닙니다.\n',encoding='utf-8')
 assert len(list((pkg/'frames').glob('*.jpg')))==100
 for r in records:
  assert (pkg/r['before_img']).is_file() and (pkg/r['after_img']).is_file()
 with zipfile.ZipFile(OUT/'BB100_placement_review_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(pkg.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(pkg))
 with zipfile.ZipFile(OUT/'BB100_placement_review_windows.zip') as z:
  assert z.testzip() is None;names=set(z.namelist());assert all(r['before_img'] in names and r['after_img'] in names for r in records)
 stats={'frames':100,'order_frame_and_monotonic_time':'PASS','unique_poses':len(groups),'contiguous_pose_runs_candidate':plac,'boundaries':len(bound),'pose_change_boundaries':sum(not b['exact_pose_equal'] for b in bound),'amodal_unique_renders':len(cache),'sam_candidates':len(samstats),'camera_K':'all match Phase14','cad_unit':'BOP convention mm; actual physical BB scale unverified','p99_reprojection_px_max':max(max(v['p99_front'],v['p99_back']) for v in cache.values()),'output_zip_references':'PASS','training_supervision_pixels':0,'elapsed_sec':round(time.monotonic()-start,2)}
 wj('build_summary.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
