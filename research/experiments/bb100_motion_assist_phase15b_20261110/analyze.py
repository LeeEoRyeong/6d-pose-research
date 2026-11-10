from pathlib import Path
import os,sys,csv,json,hashlib,collections,zipfile,shutil,time
import numpy as np,cv2
P=Path('/home/oylee/projects/HCCEPose_experiments'); OLD=P/'bb100_placement_targets_phase15_20261109';OUT=Path(__file__).resolve().parent
cv2.setNumThreads(1);cv2.setRNGSeed(42)
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def H(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def mxy(kp):return np.array([x.pt for x in kp],np.float32)
def match(fa,fb,region,detector):
 ka,da=fa[detector];kb,db=fb[detector]
 if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
 norm=cv2.NORM_L2 if detector=='SIFT' else cv2.NORM_HAMMING;matcher=cv2.BFMatcher(norm)
 pairs=matcher.knnMatch(da,db,k=2);best={}
 for pair in pairs:
  if len(pair)<2:continue
  a,b=pair
  if a.distance>=.75*b.distance:continue
  x,y=ka[a.queryIdx].pt;u,v=kb[a.trainIdx].pt;ix,iy=int(round(x)),int(round(y));iu,iv=int(round(u)),int(round(v))
  if not (0<=ix<1280 and 0<=iu<1280 and 0<=iy<720 and 0<=iv<720):continue
  if not (region[0][iy,ix] and region[1][iv,iu]):continue
  if a.trainIdx not in best or a.distance<best[a.trainIdx].distance:best[a.trainIdx]=a
 ms=list(best.values());return np.array([ka[m.queryIdx].pt for m in ms],np.float32).reshape(-1,2),np.array([kb[m.trainIdx].pt for m in ms],np.float32).reshape(-1,2)
def fit(a,b,bg=False):
 if len(a)<(8 if bg else 4):return None,None,0,0,None,0
 cv2.setRNGSeed(42)
 if bg:
  M,ins=cv2.findHomography(a,b,cv2.RANSAC,2.0,maxIters=2000,confidence=.995)
 else:
  M,ins=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=2000,confidence=.995,refineIters=10)
  if M is not None:M=np.vstack([M,[0,0,1]])
 if M is None or ins is None or not np.isfinite(M).all():return None,None,0,0,None,0
 mask=ins.ravel().astype(bool);n=int(mask.sum());uv=cv2.perspectiveTransform(a[:,None,:],M)[:,0,:];err=np.linalg.norm(uv-b,axis=1);res=float(np.median(err[mask])) if n else None
 # coverage of 4x3 image cells for background, 3x3 ROI extent for object.
 if n:
  pts=a[mask]; mn=np.array([0,0]) if bg else a.min(0);mx=np.array([1280,720]) if bg else a.max(0);dim=np.array([4,3]) if bg else np.array([3,3]);cells=np.floor((pts-mn)/np.maximum(mx-mn,1)*dim).astype(int);cells=np.clip(cells,[0,0],dim-1);coverage=len(set(map(tuple,cells)))
 else:coverage=0
 return M,mask,n,n/len(a),res,coverage
def transform(M,pt):
 if M is None:return None
 out=M@np.r_[pt,1.];return out[:2]/out[2] if abs(out[2])>1e-8 else None
def rounded(v):return None if v is None else round(float(v),4)
def main():
 start=time.monotonic();frames=read(P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv');byid={int(r['frame']):r for r in frames};bounds=read(OLD/'pose_changes.csv');assert len(bounds)==99 and len(byid)==100
 for x in ['package','package/panels','package/frames']:(OUT/x).mkdir()
 sift=cv2.SIFT_create(nfeatures=3500);orb=cv2.ORB_create(nfeatures=5000);features={};imgs={};masks={}
 for fid,row in byid.items():
  im=cv2.imread(row['image_path']);assert im is not None;gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=cv2.imread(str(OLD/'amodal_masks'/f"pose_{row['pose_group']}.png"),0);assert mask is not None
  # Conservative erosion does not certify visibility: gripper can still be inside.
  mask=cv2.erode(mask,np.ones((7,7),np.uint8))>0
  imgs[fid]=im;masks[fid]=mask;features[fid]={'SIFT':sift.detectAndCompute(gray,None),'ORB':orb.detectAndCompute(gray,None)}
  shutil.copy2(OLD/'review_package/frames'/f'{fid:06d}.jpg',OUT/'package/frames'/f'{fid:06d}.jpg')
 results=[]; data=[]
 for index,item in enumerate(bounds):
  fa,fb=int(item['before_frame']),int(item['after_frame']);im1,im2=imgs[fa],imgs[fb];m1,m2=masks[fa],masks[fb];
  # Exclude both projected object footprints and large margins from both background frames.
  exclusion=cv2.dilate((m1|m2).astype('uint8'),np.ones((151,151),np.uint8))>0; bg=(~exclusion,~exclusion)
  detail={};visual=None
  for method in ('SIFT','ORB'):
   a,b=match(features[fa],features[fb],(m1,m2),method);M,ins,n,ratio,res,cov=fit(a,b)
   detail[method]=(a,b,M,ins,n,ratio,res,cov)
   if method=='SIFT':visual=(a,b,ins)
  ba,bb=match(features[fa],features[fb],bg,'SIFT');BM,bi,bn,br,bres,bcov=fit(ba,bb,True)
  bg_ok=bool(BM is not None and bn>=12 and br>=.5 and bcov>=6 and bres is not None and bres<=2)
  maskunion=m1|m2;ys,xs=np.nonzero(maskunion);assert len(xs)>0;center=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2]);d={};
  for method in ('SIFT','ORB'):
   a,b,M,ins,n,ratio,res,cov=detail[method];point=transform(M,center);back=transform(BM,center)
   d[method]=None if point is None or back is None else point-back
  agreement=None
  if all(d[m] is not None for m in ('SIFT','ORB')):agreement=float(np.linalg.norm(d['SIFT']-d['ORB']))
  # Edge distance is descriptive; neither edge nor KLT is verified as belonging to object.
  g1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY);g2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY);e1=cv2.Canny(g1,70,150)>0;e2=cv2.Canny(g2,70,150)>0
  edge=None
  if np.any(e1&m1) and np.any(e2&m2):edge=float(np.median(cv2.distanceTransform((~e2).astype('uint8'),cv2.DIST_L2,3)[e1&m1]))
  # ECC is a supplementary crop-only photometric fit; never used for decision.
  x0=max(0,xs.min()-35);x1=min(1280,xs.max()+36);y0=max(0,ys.min()-35);y1=min(720,ys.max()+36);ecc=None
  try:
   z1=g1[y0:y1,x0:x1].astype('float32')/255;z2=g2[y0:y1,x0:x1].astype('float32')/255;W=np.eye(2,3,dtype='float32');ecc,W=cv2.findTransformECC(z1,z2,W,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-4),None,3);ecc=float(ecc)
  except cv2.error:pass
  s=detail['SIFT'];o=detail['ORB'];identical=byid[fa]['image_sha256']==byid[fb]['image_sha256'];
  # No calibrated visible-object mask or stationary/moved reference distribution.
  label='LIKELY_STATIONARY' if identical else 'UNCERTAIN'
  reason='identical JPEG SHA-256: image observations identical; physical interval unverified' if identical else ('background registration quality insufficient; object support/visibility unverified' if not bg_ok else 'CAD amodal interior may include gripper or holes; object ownership and motion thresholds unvalidated')
  row={'boundary_id':item['boundary_id'],'before_frame':fa,'after_frame':fb,'occlusion_before':byid[fa]['occlusion'],'occlusion_after':byid[fb]['occlusion'],'mask_provenance':'eroded_CAD_amodal_not_visible_GT','image_sha256_equal':identical,'bg_sift_matches':len(ba),'bg_inliers':bn,'bg_inlier_ratio':rounded(br),'bg_median_residual_px':rounded(bres),'bg_coverage_4x3':bcov,'bg_quality_heuristic':bg_ok,'sift_matches':len(s[0]),'sift_inliers':s[4],'sift_inlier_ratio':rounded(s[5]),'sift_median_residual_px':rounded(s[6]),'sift_coverage_3x3':s[7],'orb_matches':len(o[0]),'orb_inliers':o[4],'orb_inlier_ratio':rounded(o[5]),'orb_median_residual_px':rounded(o[6]),'orb_coverage_3x3':o[7],'sift_relative_dx_px':rounded(None if d['SIFT'] is None else d['SIFT'][0]),'sift_relative_dy_px':rounded(None if d['SIFT'] is None else d['SIFT'][1]),'sift_relative_displacement_px':rounded(None if d['SIFT'] is None else np.linalg.norm(d['SIFT'])),'orb_relative_dx_px':rounded(None if d['ORB'] is None else d['ORB'][0]),'orb_relative_dy_px':rounded(None if d['ORB'] is None else d['ORB'][1]),'orb_relative_displacement_px':rounded(None if d['ORB'] is None else np.linalg.norm(d['ORB'])),'method_disagreement_px':rounded(agreement),'ecc_score':rounded(ecc),'roi_edge_median_distance_px':rounded(edge),'auto_label':label,'reason':reason,'review_status':'UNREVIEWED','gt_rotation_deg_supplement':item['rotation_deg'],'gt_translation_cad_units_supplement':item['translation_cad_units'],'gt_exact_pose_equal_supplement':item['exact_pose_equal']}
  results.append(row)
  # panels: before and after crop, 50% overlay, raw abs diff, after inverse aligned by object model, its diff, SIFT inliers side by side.
  crop1=im1[y0:y1,x0:x1];crop2=im2[y0:y1,x0:x1];blend=cv2.addWeighted(crop1,.5,crop2,.5,0);diff=cv2.absdiff(crop1,crop2);M=s[2]
  aligned=cv2.warpPerspective(im2,np.linalg.inv(M),(1280,720)) if M is not None and abs(np.linalg.det(M))>1e-10 else im2
  ac=aligned[y0:y1,x0:x1];ab=cv2.addWeighted(crop1,.5,ac,.5,0);ad=cv2.absdiff(crop1,ac)
  def tile(im,title):
   out=cv2.resize(im,(360,260));cv2.rectangle(out,(0,0),(360,25),(0,0,0),-1);cv2.putText(out,title,(5,18),0,.48,(255,255,255),1,cv2.LINE_AA);return out
  top=np.hstack([tile(crop1,'BEFORE RGB'),tile(crop2,'AFTER RGB'),tile(blend,'BEFORE ALIGN 50/50'),tile(diff,'BEFORE ALIGN DIFF')]);bottom=np.hstack([tile(ab,'SIFT ALIGNED 50/50' if M is not None else 'NO SIFT MODEL - UNALIGNED'),tile(ad,'SIFT ALIGNED DIFF' if M is not None else 'NO SIFT MODEL - RAW DIFF')]);
  # Connections in crop coordinates; object membership is unverified.
  matchpanel=np.hstack([cv2.resize(crop1,(360,260)),cv2.resize(crop2,(360,260))]);a,b,ins=visual
  if ins is not None:
   for A,B in list(zip(a[ins],b[ins]))[:60]:
    pa=(int((A[0]-x0)*360/(x1-x0)),int((A[1]-y0)*260/(y1-y0)));pb=(360+int((B[0]-x0)*360/(x1-x0)),int((B[1]-y0)*260/(y1-y0)));cv2.line(matchpanel,pa,pb,(0,255,255),1,cv2.LINE_AA)
  cv2.rectangle(matchpanel,(0,0),(720,25),(0,0,0),-1);cv2.putText(matchpanel,'SIFT INLIERS (CAD REGION; NOT VERIFIED)',(5,18),0,.48,(255,255,255),1,cv2.LINE_AA)
  panel=np.vstack([top,np.hstack([bottom,matchpanel])]);cv2.imwrite(str(OUT/'package/panels'/f"{item['boundary_id']}.jpg"),panel,[cv2.IMWRITE_JPEG_QUALITY,86])
  v=dict(row);v['panel']=f"panels/{item['boundary_id']}.jpg";v['old_before']=f'frames/{fa:06d}.jpg';v['old_after']=f'frames/{fb:06d}.jpg';data.append(v)
  if index%20==0:print('boundary',index+1,flush=True)
 write(OUT/'boundary_analysis.csv',results)
 # Conservative connectivity: only independently established actual boundaries can split, so both UNCERTAIN and LIKELY labels merge.
 parent={i:i for i in byid}
 def root(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):parent[root(a)]=root(b)
 for r in results:union(r['before_frame'],r['after_frame'])
 component=collections.defaultdict(list)
 for i in byid:component[root(i)].append(i)
 counts=collections.Counter(r['auto_label'] for r in results);bgfail=sum(not r['bg_quality_heuristic'] for r in results);both=collections.Counter((r['occlusion_before'],r['occlusion_after']) for r in results)
 valid=[r for r in results if r['bg_quality_heuristic'] and r['sift_relative_displacement_px'] is not None and r['orb_relative_displacement_px'] is not None];
 def quant(k):
  a=[r[k] for r in valid if r[k] is not None];return {'n':len(a),'p10':rounded(np.quantile(a,.1)),'median':rounded(np.median(a)),'p90':rounded(np.quantile(a,.9))} if a else {'n':0}
 summary={'frames':100,'boundaries':len(results),'label_counts':dict(counts),'mandatory_review_conservative':99,'mandatory_review_if_identical_pixel_boundary_triaged_as_low_priority':sum(r['auto_label']=='UNCERTAIN' for r in results),'why_99':'No independently validated object visibility or stationary/moved truth; even identical endpoint pixels do not prove physical interval','bg_quality_failed':bgfail,'quality_heuristic_not_calibrated':True,'stratification':{'/'.join(k):v for k,v in both.items()},'bg_valid_both_methods_count':len(valid),'sift_relative_displacement_px':quant('sift_relative_displacement_px'),'orb_relative_displacement_px':quant('orb_relative_displacement_px'),'method_disagreement_px':quant('method_disagreement_px'),'conservative_components':[sorted(v) for v in component.values()],'independent_split':'BLOCKED','elapsed_sec':round(time.monotonic()-start,2)}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(OUT/'package'/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');
 template=(OUT/'template.html').read_text();(OUT/'package'/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8');shutil.copy2(OUT/'boundary_analysis.csv',OUT/'package'/'boundary_analysis.csv')
 (OUT/'package'/'사용법.txt').write_text('압축 전체를 해제하고 index.html을 오프라인 Chrome으로 여세요. 자동 분류는 정답이 아니며 원본 Phase 15 검수 CSV를 가져오기 전에는 사람이 검수한 결과가 없습니다. CSV를 자주 내보내세요. 경계 ACCEPT=객체 재배치, REJECT=객체가 정지, UNCERTAIN=판단 불가. 비인접 동일 배치도 별도 확인해야 합니다.\n',encoding='utf-8')
 with zipfile.ZipFile(OUT/'BB100_motion_assist_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted((OUT/'package').rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT/'package'))
 with zipfile.ZipFile(OUT/'BB100_motion_assist_windows.zip') as z:
  assert not z.testzip();names=set(z.namelist());assert all(r['panel'] in names and r['old_before'] in names and r['old_after'] in names for r in data)
 print(json.dumps({k:v for k,v in summary.items() if k!='conservative_components'},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
