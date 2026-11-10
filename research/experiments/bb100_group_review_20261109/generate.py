"""BB100 group review package; image deltas are unvalidated prioritization clues."""
import csv,json,hashlib,sys,collections,shutil
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps
P=Path(__file__).resolve().parent;E=P.parent;A=E/'bb100_gt_camera_verification_20261109';V=E/'bb100_visual_review_20261109';B=E/'real_bb100_20261108/BB_obj04';PK=P/'package';sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def patch(img,box):return img[box[1]:box[3],box[0]:box[2]]
def klt(ref,cur,mask):
 features=cv2.goodFeaturesToTrack(ref,maxCorners=150,qualityLevel=.01,minDistance=4,mask=mask,blockSize=5)
 if features is None or len(features)==0:return 0,None,None,None
 nxt,status,_=cv2.calcOpticalFlowPyrLK(ref,cur,features,None,winSize=(21,21),maxLevel=3)
 if nxt is None:return 0,None,None,None
 back,stat2,_=cv2.calcOpticalFlowPyrLK(cur,ref,nxt,None,winSize=(21,21),maxLevel=3)
 if back is None:return 0,None,None,None
 f=features.reshape(-1,2);q=nxt.reshape(-1,2);b=back.reshape(-1,2);h,w=mask.shape;ix=np.rint(q[:,0]).astype(int);iy=np.rint(q[:,1]).astype(int);inside=(ix>=0)&(ix<w)&(iy>=0)&(iy<h);inside_idx=np.flatnonzero(inside);inside[inside_idx]&=mask[iy[inside_idx],ix[inside_idx]]>0;good=(status.ravel()>0)&(stat2.ravel()>0)&(np.linalg.norm(f-b,axis=1)<1)&inside
 if not good.any():return 0,None,None,None
 d=q[good]-f[good];return int(good.sum()),float(np.median(d[:,0])),float(np.median(d[:,1])),float(np.median(np.linalg.norm(d,axis=1)))
def edge_change(ref,cur,mask):
 a=cv2.Canny(ref,50,120)>0;b=cv2.Canny(cur,50,120)>0;inside=mask>0;a&=inside;b&=inside
 if a.sum()<20 or b.sum()<20:return None,int(a.sum()),int(b.sum())
 da=cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,3);db=cv2.distanceTransform((~b).astype('uint8'),cv2.DIST_L2,3);return float((np.median(db[a])+np.median(da[b]))/2),int(a.sum()),int(b.sum())
def main():
 cv2.setNumThreads(2);audit=read(A/'bb100_frame_audit.csv');groups=read(V/'pose_groups.csv');reviews=read(V/'review_template.csv');assert len(audit)==len(reviews)==100 and len(groups)==36;by={int(r['frame']):r for r in audit};rev={int(r['frame']):r for r in reviews};assert all(r['review_status']=='UNREVIEWED' and r['training_status']=='HOLD' for r in reviews);gt=json.loads((B/'scene_gt_obj04.json').read_text());cams=json.loads((B/'scene_camera_obj04.json').read_text());v,faces,_,_=symmetry.load();sources=[A/'bb100_frame_audit.csv',V/'pose_groups.csv',V/'review_template.csv',B/'scene_gt_obj04.json',B/'scene_camera_obj04.json']+[Path(r['image_labeling_path']) for r in audit]+[Path(r['overlay_path']) for r in audit]+[V/r['comparison_image'] for r in reviews];before={str(p):sha(p) for p in sources};(PK/'frames').mkdir();(PK/'groups').mkdir()
 for r in reviews:shutil.copyfile(V/r['comparison_image'],PK/'frames'/f"{int(r['frame']):06d}.jpg")
 diagnostics=[];metadata=[];frame_template=[];group_template=[]
 for n,g in enumerate(groups):
  gid=int(g['pose_group']);ids=[int(s) for s in g['frames'].split(',')];assert len(ids)==int(g['n']) and all(int(by[i]['pose_exact_group'])==gid and int(rev[i]['pose_group'])==gid for i in ids);first=ids[0];z=gt[str(first)][0];R=np.array(z['cam_R_m2c']).reshape(3,3);t=np.array(z['cam_t_m2c']);K=np.array(cams[str(first)]['cam_K']).reshape(3,3)
  for i in ids:
   a=gt[str(i)][0];assert np.array_equal(np.array(a['cam_R_m2c']).reshape(3,3),R) and np.array_equal(np.array(a['cam_t_m2c']),t)
  pc=v@R.T+t;uv=pc@K.T;uv=uv[:,:2]/uv[:,2:];bounds=np.r_[uv.min(0),uv.max(0)];center=(bounds[:2]+bounds[2:])/2;side=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*1.55;box=(max(0,int(center[0]-side/2)),max(0,int(center[1]-side/2)),min(1280,int(center[0]+side/2)),min(720,int(center[1]+side/2)));assert box[2]>box[0] and box[3]>box[1]
  maskfull=np.zeros((720,1280),np.uint8)
  for poly in np.rint(uv[faces]).astype(np.int32):cv2.fillConvexPoly(maskfull,poly,255)
  maskfull=cv2.erode(maskfull,np.ones((5,5),np.uint8),iterations=1);mask=patch(maskfull,box);assert (mask>0).sum()>20
  ref=cv2.imread(by[first]['image_labeling_path']);ref_gray=cv2.cvtColor(patch(ref,box),cv2.COLOR_BGR2GRAY);local=[]
  for j,i in enumerate(ids):
   im=cv2.imread(by[i]['image_labeling_path']);assert im is not None;gray=cv2.cvtColor(patch(im,box),cv2.COLOR_BGR2GRAY);inside=mask>0;delta=gray.astype(np.float32)-ref_gray.astype(np.float32);raw=float(np.mean(np.abs(delta[inside])));offset=float(np.median(delta[inside]));corrected=float(np.mean(np.abs(delta[inside]-offset)));ed,ne1,ne2=edge_change(ref_gray,gray,mask);count,dx,dy,flow=klt(ref_gray,gray,mask) if i!=first else (0,None,None,None);dt=abs(float(by[i]['frame_mono'])-float(by[first]['frame_mono']));rec={'pose_group':gid,'frame':i,'sequence_index':j,'capture_time':by[i]['frame_time'],'seconds_from_first':dt,'state':by[i]['state'],'previous_priority':by[i]['priority'],'reference_frame':first,'roi_xyxy':json.dumps(box),'interior_pixels':int(inside.sum()),'roi_gray_mae_vs_first':raw,'roi_gray_median_brightness_shift':offset,'roi_gray_mae_after_median_shift':corrected,'edge_symmetric_median_distance_px':ed if ed is not None else '', 'edge_reference_count':ne1,'edge_target_count':ne2,'klt_consistent_feature_count':count,'klt_median_dx_px':dx if dx is not None else '', 'klt_median_dy_px':dy if dy is not None else '', 'klt_median_displacement_px':flow if flow is not None else '', 'actual_motion':'UNKNOWN','occlusion':'UNKNOWN','reflection':'UNKNOWN','boundary':'UNKNOWN','interpretation':'unvalidated image changes; amodal support may include gripper and specular effects, no motion conclusion'};diagnostics.append(rec);local.append(rec)
   frame_template.append({'pose_group':gid,'frame':i,'sequence_index':j,'capture_time':by[i]['frame_time'],'state':by[i]['state'],'previous_priority':by[i]['priority'],'frame_status':'UNREVIEWED','motion_review':'UNKNOWN','occlusion':'UNKNOWN','reflection':'UNKNOWN','boundary':'UNKNOWN','blur':'UNKNOWN','reviewer':'','reviewed_at':'','notes':'','training_status':'HOLD','phase12_frame_image':f'frames/{i:06d}.jpg'})
  maxi=max(local,key=lambda x:x['roi_gray_mae_vs_first']);chosen=[]
  for i in [first,ids[len(ids)//2],ids[-1],maxi['frame']]:
   if i not in chosen:chosen.append(i)
  if len(ids)==1:assert len(chosen)==1
  panel=Image.new('RGB',(max(1,len(chosen))*340,775),(14,22,28));draw=ImageDraw.Draw(panel);draw.text((10,8),f'POSE GROUP {gid:03d}  |  frames (capture order): '+','.join(map(str,ids)),fill='white');draw.text((10,28),'Identical numeric pose is INTENTIONAL reuse; actual object motion UNKNOWN. Top: label RGB; bottom: CAD amodal overlay.',fill='#ffd166');draw.text((10,48),f'Fixed pixel crop {box}; max ROI delta vs first: {maxi["frame"]} ({maxi["roi_gray_mae_vs_first"]:.2f}/255). Gripper/reflection may dominate.',fill='white')
  for j,i in enumerate(chosen):
   im=Image.open(by[i]['image_labeling_path']).convert('RGB').crop(box);ov=Image.open(by[i]['overlay_path']).convert('RGB').crop(box);im=ImageOps.contain(im,(320,320));ov=ImageOps.contain(ov,(320,320));panel.paste(im,(j*340+(340-im.width)//2,100+(320-im.height)//2));panel.paste(ov,(j*340+(340-ov.width)//2,440+(320-ov.height)//2));tags=[]
   if i==first:tags.append('first/representative')
   if i==ids[len(ids)//2]:tags.append('middle')
   if i==ids[-1]:tags.append('last')
   if i==maxi['frame']:tags.append('max delta')
   draw.text((j*340+8,76),f'{i:03d} {by[i]["state"]} [{"/".join(tags)}]',fill='#71e2fd');draw.text((j*340+8,421),f'{by[i]["frame_time"]}',fill='white')
  path=f'groups/{gid:06d}.jpg';panel.save(PK/path,quality=90)
  anyp=any(by[i]['priority']=='P1' for i in ids);known=43 in ids and 60 in ids;metadata.append({'pose_group':gid,'frame_count':len(ids),'frames_capture_order':','.join(map(str,ids)),'capture_first':g['capture_first'],'capture_last':g['capture_last'],'states':g['states'],'connected_group':g['connected_groups'],'representative_frame':first,'middle_frame':ids[len(ids)//2],'last_frame':ids[-1],'max_delta_frame':maxi['frame'],'selected_panel_frames':','.join(map(str,chosen)),'max_roi_gray_mae':maxi['roi_gray_mae_vs_first'],'max_brightness_corrected_mae':max(x['roi_gray_mae_after_median_shift'] for x in local),'p1_frame_count':sum(by[i]['priority']=='P1' for i in ids),'prior_observed_tool_overlap_frames':','.join(str(i) for i in ids if 'gripper overlaps' in by[i]['occlusion_status'].lower()),'contains_43_60_copy_history':known,'actual_object_motion':'UNKNOWN','boundary_misalignment':'UNKNOWN','occlusion':'UNKNOWN','reflection':'UNKNOWN','numeric_pose_reuse':'INTENTIONAL per user; motion needs review','pose_R_m2c':json.dumps(R.tolist()),'pose_t_m2c':json.dumps(t.tolist()),'group_comparison_image':path,'training_status':'HOLD'})
  group_template.append({'pose_group':gid,'frame_count':len(ids),'frames_capture_order':','.join(map(str,ids)),'group_status':'UNREVIEWED','motion_review':'UNKNOWN','reviewer':'','reviewed_at':'','notes':'','training_status':'HOLD'})
  if (n+1)%9==0:print('GROUPS',n+1,flush=True)
 # Copy original group/frame ID order and retain conservative group mapping; do not promote repeats to error.
 assert len(diagnostics)==len(frame_template)==100 and len(metadata)==len(group_template)==36 and before=={x:sha(Path(x)) for x in before};write(PK/'group_metadata.csv',metadata);write(PK/'motion_diagnostics.csv',diagnostics);write(PK/'frame_review_template.csv',frame_template);write(PK/'group_review_template.csv',group_template);write(P/'group_metadata.csv',metadata);write(P/'motion_diagnostics.csv',diagnostics);write(P/'frame_review_template.csv',frame_template)
 manifest={'frames':100,'groups':36,'singletons':sum(x['frame_count']==1 for x in metadata),'repeated_groups':sum(x['frame_count']>1 for x in metadata),'repeated_frames':sum(x['frame_count'] for x in metadata if x['frame_count']>1),'user_statement':'unchanged pose intentionally reused when object stationary; no assertion all frames stationary','phase12_review_export':'not found in searched Phase12 directory; localStorage not inspectable; explicit CSV import available','source_files':len(before),'source_sha256':before,'source_unchanged':True,'GT_acceptance':'not inferred; all statuses UNREVIEWED, HOLD','metrics':'heuristics on fixed projected object neighborhood; not motion detection, occluder/lighting may dominate','packages':'standalone HTML embeds group metadata; relative local images only; no network required'};json.dump(manifest,(P/'manifest.json').open('x'),indent=2,ensure_ascii=False);print('PASS groups',len(metadata),'frames',len(frame_template),flush=True)
if __name__=='__main__':main()
