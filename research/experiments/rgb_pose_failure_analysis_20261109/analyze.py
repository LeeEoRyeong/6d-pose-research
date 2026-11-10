import sys,os,csv,json,hashlib,itertools,collections,math
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;EXP=P.parent;ADV=EXP/'official_rgb_pose_advanced_20261108';Y=EXP/'yolo_hcce_rgb_pipeline';OLD=EXP/'itodd_symmetry_eval_20261108';sys.path.insert(0,str(OLD));import evaluate as e,symmetry

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def main():
 assert not e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied';torch.set_num_threads(2);cv2.setNumThreads(1)
 files=[Y/'pose_eval.csv',Y/'detector_eval.csv',Y/'failure_analysis.csv',ADV/'per_frame.csv',ADV/'bb100_pose_diagnostic.csv',ADV/'bb100_audit.csv',Y/'runs/yolo11n_pbr_attempt3/weights/best.pt',e.AUD/'itodd_obj04_1_0000step50000.pt'];before={str(p):sha(p) for p in files}
 v,faces,syms,info=symmetry.load();data,_=e.get_data(v); samples=data['real'];dets={int(r['frame']):r for r in read(Y/'detector_eval.csv')};oldy={int(r['frame']):r for r in read(Y/'pose_eval.csv')};oldoracle={int(r['frame']):r for r in read(ADV/'per_frame.csv') if r['model']=='official'};assert len(samples)==len(dets)==len(oldy)==len(oldoracle)==27
 mod=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+a] for a in 'xyz']);sz=np.array([li['size_'+a] for a in 'xyz']);mn=torch.tensor(lo,dtype=torch.float32,device='cuda');size=torch.tensor(sz,dtype=torch.float32,device='cuda');net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=mn,size_xyz=size)
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval();del ck
 crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});pad=e.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});th=.1*info['diameter'];rows=[];stability=[];per_frame=[];overlay=P/'overlays'/'roi';overlay.mkdir(parents=True)
 for s in samples:
  dr=dets[s['id']];old=oldy[s['id']];ora=oldoracle[s['id']];bb=np.array(json.loads(dr['detector_bbox_xyxy']),float);wh=bb[2:]-bb[:2];center=(bb[:2]+bb[2:])/2;cad=np.array(json.loads(dr['cad_amodal_bbox_xyxy']),float);cx=(cad[:2]+cad[2:])/2;dw=cad[2:]-cad[:2];category=('both_success' if old['symmetry_ADD_pass']=='True' else 'oracle_only') if ora['symmetry_ADD_pass']=='True' else ('yolo_only' if old['symmetry_ADD_pass']=='True' else 'both_fail')
  per_frame.append({'frame':s['id'],'category':category,'oracle_pnp':ora['pnp_valid'],'yolo_pnp':old['pnp_valid'],'oracle_symmetry_ADD':ora['symmetry_ADD'],'yolo_symmetry_ADD':old['symmetry_ADD'],'bbox_iou':dr['iou_to_cad_amodal'],'bbox_center_dx_px':float(center[0]-cx[0]),'bbox_center_dy_px':float(center[1]-cx[1]),'bbox_center_distance_px':dr['center_error_px'],'bbox_width_ratio':float(wh[0]/dw[0]),'bbox_height_ratio':float(wh[1]/dw[1]),'confidence':dr['confidence'],'previous_GT_uncertain':s['id'] in (66,68,70,72,73)})
  for scale in [1.,1.05,1.1,1.15,1.2]:
   w=wh*scale;box=pad(np.r_[center-w/2,w],1.5);image=crop(s['image'],box,256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255;x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):o=net.inference_batch(x[None].cuda(),torch.tensor(box[None],dtype=torch.float32,device='cuda'))
   mask=o['pred_mask'][0].cpu().numpy()>0;front=o['pred_front_code_obj'][0].float().cpu().numpy();uv=o['coord_2d_image'][0].float().cpu().numpy();r={'frame':s['id'],'bbox_scale':scale,'detector_confidence':float(dr['confidence']),'pnp_valid':False,'failure':'','mask_pixels':int(mask.sum()),'ADD':'','ADD_S':'','symmetry_ADD':'','symmetry_index':'','ADD_pass':False,'ADD_S_pass':False,'symmetry_ADD_pass':False,'rotation_sym_deg':'','translation_sym_mm':'','bbox_xywh':json.dumps(box.tolist()),'xyz_stability_median_mm':'','xyz_stability_overlap_pixels':''};pose=None
   try:
    if mask.sum()<=4:raise ValueError('mask <=4 pixels')
    cv2.setRNGSeed(42+s['id']);z=solve((mask,front,uv,s['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
    if not z['success']:raise ValueError('PnP failed')
    R=np.asarray(z['rot'],float);t=np.asarray(z['tvecs'],float).reshape(3)
    if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:raise ValueError('invalid pose')
    a=symmetry.add(v,R,t,s['R'],s['t']);b=symmetry.add_s(v,R,t,s['R'],s['t']);c,idx=symmetry.symmetry_add(v,R,t,s['R'],s['t'],syms);rg,tg=symmetry.compose(s['R'],s['t'],syms[idx]);r.update(pnp_valid=True,ADD=a,ADD_S=b,symmetry_ADD=c,symmetry_index=idx,ADD_pass=a<th,ADD_S_pass=b<th,symmetry_ADD_pass=c<th,rotation_sym_deg=angle(R,rg),translation_sym_mm=float(np.linalg.norm(t-tg)));pose=(R,t)
   except Exception as ex:r['failure']=str(ex)
   # compare predictions at corresponding image coordinates, only shared predicted-mask support; not accuracy or GT
   if scale==1.:base=(box.copy(),mask.copy(),front.copy());assert r['pnp_valid']==(old['pnp_valid']=='True');assert r['symmetry_ADD_pass']==(old['symmetry_ADD_pass']=='True');assert not r['pnp_valid'] or abs(r['symmetry_ADD']-float(old['symmetry_ADD']))<1e-3
   else:
    box0,m0,f0=base;coords=np.indices(m0.shape,dtype=np.float32); xpix=box0[0]+coords[1]*box0[2]/m0.shape[1];ypix=box0[1]+coords[0]*box0[3]/m0.shape[0];mapx=((xpix-box[0])*mask.shape[1]/box[2]).astype(np.float32);mapy=((ypix-box[1])*mask.shape[0]/box[3]).astype(np.float32);m1=cv2.remap(mask.astype(np.uint8),mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;f1=cv2.remap(front,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);valid=m0&m1;cnt=int(valid.sum());r['xyz_stability_overlap_pixels']=cnt;r['xyz_stability_median_mm']=float(np.median(np.linalg.norm(f0[valid]-f1[valid],axis=1))) if cnt else ''
   rows.append(r);view=s['image'].copy();polys=[np.clip(e.project(v,s['R'],s['t'],s['K'])[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in faces];cv2.polylines(view,polys,True,(0,255,0),1,cv2.LINE_AA)
   if pose:
    pp=e.project(v,pose[0],pose[1],s['K']);cv2.polylines(view,[np.clip(pp[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in faces],True,(255,0,255),1,cv2.LINE_AA)
   cv2.rectangle(view,tuple(box[:2].astype(int)),tuple((box[:2]+box[2:]).astype(int)),(0,255,255),2);cv2.putText(view,f'frame={s["id"]} bbox scale={scale:.2f} sym pass={r["symmetry_ADD_pass"]}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA);cv2.imwrite(str(overlay/f'{s["id"]:06d}_scale_{scale:.2f}.jpg'),view)
  print('ROI FRAME',s['id'],flush=True)
 write(P/'frame_comparison.csv',per_frame);write(P/'roi_comparison.csv',rows)
 roi=[]
 for scale in [1.,1.05,1.1,1.15,1.2]:
  z=[r for r in rows if r['bbox_scale']==scale];vals=[r['xyz_stability_median_mm'] for r in z if r['xyz_stability_median_mm']!=''];roi.append({'bbox_scale':scale,'n':len(z),'pnp':sum(r['pnp_valid'] for r in z),'ADD':sum(r['ADD_pass'] for r in z),'ADD_S':sum(r['ADD_S_pass'] for r in z),'symmetry_ADD':sum(r['symmetry_ADD_pass'] for r in z),'xyz_stability_frame_median_mm':float(np.median(vals)) if vals else None})
 json.dump(roi,(P/'roi_summary.json').open('x'),indent=2);print('ROI',roi,flush=True)
 # BB100 independent audit of stored predictions. Crops from existing GT/pred overlay retained for all 45 failures.
 bbr={int(r['frame']):r for r in read(ADV/'bb100_pose_diagnostic.csv')};bba={int(r['frame']):r for r in read(ADV/'bb100_audit.csv')};gt=e.json_read(EXP/'real_bb100_20261108/BB_obj04/scene_gt_obj04.json');posegroups={};bbrows=[];bd=P/'overlays'/'bb100_failures';bd.mkdir(parents=True)
 for i in sorted(bbr):
  r=bbr[i];a=bba[i];g=gt[str(i)][0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c']);key=tuple(np.round(np.r_[R.flatten(),t],8));posegroups.setdefault(key,[]).append(i);pnp=r['pnp_valid']=='True';rot=trans=''
  if pnp:
   rp=np.array(json.loads(r['pred_R']));tp=np.array(json.loads(r['pred_t']));rg,tg=symmetry.compose(R,t,syms[int(r['symmetry_index'])]);rot=angle(rp,rg);trans=float(np.linalg.norm(tp-tg))
  fail=r['symmetry_ADD_pass']!='True';row={'frame':i,'state':a['state'],'time_group':a['group_id'],'symmetry_ADD':r['symmetry_ADD'],'ADD_S':r['ADD_S'],'symmetry_ADD_pass':not fail,'pnp_valid':pnp,'mask_pixels':r['mask_pixels'],'symmetry_index':r['symmetry_index'],'rotation_sym_deg':rot,'translation_sym_mm':trans,'pose_group':'','occlusion_estimate':'unknown_not_measured','calibration':'unknown_BB_D_unverified','GT_pose':'unknown_not_independently_validated','rotation_cause':'unknown','roi_cause':'unknown_oracle_roi','XYZ_cause':'unknown','PnP_cause':'unknown_valid_return_does_not_establish_stability' if pnp else 'pnp_failure','overall_cause':'unknown','visual_review':'overlay_and_XYZ_panel_available_no_ground_truth_visible_mask'};bbrows.append(row)
  if fail:
   img=cv2.imread(str(ADV/'bb100_inference/official/bb100'/f'{i:06d}_pose_overlay.png'));panel=cv2.imread(str(ADV/'bb100_inference/official/bb100'/f'{i:06d}_prediction_panel.png'));assert img is not None and panel is not None
   bbox=np.array(json.loads(a['projected_cad_amodal_bbox_xywh']));cx=bbox[0]+bbox[2]/2;cy=bbox[1]+bbox[3]/2;side=max(bbox[2],bbox[3])*1.9;x1=max(0,int(cx-side/2));x2=min(img.shape[1],int(cx+side/2));y1=max(0,int(cy-side/2));y2=min(img.shape[0],int(cy+side/2));cut=img[y1:y2,x1:x2];cut=cv2.resize(cut,(512,512));pan=cv2.resize(panel,(1024,285));canvas=np.zeros((797,1024,3),np.uint8);canvas[:512,256:768]=cut;canvas[512:]=pan;cv2.putText(canvas,f'BB {i:03d}: GT green / prediction magenta; bottom mask & XYZ predictions',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);cv2.imwrite(str(bd/f'{i:06d}.jpg'),canvas)
 for key,ids in posegroups.items():
  for r in bbrows:
   if r['frame'] in ids:r['pose_group']=min(ids)
 write(P/'bb100_failure_analysis.csv',bbrows);fails=[r for r in bbrows if not r['symmetry_ADD_pass']];assert len(fails)==45
 # Contact sheets preserve individual crops for visual inspection; whole group analysis does not claim per-pixel accuracy.
 for offset in range(0,len(fails),9):
  page=Image.new('RGB',(1200,1320),'white');draw=ImageDraw.Draw(page)
  for j,r in enumerate(fails[offset:offset+9]):
   img=Image.open(bd/f'{r["frame"]:06d}.jpg').crop((256,0,768,512)).resize((390,390));x=(j%3)*400;y=(j//3)*440;page.paste(img,(x,y+25));draw.text((x+5,y+4),f"{r['frame']:03d} symADD={float(r['symmetry_ADD']):.1f} group={r['pose_group']}",fill='black')
  page.save(bd/f'contact_{offset//9+1}.jpg',quality=90)
 stats={'categories':dict(collections.Counter(r['category'] for r in per_frame)),'bb100_failed':len(fails),'bb_failure_pnp_valid':sum(r['pnp_valid'] for r in fails),'bb_failure_by_time_group':dict(collections.Counter(r['time_group'] for r in fails)),'bb_all_by_time_group':dict(collections.Counter(r['time_group'] for r in bbrows)),'bb_failure_by_state':dict(collections.Counter(r['state'] for r in fails)),'bb_all_by_state':dict(collections.Counter(r['state'] for r in bbrows)),'bb_failure_symmetry_index':dict(collections.Counter(r['symmetry_index'] for r in fails)),'bb_pose_groups':len(posegroups),'bb_failure_pose_groups':len(set(r['pose_group'] for r in fails)),'bb_same_pose_mixed_success_groups':sum(len({next(r['symmetry_ADD_pass'] for r in bbrows if r['frame']==i) for i in ids})>1 for ids in posegroups.values()),'bb_rotation_failed_median_deg':float(np.median([r['rotation_sym_deg'] for r in fails if r['rotation_sym_deg']!=''])),'bb_translation_failed_median_mm':float(np.median([r['translation_sym_mm'] for r in fails if r['translation_sym_mm']!=''])),'roi_summaries':roi,'hashes_unchanged':before=={p:sha(Path(p)) for p in before},'input_sha256':before,'input_hash_coverage':'key inputs, not all original files'}
 # Correlations for matched successful PnP, diagnostics only.
 vals=[(float(r['bbox_iou']),float(r['yolo_symmetry_ADD'])) for r in per_frame if r['bbox_iou'] and r['yolo_symmetry_ADD']];stats['bbox_iou_vs_symmetry_ADD_pearson']=float(np.corrcoef(np.array(vals).T)[0,1]);stats['correlation_n']=len(vals)
 json.dump(stats,(P/'failure_statistics.json').open('x'),indent=2,allow_nan=False);print('DONE',json.dumps(stats),flush=True)
if __name__=='__main__':main()
