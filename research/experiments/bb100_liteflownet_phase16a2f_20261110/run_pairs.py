from pathlib import Path
import sys,csv,json,hashlib,time,subprocess,math
import numpy as np,cv2,torch,cupy
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;PH=P/'bb100_visible_mask_phase16a1_20261110';PH15=P/'bb100_placement_targets_phase15_20261109';root=Path('/home/oylee/projects/qwen_smt/data/BB_rgb_obj04/run_20260813_164421');sys.path.insert(0,str(O/'pytorch-liteflownet'));sys.argv=['run.py'];torch.hub.load_state_dict_from_url=lambda **kw:torch.load(O/'network-default.pytorch',map_location='cpu',weights_only=True)
import run
cv2.setNumThreads(1)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def estimate(net,a,b):
 h,w=a.shape[:2];H=((h+31)//32)*32;W=((w+31)//32)*32
 a=torch.from_numpy(a.transpose(2,0,1).copy()).float().cuda().unsqueeze(0)/255;b=torch.from_numpy(b.transpose(2,0,1).copy()).float().cuda().unsqueeze(0)/255
 a=torch.nn.functional.interpolate(a,size=(H,W),mode='bilinear',align_corners=False);b=torch.nn.functional.interpolate(b,size=(H,W),mode='bilinear',align_corners=False)
 with torch.inference_mode(),cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
  f=net(a,b);f=torch.nn.functional.interpolate(f,size=(h,w),mode='bilinear',align_corners=False);f[:,0]*=w/W;f[:,1]*=h/H
 torch.cuda.synchronize();return f[0].permute(1,2,0).cpu().numpy()
def flow_hsv(flow,limit):
 mag,ang=cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True);hsv=np.zeros((*mag.shape,3),np.uint8);hsv[:,:,0]=(ang/2).astype('uint8');hsv[:,:,1]=255;hsv[:,:,2]=(np.clip(mag/limit,0,1)*255).astype('uint8');return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
def main():
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip();assert not compute,'GPU compute already in use'
 rows=read(P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv');frames={int(x['frame']):x for x in rows};pairs=[(43,44,'A_stationary_gripper_motion_hypothesis'),(75,77,'B_joint_motion_hypothesis'),(99,101,'C_reposition_hypothesis'),(221,222,'B_joint_motion_hypothesis'),(235,237,'D_occlusion_registration_failure'),(340,341,'D_occlusion_registration_failure')]
 prev=read(P/'bb100_motion_assist_phase15b_20261110/boundary_analysis.csv');old={(int(x['before_frame']),int(x['after_frame'])):x for x in prev};audit={int(x['frame']):x for x in read(P/'bb100_gt_camera_verification_20261109/bb100_frame_audit.csv')};mapping=read(root/'frame_opc_mapping.csv');available={int(Path(x['frame_name']).stem) for x in mapping};rawids={int(f.stem) for f in (root/'rgb').rglob('*.jpg') if f.stem.isdigit()}
 for d in ['flow','masks','package','package/panels','package/masks']:(O/d).mkdir()
 net=run.Network().cuda().eval();torch.cuda.reset_peak_memory_stats();inputs={str(O/'network-default.pytorch'):sha(O/'network-default.pytorch')};results=[];data=[]
 # Identical JPEG test is a diagnostic, not a universal optical flow accuracy test.
 im=cv2.imread(frames[32]['image_path']);zero=estimate(net,im,im);identical={'median_magnitude_px':float(np.median(np.linalg.norm(zero,axis=2))),'p95_magnitude_px':float(np.quantile(np.linalg.norm(zero,axis=2),.95)),'finite':bool(np.isfinite(zero).all())};assert identical['finite'];print('identical smoke',identical,flush=True)
 for a,b,category in pairs:
  before=Path(frames[a]['image_path']);after=Path(frames[b]['image_path']);sam_path=PH/'sam'/f'{a:06d}_selected_NOT_GT.png';cad_path=PH15/'amodal_masks'/f"pose_{frames[a]['pose_group']}.png";cad2_path=PH15/'amodal_masks'/f"pose_{frames[b]['pose_group']}.png"
  for f in (before,after,sam_path,cad_path,cad2_path):inputs[str(f)]=sha(f)
  im1=cv2.imread(str(before));im2=cv2.imread(str(after));cad=cv2.imread(str(cad_path),0)>0;cad2=cv2.imread(str(cad2_path),0)>0;sam=cv2.imread(str(sam_path),0)>0;assert im1.shape==im2.shape==(720,1280,3)
  t=time.monotonic();flow=estimate(net,im1,im2);back=estimate(net,im2,im1);assert np.isfinite(flow).all() and np.isfinite(back).all();H,W=cad.shape;yy,xx=np.mgrid[:H,:W].astype('float32');mapx=xx+flow[:,:,0];mapy=yy+flow[:,:,1];inside=(mapx>=0)&(mapx<=W-1)&(mapy>=0)&(mapy<=H-1);wb=cv2.remap(back,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);fb=np.linalg.norm(flow+wb,axis=2);mag=np.linalg.norm(flow,axis=2);bmag=np.linalg.norm(wb,axis=2);consistent=inside&(fb<=1.0+.05*(mag+bmag));bg=~(cv2.dilate((cad|cad2).astype('uint8'),np.ones((151,151),np.uint8))>0);bgvalid=bg&consistent
  med=np.median(flow[bgvalid],axis=0) if bgvalid.sum()>1000 else np.array([np.nan,np.nan]);residual=np.linalg.norm(flow-med,axis=2);thr=float(np.quantile(residual[bgvalid],.99)) if bgvalid.sum()>1000 else None;phase=old[(a,b)];bg_ok=phase['bg_quality_heuristic']=='True';move=(consistent&(residual>thr)) if (bg_ok and thr is not None) else np.zeros((H,W),bool);unknown=~consistent
  base=sam&cad;c=sam&~move;d=base&~move;outid=f'{a:06d}_{b:06d}'
  np.savez_compressed(O/'flow'/f'{outid}_flow_NOT_GT.npz',forward=flow,backward=back,fb_error=fb,consistent=consistent.astype('uint8'))
  masks={'sam_before':sam,'cad_sam_before':base,'flow_move_not_gripper':move,'sam_minus_motion_exploratory':c,'cad_sam_minus_motion_exploratory':d,'fb_inconsistent':unknown}
  for name,m in masks.items():
   path=O/'masks'/f'{outid}_{name}_NOT_GT.png';assert cv2.imwrite(str(path),m.astype('uint8')*255);cv2.imwrite(str(O/'package/masks'/path.name),m.astype('uint8')*255)
  # RGB: original, colored dense vector field, and raw magnitude bar share per-pair range.
  maglim=max(float(np.quantile(mag,.95)),.01);color=flow_hsv(flow,maglim);flowov=cv2.addWeighted(im1,.5,color,.5,0)
  def overlay(mask,clr):
   v=im1.copy();v[mask]=(v[mask]*.45+np.array(clr)*.55).astype('uint8');return v
  ys,xs=np.nonzero(cad|cad2);pad=65;x0=max(0,int(xs.min())-pad);x1=min(W,int(xs.max())+pad+1);y0=max(0,int(ys.min())-pad);y1=min(H,int(ys.max())+pad+1)
  def tile(v,title):
   v=v[y0:y1,x0:x1];out=cv2.resize(v,(420,300));cv2.rectangle(out,(0,0),(420,24),(0,0,0),-1);cv2.putText(out,title,(5,17),0,.46,(255,255,255),1,cv2.LINE_AA);return out
  diff=np.zeros_like(im1);diff[move]=[0,0,255];diff[unknown]=[0,200,255]
  frames8=[im1,im2,color,flowov,overlay(sam,[255,0,200]),overlay(base,[0,200,0]),overlay(c,[255,0,200]),overlay(d,[0,200,0]),diff,overlay(unknown,[0,200,255])]
  titles=['BEFORE RGB','AFTER RGB','FLOW DIRECTION / p95 SCALE','FLOW OVERLAY','SAM BEFORE NOT GT','CAD & SAM BEFORE NOT GT','SAM - MOVING NOT GT','CAD SAM - MOVING NOT GT','MOVING red / INCONSISTENT yellow','FB INCONSISTENT NOT GT']
  tiles=[tile(x,y) for x,y in zip(frames8,titles)];panel=np.vstack([np.hstack(tiles[:5]),np.hstack(tiles[5:])]);p=O/'package/panels'/f'{outid}.jpg';cv2.imwrite(str(p),panel,[cv2.IMWRITE_JPEG_QUALITY,90])
  objvals=mag[cad]; bgvals=mag[bg];seq=[i for i in range(a+1,b) if i in available];raw=[i for i in seq if i in rawids]
  rec={'before_frame':a,'after_frame':b,'hypothesis':category,'not_motion_gt':True,'delta_seconds':float(audit[b]['frame_mono'])-float(audit[a]['frame_mono']),'intermediate_mapped_frame_ids':','.join(map(str,seq)),'intermediate_rgb_available_ids':','.join(map(str,raw)),'phase15b_boundary':phase['boundary_id'],'phase15b_auto_label':phase['auto_label'],'phase15b_bg_quality_pass':bg_ok,'phase15b_sift_relative_px':phase['sift_relative_displacement_px'],'phase15b_orb_relative_px':phase['orb_relative_displacement_px'],'cad_region_median_flow_px':float(np.median(objvals)),'background_region_median_flow_px':float(np.median(bgvals)),'background_median_u_px':float(med[0]) if np.isfinite(med[0]) else None,'background_median_v_px':float(med[1]) if np.isfinite(med[1]) else None,'background_residual_p99_px_threshold_not_gt':thr,'fb_consistent_fraction':float(consistent.mean()),'fb_consistent_in_cad_fraction':float(consistent[cad].mean()),'fb_error_cad_median_px':float(np.median(fb[cad])),'bg_consistent_pixels':int(bgvalid.sum()),'flow_moving_heuristic_pixels':int(move.sum()),'sam_pixels':int(sam.sum()),'cad_sam_pixels':int(base.sum()),'sam_minus_motion_removed_pixels':int((sam&move).sum()),'cad_sam_minus_motion_removed_pixels':int((base&move).sum()),'flow_suppression_enabled':bool(bg_ok and thr is not None),'elapsed_sec_both':round(time.monotonic()-t,3),'flow_color_p95_px':maglim,'status':'EXPLORATORY_NO_GT'};results.append(rec);data.append({**rec,'panel':f'panels/{outid}.jpg','mask_prefix':outid});print(outid,'F/B pass',round(rec['fb_consistent_in_cad_fraction'],3),'motion suppression',rec['flow_suppression_enabled'],'removed',rec['sam_minus_motion_removed_pixels'],flush=True)
 for n in inputs:assert sha(Path(n))==inputs[n]
 with (O/'pairs.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
 summary={'model':'sniklaus/pytorch-liteflownet default','weights_sha256':inputs[str(O/'network-default.pytorch')],'source_commit':subprocess.check_output(['git','-C',str(O/'pytorch-liteflownet'),'rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cupy':cupy.__version__,'gpu':torch.cuda.get_device_name(0),'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),'input_hashes_unchanged':True,'input_sha256':inputs,'pairs':6,'forward_backward_inferences':12,'sanity_identical':identical,'uncertainty':'forward/backward mismatch not a calibrated confidence; moving is NOT gripper','labels_are_not_gt':True,'resize':'official estimate interpolation to multiple 32, BGR, resize flow and component scale; official example-only shape assertion skipped in separate wrapper'}
 (O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');(O/'data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
