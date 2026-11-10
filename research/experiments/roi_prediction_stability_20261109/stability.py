"""GT-free YOLO ROI stability. Inputs: raw RGB, K/D, saved predicted boxes, CAD, model."""
import os,sys,json,csv,itertools,hashlib,time,argparse,collections
from pathlib import Path
import cv2,numpy as np,torch
P=Path(__file__).resolve().parent;E=P.parent;Y=E/'yolo_hcce_rgb_pipeline';OLD=E/'itodd_symmetry_eval_20261108';DEV=Path('/home/oylee/projects/HCCEPose_obj04_transfer');sys.path.insert(0,str(OLD));import evaluate as e,symmetry

def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def sync():torch.cuda.synchronize();return time.perf_counter()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=P/('smoke' if a.smoke else 'full');out.mkdir(exist_ok=False)
 if not a.smoke:assert json.loads((P/'smoke/smoke_check.json').read_text())['status']=='PASS'
 assert not e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied';torch.set_num_threads(2);cv2.setNumThreads(1)
 predicted=read(Y/'detector_eval.csv');ids=sorted(int(r['frame']) for r in predicted);assert len(ids)==27;det={int(r['frame']):r for r in predicted};ids=[26,68,94] if a.smoke else ids
 ref={int(x['frame']):x for x in read(Y/'pose_eval.csv')};cal=json.loads((DEV/'camera_params.json').read_text());K=np.array(cal['camera_matrix'],float);D=np.array(cal['distortion_coefficients'],float);assert K.shape==(3,3) and np.isfinite(K).all() and np.isfinite(D).all();maps=cv2.initUndistortRectifyMap(K,D,None,K,(1280,720),cv2.CV_32FC1)
 v,faces,syms,info=symmetry.load();assert symmetry.validate()['status']=='PASS';R0=np.eye(3);t0=np.array([0.,0.,500.]);R1,t1=symmetry.compose(R0,t0,syms[1]);assert min(angle(R1,R0@S[:3,:3]) for S in syms)<1e-5 and min(np.linalg.norm(t1-(R0@S[:3,3]+t0)) for S in syms)<1e-7
 mod=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+x] for x in 'xyz']);sz=np.array([li['size_'+x] for x in 'xyz']);mn=torch.tensor(lo,dtype=torch.float32,device='cuda');size=torch.tensor(sz,dtype=torch.float32,device='cuda');net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=mn,size_xyz=size);cp=e.AUD/'itodd_obj04_1_0000step50000.pt'
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(cp,map_location='cpu',weights_only=True)
 net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval();del ck;crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});pad=e.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools})
 sources=[cp,Y/'runs/yolo11n_pbr_attempt3/weights/best.pt',Y/'detector_eval.csv',Y/'pose_eval.csv',DEV/'camera_params.json',e.AUD/'official_obj_000004.ply',e.AUD/'official_models_info.json']+[DEV/'image'/f'{i:06d}.jpg' for i in ids];before={str(p):sha(p) for p in sources};rows=[];baseline_repeat=[];vis=out/'overlays';vis.mkdir();torch.cuda.reset_peak_memory_stats()
 # warm up using actual image, no reference pose or GT. No measurement for warmup.
 im=cv2.remap(cv2.imread(str(DEV/'image'/f'{ids[0]:06d}.jpg')),maps[0],maps[1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);d=det[ids[0]];bb=np.array(json.loads(d['detector_bbox_xyxy']),float);basebox=pad(np.r_[bb[:2],bb[2:]-bb[:2]],1.5)
 def evaluate(im,box,i):
  begin=sync();imcrop=crop(im,box,256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(imcrop.transpose(2,0,1))).float()/255;x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None];x=x[None].cuda();bx=torch.tensor(box[None],dtype=torch.float32,device='cuda');t1=sync()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):o=net.inference_batch(x,bx)
  m=o['pred_mask'][0].cpu().numpy()>0;xyz=o['pred_front_code_obj'][0].float().cpu().numpy();t2=sync();r={'ok':False,'error':'','R':None,'t':None};uv=o['coord_2d_image'][0].float().cpu().numpy()
  try:
   if m.sum()<=4:raise ValueError('mask <= 4')
   cv2.setRNGSeed(42+i);z=solve((m,xyz,uv,K),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
   if not z['success']:raise ValueError('PnP returned failure')
   R=np.asarray(z['rot'],float);t=np.asarray(z['tvecs'],float).reshape(3)
   if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:raise ValueError('invalid pose')
   r.update(ok=True,R=R,t=t)
  except Exception as ex:r['error']=str(ex)
  end=sync();r.update(mask=m,xyz=xyz,box=box.copy(),time_crop_ms=1000*(t1-begin),time_network_decode_ms=1000*(t2-t1),time_pnp_ms=1000*(end-t2),time_total_ms=1000*(end-begin));return r
 for _ in range(3):evaluate(im,basebox,ids[0])
 cond=[('baseline',1.,0.,0.)]+[(f'scale_{s:.2f}',s,0.,0.) for s in [.85,1.2,1.4]]+[(f'{axis}_{x:+.2f}',1.,x if axis=='x' else 0.,x if axis=='y' else 0.) for axis in ['x','y'] for x in [-.1,-.05,.05,.1]]
 for i in ids:
  im=cv2.remap(cv2.imread(str(DEV/'image'/f'{i:06d}.jpg')),maps[0],maps[1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);reference=cv2.imread(str(e.BASE/'agent_run_v1/stage1/controlled_rgb'/f'{i:06d}.png'));assert reference is not None and np.array_equal(im,reference),'undistortion mismatch';dr=det[i];assert dr['detected']=='True';xy=np.array(json.loads(dr['detector_bbox_xyxy']),float);box0=pad(np.r_[xy[:2],xy[2:]-xy[:2]],1.5);center=box0[:2]+box0[2:]/2;base=None
  for label,scale,dx,dy in cond:
   wh=box0[2:]*scale;c=center+np.array([dx,dy])*box0[2:];box=np.r_[c-wh/2,wh];z=evaluate(im,box,i)
   if label=='baseline':
    base=z;assert z['ok']==(ref[i]['pnp_valid']=='True')
    if z['ok']:assert np.allclose(z['R'],json.loads(ref[i]['pred_R']),atol=1e-5) and np.allclose(z['t'],json.loads(ref[i]['pred_t']),atol=1e-3),'baseline predicted pose mismatch'
    if a.smoke and i==26:
     for rep in range(3):
      zz=evaluate(im,box,i);baseline_repeat.append({'frame':i,'repeat':rep,'same_pnp_valid':zz['ok']==z['ok'],'rotation_diff_deg':angle(zz['R'],z['R']) if zz['ok'] and z['ok'] else None,'translation_diff':float(np.linalg.norm(zz['t']-z['t'])) if zz['ok'] and z['ok'] else None,'same_mask':bool(np.array_equal(z['mask'],zz['mask'])),'same_xyz':bool(np.array_equal(z['xyz'],zz['xyz']))})
   rr={'frame':i,'condition':label,'scale':scale,'dx_fraction_of_padded_roi':dx,'dy_fraction_of_padded_roi':dy,'pnp_valid':z['ok'],'baseline_pnp_valid':base['ok'],'failure':z['error'],'mask_pixels':int(z['mask'].sum()),'bbox_xywh':json.dumps(box.tolist()),'rotation_raw_deg':'','rotation_sym_deg':'','translation_raw_cad_units':'','translation_sym_aligned_cad_units':'','rotation_selected_symmetry_index':'','xyz_median_cad_units':'','xyz_p95_cad_units':'','xyz_common_pixels':0,'xyz_common_fraction_baseline_mask':'','time_crop_ms':z['time_crop_ms'],'time_network_decode_ms':z['time_network_decode_ms'],'time_pnp_ms':z['time_pnp_ms'],'time_total_ms':z['time_total_ms']}
   if z['ok'] and base['ok']:
    ang=[angle(z['R'],base['R']@S[:3,:3]) for S in syms];idx=int(np.argmin(ang));rb,tb=symmetry.compose(base['R'],base['t'],syms[idx]);rr.update(rotation_raw_deg=angle(z['R'],base['R']),rotation_sym_deg=ang[idx],translation_raw_cad_units=float(np.linalg.norm(z['t']-base['t'])),translation_sym_aligned_cad_units=float(np.linalg.norm(z['t']-tb)),rotation_selected_symmetry_index=idx)
   grid=np.indices(base['mask'].shape,dtype=np.float32);xp=box0[0]+grid[1]*box0[2]/base['mask'].shape[1];yp=box0[1]+grid[0]*box0[3]/base['mask'].shape[0];mx=((xp-box[0])*z['mask'].shape[1]/box[2]).astype(np.float32);my=((yp-box[1])*z['mask'].shape[0]/box[3]).astype(np.float32);zm=cv2.remap(z['mask'].astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;zx=cv2.remap(z['xyz'],mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);common=base['mask']&zm;cnt=int(common.sum());rr['xyz_common_pixels']=cnt;rr['xyz_common_fraction_baseline_mask']=cnt/max(1,int(base['mask'].sum()))
   if cnt:
    dist=np.linalg.norm(base['xyz'][common]-zx[common],axis=1);rr.update(xyz_median_cad_units=float(np.median(dist)),xyz_p95_cad_units=float(np.quantile(dist,.95)))
   if label=='baseline':assert not cnt or rr['xyz_median_cad_units']==0
   rows.append(rr)
   if i in (26,68,94) and label in ('baseline','scale_0.85','scale_1.40','x_+0.10','y_+0.10'):
    canvas=im.copy();cv2.rectangle(canvas,tuple(box[:2].astype(int)),tuple((box[:2]+box[2:]).astype(int)),(0,255,255),2)
    if z['ok']:
     uvp=e.project(v,z['R'],z['t'],K);cv2.polylines(canvas,[np.clip(uvp[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in faces],True,(255,0,255),1,cv2.LINE_AA)
    cv2.putText(canvas,f'{i} {label} predicted CAD / ROI, no GT',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);cv2.imwrite(str(vis/f'{i:06d}_{label}.jpg'),canvas)
  print('FRAME',i,flush=True)
 write(out/'per_frame.csv',rows);after={p:sha(Path(p)) for p in before};assert before==after;assert len(rows)==len(ids)*12
 if a.smoke:
  assert len(baseline_repeat)==3 and all(x['same_pnp_valid'] and x['same_mask'] and x['same_xyz'] and x['rotation_diff_deg']<1e-4 and x['translation_diff']<1e-6 for x in baseline_repeat);json.dump({'status':'PASS','frames':ids,'conditions':12,'undistortion':'pixel equal to controlled PNG','baseline_matches_previous_prediction':True,'symmetry_synthetic_check':'PASS','repeat':baseline_repeat,'input_hashes_unchanged':True},(out/'smoke_check.json').open('x'),indent=2)
 else:json.dump({'frames':len(ids),'conditions':12,'pnp_seed':'42+frame','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'input_hashes_unchanged':True,'input_sha256':before,'scope':'saved YOLO bbox; no GT pose, no BB, no oracle','timing_scope':'GPU synchronized crop+normalize+H2D, network+decode+D2H, PnP; excludes detector, disk, undistortion, overlay; three warmups','units':'BOP CAD model mm; physical scale of real GT unverified'},(out/'manifest.json').open('x'),indent=2)
 print('PASS',len(rows),flush=True)
if __name__=='__main__':main()
