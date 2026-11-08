"""Synchronized GPU warm-up and pose-stage latency; detector timing separately."""
import os,sys,json,time,itertools,torch,numpy as np,cv2
from pathlib import Path
HERE=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(HERE/'.ultralytics');os.environ['MPLCONFIGDIR']=str(HERE/'.mpl')
from ultralytics import YOLO
OLD=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_symmetry_eval_20261108');sys.path.insert(0,str(OLD));import evaluate as e,symmetry

def now():torch.cuda.synchronize();return time.perf_counter()
def stat(vals):return {'mean_ms':float(np.mean(vals)),'median_ms':float(np.median(vals)),'p95_ms':float(np.quantile(vals,.95)),'n':len(vals),'samples_ms':[round(x,5) for x in vals]}
def main():
 assert json.load((HERE/'reproduction.json').open())['status']=='PASS';torch.set_num_threads(2);cv2.setNumThreads(1)
 apps=e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not apps,apps
 v,_,_,_=symmetry.load();data,_=e.get_data(v);s=next(x for x in data['real'] if x['id']==26);module=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+a] for a in 'xyz']);sz=np.array([li['size_'+a] for a in 'xyz']);mn=torch.tensor(lo,dtype=torch.float32,device='cuda');size=torch.tensor(sz,dtype=torch.float32,device='cuda')
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 net=module.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=mn,size_xyz=size);net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval();del ck
 det=YOLO(str(e.EXP/'realistic_final100/yolo11/train_obj_s/smoke_train/weights/best.pt'));crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None];bbox=torch.tensor(np.array(s['box'])[None],dtype=torch.float32,device='cuda')
 timings={'rgb_preprocess_crop_normalize_h2d':[],'hcce_network_forward':[],'hcce_decode_and_gpu_to_cpu':[],'pnp_cpu':[],'pose_total':[],'detector_predict':[]};pnp_ok=0;detected=0
 for i in range(35):
  begin=now();rgb=crop(s['image'],s['box'],256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255;x=((x-mean)/std)[None].cuda();t1=now()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net.net(x)
  t2=now()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
   mask=torch.sigmoid(pm)[:,0]>.5;bits=(torch.sigmoid(pc)>.5).to(pc.dtype).permute(0,2,3,1);front=net.hcce_decode(bits[...,:24])/255;back=net.hcce_decode(bits[...,24:])/255
   yy=torch.arange(front.shape[1],device='cuda').float()/front.shape[1];xx=torch.arange(front.shape[2],device='cuda').float()/front.shape[2];X,Y=torch.meshgrid(xx,yy,indexing='xy');coords=torch.stack([X,Y],dim=-1)[None];coords=coords*bbox[:,None,None,2:]+bbox[:,None,None,:2]
   xyz=front*size+mn;mask_np=mask[0].cpu().numpy();xyz_np=xyz[0].float().cpu().numpy();uv=coords[0].float().cpu().numpy()
  t3=now();cv2.setRNGSeed(42+s['id'])
  try:result=solve((mask_np,xyz_np,uv,s['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150);ok=bool(result['success'])
  except Exception:ok=False
  end=now()
  # Verify split forward/decode against the original inference method before timing results are accepted.
  if i==0:
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):reference=net.inference_batch(x,bbox)
   assert np.array_equal(mask_np,reference['pred_mask'][0].cpu().numpy()>0)
   assert np.array_equal(xyz_np,reference['pred_front_code_obj'][0].float().cpu().numpy())
   assert np.array_equal(uv,reference['coord_2d_image'][0].float().cpu().numpy())
  if i>=5:
   for k,val in [('rgb_preprocess_crop_normalize_h2d',t1-begin),('hcce_network_forward',t2-t1),('hcce_decode_and_gpu_to_cpu',t3-t2),('pnp_cpu',end-t3),('pose_total',end-begin)]:timings[k].append(val*1000)
   pnp_ok+=int(ok)
 for i in range(35):
  start=now();p=det.predict(source=s['image'],imgsz=640,conf=.25,iou=.45,max_det=10,device=0,verbose=False,save=False)[0];end=now()
  if i>=5:timings['detector_predict'].append((end-start)*1000);detected+=int(len(p.boxes)>0)
 dataout={'status':'PASS_SYNCHRONIZED_SINGLE_FRAME','frame':26,'warmup_iterations':5,'measured_iterations':30,'gpu':torch.cuda.get_device_name(0),'timings':{k:stat(v) for k,v in timings.items()},'pose_pnp_successes':pnp_ok,'detector_detections':detected,'actual_detector_to_pose_end_to_end':'not measured: detector produced no detections','preprocess_scope':'already-undistorted 1280x720 RGB, oracle ROI crop+normalize+CPU-to-GPU; excludes disk IO and undistortion','detector_scope':'YOLO.predict Python overhead included, input already-undistorted RGB, conf .25','decode_scope':'HCCE front/back + coordinate mapping + GPU-to-CPU; matched original inference_batch','warning':'repeated single frame; batch=1; no target FPS defined; pose total excludes detector'}
 with (HERE/'latency.json').open('x') as f:json.dump(dataout,f,indent=2);f.write('\n')
 print(json.dumps({k:{kk:round(vv,3) for kk,vv in v.items() if kk.endswith('_ms') and isinstance(vv,(int,float))} for k,v in dataout['timings'].items()},indent=2));print('pnp',pnp_ok,'detections',detected,flush=True)
if __name__=='__main__':main()
