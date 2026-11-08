"""RGB-only YOLO->official HCCEPose inference. GT never enters prediction."""
import os,sys,json,argparse,itertools,time
from pathlib import Path
P=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(P/'.ultralytics');os.environ['MPLCONFIGDIR']=str(P/'.mpl');sys.path.insert(0,str(P.parent/'itodd_symmetry_eval_20261108'))
import cv2,numpy as np,torch
from ultralytics import YOLO
import evaluate as e,symmetry

def sync():torch.cuda.synchronize();return time.perf_counter()
class Pipeline:
 def __init__(self,weights,conf=.25):
  self.det=YOLO(str(weights));self.conf=conf;self.v,self.faces,self.syms,self.info=symmetry.load();mod=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];self.lo=np.array([li['min_'+a] for a in 'xyz']);self.size=np.array([li['size_'+a] for a in 'xyz']);self.mn=torch.tensor(self.lo,dtype=torch.float32,device='cuda');self.sz=torch.tensor(self.size,dtype=torch.float32,device='cuda');self.net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=self.mn,size_xyz=self.sz)
  with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
  self.net.load_state_dict(ck['model_state_dict'],strict=True);self.net.cuda().eval();del ck
  self.crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});self.pad=e.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np});self.solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});self.verified=False
 def predict(self,img,K,seed=42):
  start=sync();t0=start;pred=self.det.predict(source=img,imgsz=640,conf=self.conf,iou=.45,max_det=10,device=0,verbose=False,save=False)[0];t1=sync();r={'detected':False,'pnp_valid':False,'failure':'','confidence':None,'bbox_xyxy':None,'R':None,'t':None,'timing_ms':{}}
  if len(pred.boxes)==0:r['failure']='detector_miss';r['timing_ms']={'detector_predict_including_nms':(t1-start)*1000,'end_to_end':(t1-start)*1000};return r
  # YOLO.predict has already performed class filtering and NMS. Here select top confidence only.
  j=int(pred.boxes.conf.argmax().item());xyxy=pred.boxes.xyxy[j].detach().cpu().numpy().astype(float);r['confidence']=float(pred.boxes.conf[j]);r['bbox_xyxy']=xyxy.tolist();r['detected']=True;box=self.pad(np.r_[xyxy[:2],xyxy[2:]-xyxy[:2]],1.5);t2=sync()
  crop=self.crop(img,box,256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(crop.transpose(2,0,1))).float()/255;x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None];x=x[None].cuda();bt=torch.tensor(box[None],dtype=torch.float32,device='cuda');t3=sync()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=self.net.net(x)
  t4=sync()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
   mask=torch.sigmoid(pm)[:,0]>.5;bits=(torch.sigmoid(pc)>.5).to(pc.dtype).permute(0,2,3,1);front=self.net.hcce_decode(bits[...,:24])/255;back=self.net.hcce_decode(bits[...,24:])/255;yy=torch.arange(front.shape[1],device='cuda').float()/front.shape[1];xx=torch.arange(front.shape[2],device='cuda').float()/front.shape[2];X,Y=torch.meshgrid(xx,yy,indexing='xy');uv=torch.stack([X,Y],dim=-1)[None]*bt[:,None,None,2:]+bt[:,None,None,:2];xyz=front*self.sz+self.mn;backxyz=back*self.sz+self.mn
   m=mask[0].cpu().numpy();xyz_np=xyz[0].float().cpu().numpy();uv_np=uv[0].float().cpu().numpy();back_np=backxyz[0].float().cpu().numpy()
  t5=sync()
  if not self.verified:
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):ref=self.net.inference_batch(x,bt)
   assert np.array_equal(m,ref['pred_mask'][0].cpu().numpy()>0) and np.array_equal(xyz_np,ref['pred_front_code_obj'][0].float().cpu().numpy()) and np.array_equal(uv_np,ref['coord_2d_image'][0].float().cpu().numpy()) and np.array_equal(back_np,ref['pred_back_code_obj'][0].float().cpu().numpy());self.verified=True
  try:
   if m.sum()<=4:raise ValueError('mask <=4 pixels')
   cv2.setRNGSeed(seed);res=self.solve((m,xyz_np,uv_np,np.asarray(K)),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
   if not res['success']:raise ValueError('PnP failed')
   R=np.asarray(res['rot'],float);t=np.asarray(res['tvecs'],float).reshape(3)
   if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((self.v@R.T+t)[:,2])<=0:raise ValueError('invalid pose')
   r.update(pnp_valid=True,R=R.tolist(),t=t.tolist())
  except Exception as ex:r['failure']='pnp: '+str(ex)
  t6=sync();r['timing_ms']={'detector_predict_including_nms':(t1-t0)*1000,'bbox_selection':(t2-t1)*1000,'crop_preprocess_h2d':(t3-t2)*1000,'hcce_forward':(t4-t3)*1000,'hcce_decode_copy':(t5-t4)*1000,'pnp':(t6-t5)*1000,'end_to_end':(t6-t0)*1000};return r
 def overlay(self,img,result):
  out=img.copy()
  if result['detected']:
   x1,y1,x2,y2=map(int,result['bbox_xyxy']);cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),2)
  if result['pnp_valid']:
   uv=e.project(self.v,np.array(result['R']),np.array(result['t']),self.K);polys=[np.clip(uv[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in self.faces];cv2.polylines(out,polys,True,(255,0,255),1,cv2.LINE_AA)
  return out

def main():
 ap=argparse.ArgumentParser(description='Input RGB must be undistorted using the specified output K, unless --dist is supplied.');ap.add_argument('--image',required=True);ap.add_argument('--camera',required=True,help='JSON with camera_matrix; optionally distortion_coefficients');ap.add_argument('--dist',action='store_true',help='undistort raw image with JSON distortion_coefficients, output K=input K');ap.add_argument('--weights',default=str(P/'runs/yolo11n_pbr_attempt3/weights/best.pt'));ap.add_argument('--conf',type=float,default=.25);ap.add_argument('--out',required=True,help='new output directory');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);im=cv2.imread(a.image);assert im is not None;cam=json.load(open(a.camera));K=np.asarray(cam['camera_matrix'],float).reshape(3,3)
 if a.dist:im=cv2.undistort(im,K,np.asarray(cam['distortion_coefficients']),None,K)
 pipe=Pipeline(a.weights,a.conf);pipe.K=K;r=pipe.predict(im,K);json.dump(r,(out/'pose.json').open('x'),indent=2);cv2.imwrite(str(out/'overlay.png'),pipe.overlay(im,r));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
