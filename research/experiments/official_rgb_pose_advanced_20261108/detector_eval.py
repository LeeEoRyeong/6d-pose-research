"""YOLO detections use only RGB; GT is used only after prediction for evaluation."""
import os,sys,csv,json,time,itertools,torch,numpy as np,cv2
from pathlib import Path
HERE=Path(__file__).resolve().parent;os.environ['YOLO_CONFIG_DIR']=str(HERE/'.ultralytics');os.environ['MPLCONFIGDIR']=str(HERE/'.mpl')
from ultralytics import YOLO
OLD=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_symmetry_eval_20261108');sys.path.insert(0,str(OLD));import evaluate as e,symmetry

def box_iou(a,b):
 x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[2],b[2]);yy=min(a[3],b[3]);inter=max(0,xx-x)*max(0,yy-y);return inter/(max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter+1e-9)
def main():
 assert json.load((HERE/'reproduction.json').open())['status']=='PASS';torch.set_num_threads(2);cv2.setNumThreads(1)
 apps=e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not apps,apps
 v,faces,syms,info=symmetry.load();data,_=e.get_data(v);images=data['real'];model_path=e.EXP/'realistic_final100/yolo11/train_obj_s/smoke_train/weights/best.pt';assert model_path.is_file();before=e.sha(model_path)
 det=YOLO(str(model_path));assert det.names=={0:'4'};mod=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+a] for a in 'xyz']);sz=np.array([li['size_'+a] for a in 'xyz']);minimum=torch.tensor(lo,dtype=torch.float32,device='cuda');size=torch.tensor(sz,dtype=torch.float32,device='cuda')
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=minimum,size_xyz=size);net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval()
 crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});pad=e.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});odir=HERE/'overlays/detector';odir.mkdir(exist_ok=False);rows=[]
 for s in images:
  # Only s['image'] enters detector. No CAD, ROI, mask or GT is passed to YOLO.
  torch.cuda.synchronize();start=time.perf_counter();pred=det.predict(source=s['image'],imgsz=640,conf=.25,iou=.45,max_det=10,device=0,verbose=False,save=False)[0];torch.cuda.synchronize();det_ms=1000*(time.perf_counter()-start)
  boxes=pred.boxes;num=len(boxes);row={'frame':s['id'],'num_detections':num,'detected':bool(num),'confidence':'','detector_bbox_xyxy':'','detector_vs_cad_amodal_bbox_iou':'','pnp_valid':False,'failure':'','ADD_S':'','symmetry_ADD':'','ADD_S_pass':False,'symmetry_ADD_pass':False,'detector_ms_including_python':det_ms}
  view=s['image'].copy()
  if not num:row['failure']='no detection at conf>=0.25'
  else:
   xyxy=boxes.xyxy[0].detach().cpu().numpy().astype(float);score=float(boxes.conf[0].detach().cpu());row['confidence']=score;row['detector_bbox_xyxy']=json.dumps(xyxy.tolist());bbox=np.r_[xyxy[:2],xyxy[2:]-xyxy[:2]];box=pad(bbox,1.5);cv2.rectangle(view,tuple(xyxy[:2].astype(int)),tuple(xyxy[2:].astype(int)),(0,255,255),2)
   # GT projected bbox is used only for the following diagnostic metric and pose error.
   truth=e.project(v,s['R'],s['t'],s['K']);row['detector_vs_cad_amodal_bbox_iou']=box_iou(xyxy,np.r_[truth.min(0),truth.max(0)])
   cropim=crop(s['image'],box,256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(cropim.transpose(2,0,1)),dtype=torch.float32)/255;x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
   try:
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x[None].cuda(),torch.tensor(box[None],dtype=torch.float32,device='cuda'))
    mask=out['pred_mask'][0].cpu().numpy()>0;front=out['pred_front_code_obj'][0].float().cpu().numpy();uv=out['coord_2d_image'][0].float().cpu().numpy()
    if mask.sum()<=4:raise ValueError('mask has <=4 points')
    cv2.setRNGSeed(42+s['id']);p=solve((mask,front,uv,s['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
    if not p['success']:raise ValueError('PnP returned success=False')
    R=np.asarray(p['rot'],float);t=np.asarray(p['tvecs'],float).reshape(3)
    if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:raise ValueError('invalid pose')
    aa=symmetry.add_s(v,R,t,s['R'],s['t']);ss,_=symmetry.symmetry_add(v,R,t,s['R'],s['t'],syms);th=.1*info['diameter'];row.update(pnp_valid=True,ADD_S=aa,symmetry_ADD=ss,ADD_S_pass=aa<th,symmetry_ADD_pass=ss<th)
    uvp=e.project(v,R,t,s['K']);polys=[np.clip(uvp[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in faces];cv2.polylines(view,polys,True,(255,0,255),1,cv2.LINE_AA)
   except Exception as ex:row['failure']=type(ex).__name__+': '+str(ex)
  cv2.putText(view,f'{s["id"]:06d} detections={num} PnP={row["pnp_valid"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);e.png(odir/f'{s["id"]:06d}.png',view);rows.append(row);print('DETECT',s['id'],num,row['failure'],flush=True)
 assert len(rows)==27 and e.sha(model_path)==before
 with (HERE/'detector_eval.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'frames':27,'detected':sum(r['detected'] for r in rows),'pnp_valid':sum(r['pnp_valid'] for r in rows),'adds_pass_all_frames':sum(r['ADD_S_pass'] for r in rows),'symmetry_pass_all_frames':sum(r['symmetry_ADD_pass'] for r in rows),'confidence_threshold':.25,'bbox_definition':'highest scoring class 0 detector bbox, pad1.5; CAD only for offline comparison','checkpoint':str(model_path),'checkpoint_sha256':before,'detector_training':'one epoch, fraction 0.1, from scratch; PBR smoke only','iou_note':'bbox IoU against CAD amodal projection, not visible GT','detector_input':'RAW undistorted RGB only; no GT','independent_test':'no, development27'}
 with (HERE/'detector_summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
 print('DETECTOR SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
