from pathlib import Path
import sys,cv2,numpy as np,torch,csv,json,hashlib,shutil,subprocess,time
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;PRE=P/'bb100_robotseg_phase16a2r_resume_20261110';VIS=P/'bb100_visible_mask_phase16a1_20261110';sys.path.insert(0,str(PRE/'robotseg_source'))
from robotseg.build_robotseg import build_robotseg_video_predictor
from robotseg.modeling import memory_attention_with_structure as structure
CK=Path('/home/oylee/projects/HCCEPose/robotseg.pt')
SET={75:{'roi':[220,0,700,430],'positive':[465,295],'negative':[505,330]},235:{'roi':[390,0,890,540],'positive':[640,385],'negative':[690,425]}}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 assert not p.exists() and cv2.imwrite(str(p),v)
def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip(),'GPU busy';assert structure._HAS_KORNIA
 with (VIS/'selected_frames.csv').open(encoding='utf-8-sig') as f:rows={int(r['frame']):r for r in csv.DictReader(f)}
 for x in ['inputs','masks','panels','full']: (O/x).mkdir()
 guard={str(CK):sha(CK)};assert guard[str(CK)]=='1384c38c4f313487a32503025d4566ac1d815f6be983199362fc4af19aaa06a2'
 for i in SET:
  for p in [Path(rows[i]['rgb_path']),VIS/'sam'/f'{i:06d}_selected_NOT_GT.png',VIS/'cad'/f'{i:06d}_AMODAL_NOT_VISIBLE_GT.png',PRE/('smoke' if i==235 else 'remaining')/'masks'/f'{i:06d}_robotseg_gripper_NOT_GT.png']:guard[str(p)]=sha(p)
 torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();model=build_robotseg_video_predictor('configs/robotseg-infer',str(CK),device='cuda',mode='eval',apply_postprocessing=False)
 results=[]
 for frame,s in SET.items():
  imgpath=Path(rows[frame]['rgb_path']);img=cv2.imread(str(imgpath));cad=cv2.imread(str(VIS/'cad'/f'{frame:06d}_AMODAL_NOT_VISIBLE_GT.png'),0)>0;sam=cv2.imread(str(VIS/'sam'/f'{frame:06d}_selected_NOT_GT.png'),0)>0;x0,y0,x1,y1=s['roi'];assert img.shape==(720,1280,3) and 0<=x0<x1<=1280 and 0<=y0<y1<=720
  ys,xs=np.nonzero(cad);assert xs.min()>=x0 and xs.max()<x1 and ys.min()>=y0 and ys.max()<y1
  for x,y in [s['positive'],s['negative']]:assert x0<=x<x1 and y0<=y<y1
  for kind in ('full','crop'):
   d=O/'inputs'/f'{frame:06d}_{kind}';d.mkdir();path=d/'000000.jpg'
   if kind=='full':shutil.copyfile(imgpath,path);assert sha(imgpath)==sha(path)
   else:assert cv2.imwrite(str(path),img[y0:y1,x0:x1],[cv2.IMWRITE_JPEG_QUALITY,100])
  for condition,kind,guided in [('A_full_auto','full',False),('B_crop_auto','crop',False),('C_full_points','full',True),('D_crop_points','crop',True)]:
   t=time.monotonic();d=O/'inputs'/f'{frame:06d}_{kind}'
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=model.init_state(video_path=str(d),async_loading_frames=False,offload_video_to_cpu=False,offload_state_to_cpu=False)
    if guided:
     offset=np.array([0,0] if kind=='full' else [x0,y0]);points=np.array([s['positive'],s['negative']],np.float32)-offset
     _,_,logits=model.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=0,robots='gripper',points=points,labels=np.array([1,0],np.int32),normalize_coords=True)
    else:_,_,logits=model.add_new_robot(inference_state=state,frame_idx=0,obj_id=0,robot='gripper')
   torch.cuda.synchronize();h,w=(720,1280) if kind=='full' else (y1-y0,x1-x0);assert logits.shape==(1,1,h,w) and bool(torch.isfinite(logits).all());local=(logits[0,0]>0).cpu().numpy();mask=np.zeros((720,1280),bool)
   if kind=='full':mask=local
   else:mask[y0:y1,x0:x1]=local
   prefix=f'{frame:06d}_{condition}';save(O/'masks'/f'{prefix}_gripper_NOT_GT.png',mask.astype('uint8')*255);save(O/'masks'/f'{prefix}_SAM_minus_NOT_GT.png',(sam&~mask).astype('uint8')*255);save(O/'masks'/f'{prefix}_CADSAM_minus_NOT_GT.png',(cad&sam&~mask).astype('uint8')*255);save(O/'masks'/f'{prefix}_removed_SAM_NOT_GT.png',(sam&mask).astype('uint8')*255)
   prev=cv2.imread(str(PRE/('smoke' if frame==235 else 'remaining')/'masks'/f'{frame:06d}_robotseg_gripper_NOT_GT.png'),0)>0;diff=int(np.count_nonzero(mask^prev));r={'frame':frame,'condition':condition,'roi_xyxy':s['roi'] if kind=='crop' else [0,0,1280,720],'point_positive_xy':s['positive'] if guided else None,'point_negative_xy':s['negative'] if guided else None,'mask_pixels':int(mask.sum()),'SAM_overlap_removed_pixels':int((mask&sam).sum()),'CADSAM_overlap_removed_pixels':int((mask&sam&cad).sum()),'mask_diff_vs_prior_full_pixels':diff,'baseline_matches_previous':bool(diff==0) if condition=='A_full_auto' else None,'logit_min':float(logits.min().cpu()),'logit_max':float(logits.max().cpu()),'seconds':round(time.monotonic()-t,3),'status':'PSEUDO_NOT_GT','roi_validity':'CAD and prompt inside; physical gripper extent needs human confirmation' if kind=='crop' else 'FULL_FRAME'};results.append(r);print(json.dumps(r),flush=True)
   del state,logits
  # Full-frame overlay explicitly shows any predictions outside ROI, and guides visible.
  panel=img.copy();cv2.rectangle(panel,(x0,y0),(x1-1,y1-1),(0,255,255),2);cv2.drawMarker(panel,tuple(s['positive']),(0,255,0),cv2.MARKER_CROSS,18,2);cv2.drawMarker(panel,tuple(s['negative']),(0,0,255),cv2.MARKER_CROSS,18,2);save(O/'full'/f'{frame:06d}_ROI_PROMPTS.jpg',panel)
  def over(mask,col):
   z=img.copy();z[mask]=(z[mask]*.48+np.array(col)*.52).astype('uint8');return z
  panels=[(img,'RGB'),(over(cad,[0,220,220]),'CAD AMODAL'),(panel,'ROI / POINTS'),(over(sam,[255,0,200]),'SAM BB NOT GT')]
  for cond in ('A_full_auto','B_crop_auto','C_full_points','D_crop_points'):
   m=cv2.imread(str(O/'masks'/f'{frame:06d}_{cond}_gripper_NOT_GT.png'),0)>0;panels.append((over(m,[0,0,255]),cond+' GRIPPER'))
  d=cv2.imread(str(O/'masks'/f'{frame:06d}_D_crop_points_CADSAM_minus_NOT_GT.png'),0)>0;rem=cv2.imread(str(O/'masks'/f'{frame:06d}_D_crop_points_removed_SAM_NOT_GT.png'),0)>0;panels += [(over(d,[0,200,0]),'D CAD SAM - GRIPPER'),(over(rem,[0,0,255]),'D REMOVED FROM SAM')]
  xx0=max(0,x0);yy0=max(0,y0);xx1=x1;yy1=y1
  def tile(a,title):
   crop=a[yy0:yy1,xx0:xx1];out=np.zeros((350,400,3),np.uint8);h,w=crop.shape[:2];k=min(400/w,320/h);nh,nw=int(h*k),int(w*k);out[25:25+nh,(400-nw)//2:(400-nw)//2+nw]=cv2.resize(crop,(nw,nh));cv2.putText(out,title,(4,17),0,.48,(255,255,255),1);return out
  arr=[tile(a,t) for a,t in panels];save(O/'panels'/f'{frame:06d}_comparison.jpg',np.vstack([np.hstack(arr[:5]),np.hstack(arr[5:])]))
 assert all(sha(Path(p))==v for p,v in guard.items());report={'frames':results,'source_sha_unchanged':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'input_crop_encoding':'JPEG quality 100 in separate directory','model':'official source, apply_postprocessing=False, raw logit >0','coords':'original full-frame pixels; crop offset subtracted for prompt and added for masks','positive_negative':'manually chosen from inspected frames; not GT','gpu':'RTX 5090'};(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 with (O/'results.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
if __name__=='__main__':main()
