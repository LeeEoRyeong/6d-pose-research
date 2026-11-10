from pathlib import Path
import sys,json,csv,hashlib,subprocess,time,argparse,shutil
import cv2,numpy as np,torch
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;src=O/'robotseg_source';sys.path.insert(0,str(src));from robotseg.build_robotseg import build_robotseg_video_predictor
from robotseg.modeling import memory_attention_with_structure as structure
CK=Path('/home/oylee/projects/HCCEPose/robotseg.pt');PREV=P/'bb100_visible_mask_phase16a1_20261110'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,img):
 assert not p.exists() and cv2.imwrite(str(p),img)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frames',nargs='+',type=int,required=True);ap.add_argument('--tag',required=True);args=ap.parse_args();allowed={75,99,221,235,340,359};assert set(args.frames)<=allowed and len(set(args.frames))==len(args.frames) and structure._HAS_KORNIA
 if args.tag!='smoke':assert (O/'smoke.json').exists() and json.loads((O/'smoke.json').read_text())['status']=='PASS'
 running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip();assert not running,'another compute GPU process active'
 with (PREV/'selected_frames.csv').open(encoding='utf-8-sig') as f:rows={int(r['frame']):r for r in csv.DictReader(f)}
 before={str(CK):sha(CK)};assert before[str(CK)]=='1384c38c4f313487a32503025d4566ac1d815f6be983199362fc4af19aaa06a2'
 out=O/args.tag;out.mkdir();(out/'frames').mkdir();(out/'masks').mkdir();(out/'panels').mkdir()
 for i in args.frames:
  im=Path(rows[i]['rgb_path']);cad=PREV/'cad'/f'{i:06d}_AMODAL_NOT_VISIBLE_GT.png';sam=PREV/'sam'/f'{i:06d}_selected_NOT_GT.png';inter=PREV/'intersection'/f'{i:06d}_CAD_SAM_NOT_GT.png'
  for p in (im,cad,sam,inter):before[str(p)]=sha(p)
 torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();net=build_robotseg_video_predictor('configs/robotseg-infer',str(CK),device='cuda',mode='eval',apply_postprocessing=False);rlist=[]
 for i in args.frames:
  t0=time.monotonic();r=rows[i];imgpath=Path(r['rgb_path']);sub=out/'frames'/f'{i:06d}';sub.mkdir();shutil.copyfile(imgpath,sub/'000000.jpg');assert sha(sub/'000000.jpg')==before[str(imgpath)]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=net.init_state(video_path=str(sub),async_loading_frames=False,offload_video_to_cpu=False,offload_state_to_cpu=False)
   _,obj,logits=net.add_new_robot(inference_state=state,frame_idx=0,obj_id=0,robot='gripper')
  torch.cuda.synchronize();assert logits.shape==(1,1,720,1280) and bool(torch.isfinite(logits).all());grip=(logits[0,0]>0).cpu().numpy();sam=cv2.imread(str(PREV/'sam'/f'{i:06d}_selected_NOT_GT.png'),0)>0;cad=cv2.imread(str(PREV/'cad'/f'{i:06d}_AMODAL_NOT_VISIBLE_GT.png'),0)>0;base=cv2.imread(str(PREV/'intersection'/f'{i:06d}_CAD_SAM_NOT_GT.png'),0)>0;assert np.array_equal(base,cad&sam)
  masks={'robotseg_gripper_NOT_GT':grip,'A_sam_NOT_GT':sam,'B_cad_sam_NOT_GT':base,'C_sam_minus_gripper_NOT_GT':sam&~grip,'D_cad_sam_minus_gripper_NOT_GT':base&~grip,'E_cad_minus_gripper_NOT_GT':cad&~grip,'removed_from_sam_NOT_GT':sam&grip,'removed_from_cad_NOT_GT':cad&grip}
  for name,m in masks.items():write(out/'masks'/f'{i:06d}_{name}.png',m.astype('uint8')*255)
  im=cv2.imread(str(imgpath));assert im is not None;x,y,w,h=map(int,cv2.boundingRect(cad.astype('uint8')));p=max(55,int(max(w,h)*.3));x0=max(0,x-p);x1=min(1280,x+w+p);y0=max(0,y-p);y1=min(720,y+h+p)
  def over(m,color):
   z=im.copy();z[m]=(z[m]*.5+np.array(color)*.5).astype('uint8');return z
  def tile(im,txt):
   crop=im[y0:y1,x0:x1];HH,WW=crop.shape[:2];scale=min(420/WW,315/HH);ww,hh=max(1,int(WW*scale)),max(1,int(HH*scale));out=np.zeros((350,420,3),np.uint8);out[28:28+hh,(420-ww)//2:(420-ww)//2+ww]=cv2.resize(crop,(ww,hh));cv2.putText(out,txt,(4,18),0,.49,(255,255,255),1,cv2.LINE_AA);return out
  views=[(im,'RGB'),(over(cad,[0,220,220]),'CAD AMODAL NOT GT'),(over(sam,[255,0,200]),'A SAM NOT GT'),(over(grip,[0,0,255]),'ROBOTSEG GRIPPER NOT GT'),(over(base,[0,210,0]),'B CAD & SAM NOT GT'),(over(masks['C_sam_minus_gripper_NOT_GT'],[255,0,200]),'C SAM - ROBOTSEG NOT GT'),(over(masks['D_cad_sam_minus_gripper_NOT_GT'],[0,210,0]),'D CAD & SAM - ROBOTSEG'),(over(masks['E_cad_minus_gripper_NOT_GT'],[0,210,220]),'E CAD - ROBOTSEG'),(over(masks['removed_from_sam_NOT_GT'],[0,0,255]),'REMOVED FROM SAM NOT GT'),(over(masks['removed_from_cad_NOT_GT'],[0,0,255]),'REMOVED FROM CAD NOT GT')]
  tiles=[tile(v,t) for v,t in views];write(out/'panels'/f'{i:06d}_comparison.jpg',np.vstack([np.hstack(tiles[:5]),np.hstack(tiles[5:])]))
  row={'frame':i,'image_shape':[720,1280],'robotseg_gripper_pixels':int(grip.sum()),'sam_pixels':int(sam.sum()),'cad_pixels':int(cad.sum()),'cad_sam_pixels':int(base.sum()),'removed_sam_pixels':int((sam&grip).sum()),'removed_cad_sam_pixels':int((base&grip).sum()),'removed_cad_pixels':int((cad&grip).sum()),'logit_min':float(logits.min().cpu()),'logit_max':float(logits.max().cpu()),'seconds':round(time.monotonic()-t0,3),'status':'PSEUDO_MASK_UNREVIEWED_NOT_GT'};rlist.append(row);print(json.dumps(row),flush=True);del state,logits
 assert all(sha(Path(p))==h for p,h in before.items())
 summary={'status':'PASS','tag':args.tag,'frames':rlist,'checkpoint_sha256':before[str(CK)],'source_hashes_unchanged':True,'device':torch.cuda.get_device_name(0),'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),'postprocessing':False,'guided_refine':False,'threshold':'raw logit > 0','semantic_category':'gripper','kornia':True,'architecture_strict_load':'PASS','predictions':'NOT_GT'}
 (O/f'{args.tag}.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
