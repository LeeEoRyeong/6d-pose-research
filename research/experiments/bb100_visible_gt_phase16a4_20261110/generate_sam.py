from pathlib import Path
import csv,json,hashlib,subprocess,time
import numpy as np,cv2,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
O=Path(__file__).resolve().parent;P=O.parent;OLD=P/'bb100_visible_mask_phase16a1_20261110';PH=P/'bb100_placement_targets_phase15_20261109';CK=P/'bb_sam_candidates_20261108/sam2.1_hiera_tiny.pt'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,a):
 assert not p.exists() and cv2.imwrite(str(p),a)
def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip(),'GPU busy'
 with (P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 assert len(rows)==100 and len({x['frame'] for x in rows})==100 and sha(CK)=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
 for k in ('A','B','C'):(O/'candidates'/k).mkdir()
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(CK),device='cuda',mode='eval',apply_postprocessing=False);predictor=SAM2ImagePredictor(model);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();out=[];start=time.monotonic()
 for j,r in enumerate(rows):
  i=int(r['frame']);fr=f'{i:06d}';image=Path(r['image_path']);cadpath=PH/'amodal_masks'/f"pose_{r['pose_group']}.png";img=cv2.imread(str(image));cad=cv2.imread(str(cadpath),0);assert img.shape==(720,1280,3) and cad.shape==(720,1280) and set(np.unique(cad))<={0,255};reused=OLD/'sam'/f'{fr}_selected_NOT_GT.png';scores=None;point=None
  if reused.exists():sam=cv2.imread(str(reused),0);origin='Phase16A1_existing'
  else:
   rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);x,y,w,h=map(int,cv2.boundingRect(cad));box=np.array([x,y,x+w-1,y+h-1],np.float32);dist=cv2.distanceTransform(cad,cv2.DIST_L2,5);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);ok=(dist>=10)&(gray<=90)
   if ok.any():py,px=np.unravel_index(int(np.argmax(np.where(ok,dist,-1))),dist.shape);point=np.array([[px,py]],np.float32)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    predictor.set_image(rgb);m,sc,_=predictor.predict(point_coords=point,point_labels=np.ones(1,np.int32) if point is not None else None,box=box,multimask_output=True)
   sam=m[int(np.argmax(sc))].astype('uint8')*255;scores=[float(x) for x in sc];origin='SAM2_new'
  assert sam.shape==cad.shape and set(np.unique(sam))<={0,255};c=cv2.bitwise_and(cad,sam)
  for k,a in [('A',cad),('B',sam),('C',c)]:save(O/'candidates'/k/f'{fr}.png',a)
  out.append({'frame':i,'pose_group':r['pose_group'],'occlusion':r['occlusion'],'image_path':str(image),'image_sha256':sha(image),'pose_path':r['pose_path'],'pose_sha256':r['pose_sha256'],'K':r['K'],'sam_source':origin,'sam_point_xy':None if point is None else list(map(int,point[0])),'sam_predicted_scores_not_accuracy':scores,'cad_sha256':sha(cadpath),'sam_area':int((sam>0).sum()),'cad_area':int((cad>0).sum()),'sam_cad_intersection':int((c>0).sum()),'status':'AUTO_CANDIDATE_NOT_GT'})
  if (j+1)%20==0:print('SAM frames',j+1,flush=True)
 info={'frames':out,'n':len(out),'reused':sum(x['sam_source']=='Phase16A1_existing' for x in out),'new_inference':sum(x['sam_source']=='SAM2_new' for x in out),'checkpoint_sha256':sha(CK),'gpu':torch.cuda.get_device_name(0),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'seconds':round(time.monotonic()-start,2),'warning':'candidate only, prompts not guaranteed foreground'};(O/'sam_generation.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n');print('SAM done',info['reused'],info['new_inference'],info['seconds'],flush=True)
if __name__=='__main__':main()
