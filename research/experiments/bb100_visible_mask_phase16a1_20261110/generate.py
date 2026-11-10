from pathlib import Path
import csv,json,hashlib,time,subprocess,sys
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent
OLD=P/'bb_sam_candidates_20261108';PH15=P/'bb100_placement_targets_phase15_20261109'; CK=OLD/'sam2.1_hiera_tiny.pt'
SELECTION={43:'가림 없음; 기존 SAM 비교 및 구멍 오류 기준',247:'가림 없음; 기존 SAM 비교, 235와 같은 Pose의 가림 전후 비교',383:'가림 없음; 후반 촬영 구간과 다른 Pose',420:'가림 없음; 마지막 구간의 다른 Pose',75:'부분 가림; 기존 SAM 비교, 중앙 그리퍼',235:'부분 가림; 기존 SAM 비교 및 247과 같은 Pose의 가림 전후',99:'부분 가림; 첫 구간 중간의 다른 Pose',221:'부분 가림; 촬영 간격 이후 두 번째 구간의 다른 Pose',340:'부분 가림; 후반 구간의 Pose 변화',359:'부분 가림; 후반 반복 Pose 및 그리퍼 상태'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def png(p,a):
 assert not p.exists() and cv2.imwrite(str(p),a)
def prompt(rgb,cad):
 x,y,w,h=map(int,cv2.boundingRect(cad));box=np.array([x,y,x+w-1,y+h-1],np.float32);d=cv2.distanceTransform(cad,cv2.DIST_L2,5);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);allowed=(d>=10)&(gray<=90)
 if not allowed.any():return box,None,{'criterion':'no safe dark interior candidate, box only'}
 py,px=np.unravel_index(int(np.argmax(np.where(allowed,d,-1))),d.shape);return box,np.array([[px,py]],np.float32),{'criterion':'dark pixel among CAD interior, not verified foreground','gray':int(gray[py,px]),'edge_distance_px':float(d[py,px])}
def main():
 with (P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f:allrows=list(csv.DictReader(f))
 records={int(x['frame']):x for x in allrows};assert len(allrows)==100 and len(SELECTION)==10
 selected=[records[i] for i in SELECTION];assert sum(r['occlusion']=='NONE' for r in selected)==4 and sum(r['occlusion']=='PARTIAL' for r in selected)==6
 assert len({r['pose_group'] for r in selected})==9
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip();assert not gpu,'GPU currently in use; inference aborted'
 assert torch.cuda.is_available()
 for x in ['sam','cad','intersection','uncertainty','package','package/frames']:(O/x).mkdir()
 watched={str(CK):sha(CK)}
 for r in selected:
  ip=Path(r['image_path']);cp=PH15/'amodal_masks'/f"pose_{r['pose_group']}.png";watched[str(ip)]=sha(ip);watched[str(cp)]=sha(cp)
 assert watched[str(CK)]=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(CK),device='cuda',mode='eval',apply_postprocessing=False);predictor=SAM2ImagePredictor(model);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();t0=time.monotonic();summary=[];meta=[]
 for frame,reason in SELECTION.items():
  r=records[frame];bgr=cv2.imread(r['image_path']);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);cad=cv2.imread(str(PH15/'amodal_masks'/f"pose_{r['pose_group']}.png"),0);assert bgr.shape==(720,1280,3) and set(np.unique(cad))<={0,255}
  box,pt,pinfo=prompt(rgb,cad)
  source=OLD/'results/run_20261108T140008_86355c19'/f'{frame:06d}_pseudo_mask_candidate_selected.png'
  if source.exists():
   sam=cv2.imread(str(source),0);watched[str(source)]=sha(source);origin='existing SAM candidate';scores=None;selected_index=None
  else:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    predictor.set_image(rgb);masks,scores,_=predictor.predict(point_coords=pt,point_labels=np.ones(len(pt),np.int32) if pt is not None else None,box=box,multimask_output=True)
   torch.cuda.synchronize();selected_index=int(np.argmax(scores));sam=masks[selected_index].astype('uint8')*255;scores=[float(v) for v in scores];origin='new SAM inference';
   for i,m in enumerate(masks):png(O/'sam'/f'{frame:06d}_candidate_{i}_NOT_GT.png',m.astype('uint8')*255)
  assert sam.shape==(720,1280) and set(np.unique(sam))<={0,255} and (sam>0).any()
  inter=((sam>0)&(cad>0)).astype('uint8')*255
  # Safe diagnostic treatment: uncertain throughout CAD or SAM, no asserted foreground or background training gate.
  uncertain=((sam>0)|(cad>0)).astype('uint8')*128
  png(O/'sam'/f'{frame:06d}_selected_NOT_GT.png',sam);png(O/'cad'/f'{frame:06d}_AMODAL_NOT_VISIBLE_GT.png',cad);png(O/'intersection'/f'{frame:06d}_CAD_SAM_NOT_GT.png',inter);png(O/'uncertainty'/f'{frame:06d}_unknown_128_no_supervision.png',uncertain)
  x,y,w,h=map(int,cv2.boundingRect(cad));pad=max(55,int(max(w,h)*.30));x0=max(0,x-pad);y0=max(0,y-pad);x1=min(1280,x+w+pad);y1=min(720,y+h+pad)
  def col(m):return cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
  over=bgr.copy();sel=inter>0;over[sel]=(over[sel]*.55+np.array([0,200,0])*.45).astype('uint8');cnt,_=cv2.findContours(cad,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,cnt,-1,(0,220,255),1)
  rawover=bgr.copy();sel=sam>0;rawover[sel]=(rawover[sel]*.55+np.array([255,0,180])*.45).astype('uint8')
  diff=np.zeros_like(bgr);diff[(cad>0)&(sam>0)]=[50,200,50];diff[(cad>0)&(sam==0)]=[0,0,255];diff[(cad==0)&(sam>0)]=[255,100,0]
  unknown=np.zeros_like(bgr);unknown[uncertain>0]=[0,190,255]
  panels=[bgr,col(cad),col(sam),rawover,col(inter),over,diff,unknown];labels=['RGB','CAD AMODAL NOT GT','SAM NOT GT','SAM ON RGB','CAD & SAM NOT GT','INTERSECTION ON RGB','BOTH green / CAD red / SAM blue','UNKNOWN UNION; NO SUPERVISION']
  tiles=[]
  for im,label in zip(panels,labels):
   crop=im[y0:y1,x0:x1];tile=np.zeros((360,420,3),np.uint8);h0,w0=crop.shape[:2];scale=min(420/w0,330/h0);ww,hh=max(1,int(w0*scale)),max(1,int(h0*scale));tile[28:28+hh,(420-ww)//2:(420-ww)//2+ww]=cv2.resize(crop,(ww,hh));cv2.putText(tile,label,(4,18),0,.48,(255,255,255),1,cv2.LINE_AA);tiles.append(tile)
  panel=np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]);assert cv2.imwrite(str(O/'package/frames'/f'{frame:06d}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,91])
  # Save uncropped original as separate offline reference; re-encoding only a new copy.
  assert cv2.imwrite(str(O/'package/frames'/f'{frame:06d}_RGB.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,92])
  row={'frame':frame,'occlusion':r['occlusion'],'pose_group':r['pose_group'],'capture_time':r['capture_time'],'reason':reason,'rgb_path':r['image_path'],'sam_source':origin,'cad_area_pixels':int((cad>0).sum()),'sam_area_pixels':int((sam>0).sum()),'intersection_area_pixels':int((inter>0).sum()),'cad_only_pixels':int(((cad>0)&(sam==0)).sum()),'sam_only_pixels':int(((cad==0)&(sam>0)).sum()),'unknown_area_pixels':int((uncertain>0).sum()),'validated_training_pixels':0,'status':'UNREVIEWED_NOT_GT'};summary.append(row)
  meta.append({'frame':frame,'bbox_xyxy':box.astype(int).tolist(),'positive_point_xy':None if pt is None else pt[0].astype(int).tolist(),'prompt_info':pinfo,'sam_source':origin,'selected_index_new_run':selected_index,'model_scores_new_run':scores,'scores_are_not_accuracy':True})
  print('frame',frame,origin,'CAD',row['cad_area_pixels'],'SAM',row['sam_area_pixels'],flush=True)
 with (O/'selected_frames.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
 after={p:sha(Path(p)) for p in watched};assert watched==after
 info={'model':'SAM 2.1 hiera tiny, local checkpoint','checkpoint_sha256':watched[str(CK)],'frames':meta,'six_new_inferences_only':[m['frame'] for m in meta if m['sam_source']=='new SAM inference'],'four_reused':[m['frame'] for m in meta if m['sam_source']=='existing SAM candidate'],'gpu':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_sec':round(time.monotonic()-t0,2),'source_hashes_unchanged':True,'source_hashes':watched,'uncertainty_rule':'128=CAD union SAM uncertain, 0=not assigned, NO supervised foreground or background'}
 (O/'generation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in info.items() if k not in ('source_hashes','frames')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
