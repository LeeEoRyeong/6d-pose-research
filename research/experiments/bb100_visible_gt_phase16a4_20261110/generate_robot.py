from pathlib import Path
import sys,json,hashlib,subprocess,shutil,time,cv2,numpy as np,torch
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;PRE=P/'bb100_robotseg_phase16a2r_resume_20261110';CK=Path('/home/oylee/projects/HCCEPose/robotseg.pt');sys.path.insert(0,str(PRE/'robotseg_source'))
from robotseg.build_robotseg import build_robotseg_video_predictor
from robotseg.modeling import memory_attention_with_structure as structure
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,a):
 assert not p.exists() and cv2.imwrite(str(p),a)
def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip(),'GPU busy';assert structure._HAS_KORNIA and sha(CK)=='1384c38c4f313487a32503025d4566ac1d815f6be983199362fc4af19aaa06a2'
 data=json.loads((O/'sam_generation.json').read_text())['frames'];assert len(data)==100
 for k in ('R','D','E'):(O/'candidates'/k).mkdir()
 (O/'robot_inputs').mkdir();torch.manual_seed(42);model=build_robotseg_video_predictor('configs/robotseg-infer',str(CK),device='cuda',mode='eval',apply_postprocessing=False);torch.cuda.reset_peak_memory_stats();start=time.monotonic();out=[]
 for j,r in enumerate(data):
  i=r['frame'];fr=f'{i:06d}';prior=PRE/('smoke' if i==235 else 'remaining')/'masks'/f'{fr}_robotseg_gripper_NOT_GT.png';im=Path(r['image_path']);assert sha(im)==r['image_sha256']
  if prior.exists():grip=cv2.imread(str(prior),0);origin='Phase16A2R_existing'
  else:
   d=O/'robot_inputs'/fr;d.mkdir();shutil.copyfile(im,d/'000000.jpg');assert sha(d/'000000.jpg')==r['image_sha256']
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=model.init_state(video_path=str(d),async_loading_frames=False,offload_video_to_cpu=False,offload_state_to_cpu=False);_,_,logits=model.add_new_robot(inference_state=state,frame_idx=0,obj_id=0,robot='gripper')
   torch.cuda.synchronize();assert logits.shape==(1,1,720,1280) and bool(torch.isfinite(logits).all());grip=(logits[0,0]>0).cpu().numpy().astype('uint8')*255;origin='RobotSeg_new';del state,logits
  assert grip.shape==(720,1280) and set(np.unique(grip))<={0,255};sam=cv2.imread(str(O/'candidates/B'/f'{fr}.png'),0);inter=cv2.imread(str(O/'candidates/C'/f'{fr}.png'),0);D=((sam>0)&(grip==0)).astype('uint8')*255;E=((inter>0)&(grip==0)).astype('uint8')*255
  for k,a in [('R',grip),('D',D),('E',E)]:save(O/'candidates'/k/f'{fr}.png',a)
  out.append({'frame':i,'robotseg_source':origin,'gripper_pixels':int((grip>0).sum()),'B_D_difference':int(np.count_nonzero(sam!=D)),'B_C_difference':int(np.count_nonzero(sam!=inter)),'C_E_difference':int(np.count_nonzero(inter!=E)),'cad_intersection_removed_pixels':int(((sam>0)&(inter==0)).sum()),'D_E_difference':int(np.count_nonzero(D!=E)),'warning':'pixel differences only, not GT accuracy'})
  if (j+1)%20==0:print('RobotSeg frames',j+1,flush=True)
 info={'frames':out,'n':len(out),'reused':sum(x['robotseg_source']=='Phase16A2R_existing' for x in out),'new_inference':sum(x['robotseg_source']=='RobotSeg_new' for x in out),'checkpoint_sha256':sha(CK),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'seconds':round(time.monotonic()-start,2),'model':'full image, automatic gripper prompt; raw logits > 0, postprocessing disabled','warning':'unverified automatic mask; no GT promotion'};(O/'robot_generation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print('RobotSeg done',info['reused'],info['new_inference'],info['seconds'],flush=True)
if __name__=='__main__':main()
