import sys,json,time,hashlib
from pathlib import Path
import numpy as np,cv2,torch,cupy
P=Path('/home/oylee/projects/HCCEPose_experiments');O=P/'bb100_liteflownet_phase16a2f_20261110';sys.path.insert(0,str(O/'pytorch-liteflownet'));sys.argv=['run.py'];torch.hub.load_state_dict_from_url=lambda **kw:torch.load(O/'network-default.pytorch',map_location='cpu',weights_only=True)
import run
assert not __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
net=run.Network().cuda().eval();print('checkpoint model strict load PASS',flush=True)
def estimate(im1,im2):
 h,w=im1.shape[:2];height=((h+31)//32)*32;width=((w+31)//32)*32
 a=torch.from_numpy(im1.transpose(2,0,1).copy()).float().cuda().unsqueeze(0)/255;b=torch.from_numpy(im2.transpose(2,0,1).copy()).float().cuda().unsqueeze(0)/255
 a=torch.nn.functional.interpolate(a,size=(height,width),mode='bilinear',align_corners=False);b=torch.nn.functional.interpolate(b,size=(height,width),mode='bilinear',align_corners=False)
 with torch.inference_mode(),cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
  f=net(a,b);f=torch.nn.functional.interpolate(f,size=(h,w),mode='bilinear',align_corners=False);f[:,0]*=w/width;f[:,1]*=h/height
 torch.cuda.synchronize();return f[0].permute(1,2,0).cpu().numpy()
import csv
with (P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f:r={int(x['frame']):x for x in csv.DictReader(f)}
a=cv2.imread(r[235]['image_path']);b=cv2.imread(r[237]['image_path']);assert a is not None and b is not None
start=time.monotonic();torch.cuda.reset_peak_memory_stats();z=estimate(a,b);assert z.shape==(720,1280,2) and np.isfinite(z).all();out={'status':'PASS','frame_pair':[235,237],'input_bgr':True,'flow_shape':list(z.shape),'magnitude_median_px':float(np.median(np.linalg.norm(z,axis=2))),'seconds':time.monotonic()-start,'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved(),'weights_sha256':hashlib.sha256((O/'network-default.pytorch').read_bytes()).hexdigest(),'smoke_only':True,'cpu_renderer':'not used','note':'source estimate example-size assertions bypassed in external wrapper, rest of resize protocol reproduced'}
(O/'smoke_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
