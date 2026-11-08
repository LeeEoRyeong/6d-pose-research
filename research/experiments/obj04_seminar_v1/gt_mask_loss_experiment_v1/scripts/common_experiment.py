from pathlib import Path
import sys,time,os,subprocess,json,importlib.util,cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;OLD=BASE/'runs/baseline_seed42_20261103_214353/trained_only';sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D
DEADLINE=1793717375+75*60

def guard():
 if time.time()>DEADLINE:raise TimeoutError('75분 경과: 새 GPU 작업 중지 및 보고서 우선')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load_module():
 spec=importlib.util.spec_from_file_location('gtmask_experiment_network',ROOT/'code/network_gtmask.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load_support(ids,device):
 result=[]
 for kind in ('front','back'):
  a=[]
  for i in ids:
   p=ROOT/'support'/f'{i:06d}_{kind}.png';img=cv2.imread(str(p),0)
   if img is None:raise FileNotFoundError(p)
   a.append(torch.tensor(img>0,dtype=torch.bool))
  result.append(torch.stack(a).to(device))
 return result
def batch(samples,ids,device='cuda'):
 x=D.S.batch(samples,ids,device);return x+load_support(ids,device)
def model(module,device):
 return D.S.build_network(module,torch.tensor(D.LO,dtype=torch.float32,device=device),torch.tensor(D.SIZE,dtype=torch.float32,device=device)).to(device)
