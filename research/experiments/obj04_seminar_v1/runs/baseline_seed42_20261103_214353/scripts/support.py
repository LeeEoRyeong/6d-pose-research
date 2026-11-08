import datetime,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
BASE=RUN.parents[1]
sys.path.insert(0,str(BASE/'scripts'))
from common import SCENE,SOURCE,load_json,original_function
from probe_env import build_network
CONFIG=load_json(RUN/'config.json')

def write_json(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def explicit_module():return load_module(RUN/'code/network_model_explicit.py','baseline_explicit_network')

class OtherGpuJob(RuntimeError):pass
class Deadline(RuntimeError):pass
LAST_GPU_CHECK=0.
def guard(force=False):
    global LAST_GPU_CHECK
    now=time.monotonic()
    if not force and now-LAST_GPU_CHECK<1:return
    LAST_GPU_CHECK=now
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    other=[x for x in text.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
    if other:raise OtherGpuJob('다른 GPU compute 작업 감지: '+str(other))
    end=datetime.datetime.fromisoformat(CONFIG['report_priority_utc'])
    if datetime.datetime.now(datetime.timezone.utc)>=end:raise Deadline('75분 경과: 신규 GPU 작업 중단, 보고서 우선')

def dataset():
    import numpy as np,cv2,torch
    cv2.setNumThreads(1)
    d=load_json(BASE/'data/dataset.json');train=d['split']['train'];val=d['split']['val']
    assert len(train)==80 and len(val)==20 and not set(train)&set(val)
    assert not {i//10 for i in train}&{i//10 for i in val}
    gt=load_json(SCENE/'scene_gt.json');cam=load_json(SCENE/'scene_camera.json');gi=load_json(SCENE/'scene_gt_info.json')
    enc=original_function('HccePose/bop_loader.py','train_bop_dataset_back_front.hcce_encode',{'np':np})
    pad=original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np})
    crop=original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2})
    samples={};mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None]
    for i in train+val:
        box=pad(np.array(gi[str(i)][0]['bbox_visib'],dtype=float),1.5)
        rgb=crop(cv2.imread(str(SCENE/'rgb'/f'{i:06d}.jpg')),box,256,cv2.INTER_LINEAR)
        mask=crop(cv2.imread(str(SCENE/'mask_visib'/f'{i:06d}_000000.png'),0),box,128,cv2.INTER_NEAREST)
        ff=crop(cv2.imread(str(BASE/'labels/front'/f'{i:06d}_000000.png')),box,128,cv2.INTER_NEAREST)
        bb=crop(cv2.imread(str(BASE/'labels/back'/f'{i:06d}_000000.png')),box,128,cv2.INTER_NEAREST)
        x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255
        samples[i]={'tensors':[(x-mean)/std,torch.tensor(mask/255,dtype=torch.float32),torch.tensor(enc(None,ff).transpose(2,0,1),dtype=torch.float32),torch.tensor(enc(None,bb).transpose(2,0,1),dtype=torch.float32)],'bbox':torch.tensor(box,dtype=torch.float32),'K':np.array(cam[str(i)]['cam_K']).reshape(3,3),'R':np.array(gt[str(i)][0]['cam_R_m2c']).reshape(3,3),'t':np.array(gt[str(i)][0]['cam_t_m2c']).reshape(3,1),'rgb_crop':rgb}
    return samples,train,val

def batch(samples,ids,device='cuda'):
    import torch
    return [torch.stack([samples[i]['tensors'][k] for i in ids]).to(device) for k in range(4)]

def loss_value(parts):return 3*parts['Front_L1Losses'].sum()+3*parts['Back_L1Losses'].sum()+parts['mask_loss']
