"""BB weak-supervision data adapter; default uncertainty policy is fail-closed."""
from pathlib import Path
import importlib.util, json, sys, collections
import cv2
import numpy as np
import torch

EXP=Path('/RESEARCH_DATA/HCCEPose_experiments')
BASE=EXP/'obj04_seminar_v1'
BB=EXP/'real_bb100_20261108'
SCENE=EXP/'realistic_final100/train_pbr/000000'
RUN=BASE/'runs/baseline_seed42_20261103_214353/trained_only'
sys.path.insert(0,str(BASE/'scripts'))
from common import original_function

def load_json(path): return json.loads(Path(path).read_text())
def load_renderer():
    spec=importlib.util.spec_from_file_location('weak_cpu_renderer',BASE/'scripts/generate_xyz_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def functions():
    enc=original_function('HccePose/bop_loader.py','train_bop_dataset_back_front.hcce_encode',{'np':np})
    crop=original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2})
    pad=original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np})
    return enc,crop,pad

def pose(item): return np.array(item[0]['cam_R_m2c']).reshape(3,3),np.array(item[0]['cam_t_m2c'])

def groups(gt):
    out=collections.defaultdict(list)
    for key,item in gt.items():
        out[tuple(item[0]['cam_R_m2c']+item[0]['cam_t_m2c'])].append(int(key))
    return {min(ids):sorted(ids) for ids in out.values()}

def balanced_ids(group_map, steps, seed=42):
    """Sample groups uniformly, then members uniformly; inputs must already be split."""
    if not group_map or any(not values for values in group_map.values()): raise ValueError('empty real group')
    rng=np.random.default_rng(seed); keys=sorted(group_map)
    result=[]
    while len(result)<steps:
        for k in rng.permutation(keys):
            result.append(int(rng.choice(group_map[int(k)])))
            if len(result)==steps:break
    return result

class UnknownOcclusionEstimator:
    """No validated real occlusion model: no foreground or background supervision."""
    name='unavailable_fail_closed'
    def __call__(self, rgb, reference, support):
        diagnostic=None
        if reference is not None:
            delta=np.max(np.abs(rgb.astype(np.int16)-reference.astype(np.int16)),axis=2)
            diagnostic={'mean_abs_rgb':float(np.abs(rgb.astype(np.int16)-reference.astype(np.int16))[support].mean()),
                        'fraction_delta_gt_30_diagnostic_only':float((delta[support]>30).mean())}
        zeros=np.zeros(support.shape,dtype=bool)
        return {'u_front':zeros.copy(),'u_back':zeros.copy(),'q_pos':zeros.copy(),'q_neg':zeros.copy(),
                'estimator':self.name,'reason':'no validated automatic occlusion/confidence model',
                'temporal_difference':diagnostic}

class BBAdapter:
    def __init__(self, allowed_reference_ids, max_frame_gap=20):
        self.gt=load_json(BB/'BB_obj04/scene_gt_obj04.json');self.cam=load_json(BB/'BB_obj04/scene_camera_obj04.json')
        self.allowed=set(map(int,allowed_reference_ids));self.gap=max_frame_gap;self.renderer=load_renderer()
        self.v,self.faces=self.renderer.mesh(); self.info=load_json(BASE/'data/models/models_info.json')['4']
        self.rgb_paths={}
        for p in (BB/'rgb_undistorted').rglob('*.jpg'):
            i=int(p.stem)
            if str(i) in self.gt:
                if i in self.rgb_paths:raise ValueError('duplicate RGB frame')
                self.rgb_paths[i]=p
        self.group_map=groups(self.gt);self.encoder,self.crop,self.pad=functions();self.estimator=UnknownOcclusionEstimator()
    def load(self,frame):
        frame=int(frame)
        if frame not in self.allowed:raise ValueError('frame outside allowed split')
        R,t=pose(self.gt[str(frame)]);K=np.array(self.cam[str(frame)]['cam_K']).reshape(3,3)
        rgb=cv2.imread(str(self.rgb_paths[frame]));assert rgb is not None
        H,W=rgb.shape[:2];near,far,support,cam_vertices=self.renderer.render(self.v,self.faces,R,t,K,H,W,.5)
        if not support.any() or not np.isfinite(near[support]).all() or not np.isfinite(far[support]).all():raise ValueError('invalid renderer support')
        if not (near[support]<=far[support]).all():raise ValueError('depth order')
        group_id=next(g for g,ids in self.group_map.items() if frame in ids)
        choices=[i for i in self.group_map[group_id] if i in self.allowed and i!=frame and abs(i-frame)<=self.gap]
        refid=min(choices,key=lambda i:(abs(i-frame),i)) if choices else None
        ref=cv2.imread(str(self.rgb_paths[refid])) if refid is not None else None
        gates=self.estimator(rgb,ref,support)
        yy,xx=np.nonzero(support); lo=np.array([self.info['min_'+a] for a in 'xyz']);size=np.array([self.info['size_'+a] for a in 'xyz'])
        labels=[]
        for z in (near,far):
            depth=z[support];xyz=(np.column_stack(((xx-K[0,2])/K[0,0]*depth,(yy-K[1,2])/K[1,1]*depth,depth))-t)@R
            q=np.rint(np.clip((xyz-lo)/size,0,1)*255).astype(np.uint8); label=np.zeros((H,W,3),np.uint8);label[support]=q;labels.append(label)
        bbox=np.array(cv2.boundingRect(support.astype('uint8')),dtype=float);box=self.pad(bbox,1.5)
        return {'id':frame,'rgb':rgb,'K':K,'R':R,'t':t,'cad':support,'front':labels[0],'back':labels[1],
                'near':near,'far':far,'camera_vertices':cam_vertices,'bbox':bbox,'box':box,'gates':gates,'group_id':group_id,'reference_id':refid}
    def tensors(self,s):
        rgb=self.crop(s['rgb'],s['box'],256,cv2.INTER_LINEAR)
        x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255
        x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
        out={'rgb':x}
        for name in ('front','back'):
            crop=self.crop(s[name],s['box'],128,cv2.INTER_NEAREST)
            out[name]=torch.tensor(self.encoder(None,crop).transpose(2,0,1),dtype=torch.float32)
        for name in ('u_front','u_back','q_pos','q_neg'):
            crop=self.crop(s['gates'][name].astype('uint8'),s['box'],128,cv2.INTER_NEAREST)
            out[name]=torch.tensor(crop,dtype=torch.float32)
        return out
