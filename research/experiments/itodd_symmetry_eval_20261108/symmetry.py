"""Symmetry-aware metrics for BOP model-to-camera poses, using official transforms."""
import json
from pathlib import Path
import numpy as np

AUDIT=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_official_audit_20261108')

def load():
    lines=(AUDIT/'official_obj_000004.ply').read_text().splitlines();start=lines.index('end_header')+1
    n=int(next(x.split()[-1] for x in lines[:start] if x.startswith('element vertex ')));nf=int(next(x.split()[-1] for x in lines[:start] if x.startswith('element face ')))
    v=np.array([list(map(float,x.split()[:3])) for x in lines[start:start+n]],dtype=np.float64)
    faces=np.array([list(map(int,x.split()[1:])) for x in lines[start+n:start+n+nf]],dtype=np.int32)
    info=json.loads((AUDIT/'official_models_info.json').read_text())['4']
    syms=[np.eye(4)]+[np.array(s,dtype=np.float64).reshape(4,4) for s in info['symmetries_discrete']]
    return v,faces,syms,info

def transform(v,R,t):return [REDACTED_EMAIL](R).reshape(3,3).T+np.asarray(t).reshape(1,3)
def compose(R,t,S):return R@S[:3,:3],R@S[:3,3]+t.reshape(3)
def add(v,Rp,tp,Rg,tg):return float(np.mean(np.linalg.norm(transform(v,Rp,tp)-transform(v,Rg,tg),axis=1)))
def add_s(v,Rp,tp,Rg,tg):
    pred=transform(v,Rp,tp);gt=transform(v,Rg,tg);parts=[]
    for p in np.array_split(pred,32):parts.extend(np.sqrt(np.min(np.sum((p[:,None,:]-gt[None,:,:])**2,axis=2),axis=1)).tolist())
    return float(np.mean(parts))
def symmetry_add(v,Rp,tp,Rg,tg,syms):
    vals=[add(v,Rp,tp,*compose(Rg,tg,s)) for s in syms];return float(min(vals)),int(np.argmin(vals))
def validate():
    v,faces,syms,info=load();assert len(syms)==2
    S=syms[1];assert np.allclose(S[3],[0,0,0,1]) and np.linalg.det(S[:3,:3])>0.999999 and np.allclose(S[:3,:3].T@S[:3,:3],np.eye(3),atol=1e-10)
    angle=.37;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    t=np.array([22.,-31.,560.]);Rs,ts=compose(R,t,S)
    a=transform(v,Rs,ts);b=transform(transform(v,S[:3,:3],S[:3,3]),R,t)
    composition=float(np.max(np.abs(a-b)))
    same={'ADD':add(v,R,t,R,t),'ADD_S':add_s(v,R,t,R,t),'explicit':symmetry_add(v,R,t,R,t,syms)[0]}
    equiv={'ADD':add(v,Rs,ts,R,t),'ADD_S':add_s(v,Rs,ts,R,t),'explicit':symmetry_add(v,Rs,ts,R,t,syms)[0]}
    threshold=.1*info['diameter'];assert composition<1e-9 and max(same.values())<1e-9 and equiv['explicit']<1e-9 and equiv['ADD_S']<threshold and equiv['ADD']>1
    wrong_t=t; omission_error=float(np.linalg.norm(ts-wrong_t));assert omission_error>0.14
    return {'status':'PASS','composition_max_abs':composition,'same_pose_errors':same,'symmetry_pose_errors':equiv,'threshold_0_1d':threshold,'translation_omission_camera_units':omission_error,'units':'BOP CAD unit mm; real pose physical scale assumed, not independently measured','transform_order':'T_m2c_equivalent = T_m2c @ S_m2m, R2=R@Rs, t2=R@ts+t','explicit_metric':'min over identity and official discrete transforms of mean corresponding-vertex 3D distance','adds_direction':'each predicted vertex to nearest GT vertex; mean','warning':'explicit symmetry produces zero by definition even for approximate mesh symmetry; ADD-S can be nonzero due to vertex sampling'}
