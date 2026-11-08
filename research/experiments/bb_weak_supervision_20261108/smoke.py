"""CPU-only bounded smoke: 4 BB frames, 2 PBR frames, never writes source data."""
import hashlib,json,sys,traceback,datetime,uuid
from pathlib import Path
import cv2,numpy as np,torch
import torchvision.ops.misc as misc
if not hasattr(misc,'ConvNormActivation'): misc.ConvNormActivation=misc.Conv2dNormActivation
from pipeline import BBAdapter,BASE,BB,SCENE,RUN,groups,load_json,functions,balanced_ids
from common import original_function
from types import SimpleNamespace
from weak_loss import weak_terms,total_loss
from training import mixed_step

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN/'scripts'))
from support import explicit_module,build_network

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def history(c):
    return {'front_weights':[None if x is None else x.detach().cpu().tolist() for x in c.weight_front_error_ratio],
            'back_weights':[None if x is None else x.detach().cpu().tolist() for x in c.weight_back_error_ratio],
            'front_history':[[list(map(float,x.data)) for x in a] for a in c.Front_error_list],
            'back_history':[[list(map(float,x.data)) for x in a] for a in c.Back_error_list]}
def synthetic(module):
    torch.manual_seed(3);direct=module.HccePose_Loss();wrapped=module.HccePose_Loss()
    for iteration in range(3):
        orig=[torch.randn(2,24,6,7),torch.randn(2,24,6,7),torch.randn(2,1,6,7)]
        gt=[torch.randint(0,2,(2,24,6,7)).float(),torch.randint(0,2,(2,24,6,7)).float(),torch.randint(0,2,(2,6,7)).float()]
        a=[x.clone().requires_grad_() for x in orig]; b=[x.clone().requires_grad_() for x in orig]
        out=direct(a[0],a[1],a[2],gt[0],gt[1],gt[2]); old=3*out['Front_L1Losses'].sum()+3*out['Back_L1Losses'].sum()+out['mask_loss'];old.backward()
        z=torch.zeros(1,6,7);rf=torch.randn(1,24,6,7,requires_grad=True);rb=torch.randn(1,24,6,7,requires_grad=True);rm=torch.randn(1,1,6,7,requires_grad=True)
        targets={'front':torch.zeros_like(rf),'back':torch.zeros_like(rb),'u_front':z,'u_back':z,'q_pos':z,'q_neg':z}
        new,pbr,weak=total_loss(wrapped,(b[2],torch.cat([b[0],b[1]],dim=1)),(gt[2],gt[0],gt[1]),(rm,torch.cat([rf,rb],dim=1)),targets)
        new.backward();assert torch.equal(old,pbr) and torch.equal(old,new)
        for x,y in zip(a,b):assert torch.equal(x.grad,y.grad)
        assert history(direct)==history(wrapped);assert all(torch.count_nonzero(v.grad)==0 for v in (rf,rb,rm))
        assert torch.isfinite(new) and weak['xyz'].item()==0 and weak['mask'].item()==0
    rf=torch.randn(1,24,6,7,requires_grad=True);rb=torch.randn(1,24,6,7,requires_grad=True);rm=torch.randn(1,1,6,7,requires_grad=True)
    uf=torch.zeros(1,6,7);ub=uf.clone();qp=uf.clone();qn=uf.clone();uf[0,1,2]=1;ub[0,3,4]=1;qp[0,1,2]=1;qn[0,5,6]=1
    before=history(wrapped);p=weak_terms(rf,rb,rm,torch.zeros_like(rf),torch.zeros_like(rb),uf,ub,qp,qn,wrapped);(p['xyz']+p['mask']).backward();assert before==history(wrapped)
    assert torch.count_nonzero(rf.grad*(1-uf[:,None]))==0 and torch.count_nonzero(rb.grad*(1-ub[:,None]))==0 and torch.count_nonzero(rm.grad*(1-(qp+qn)[:,None]))==0
    assert torch.count_nonzero(rf.grad)>0 and torch.count_nonzero(rb.grad)>0 and torch.count_nonzero(rm.grad)>0
    return {'pbr_value_gradient_history':'3 calls bitwise equal','ignored_gradient':'exactly zero','empty_gates':'finite zero','real_history':'unchanged'}

def pbr(frame,enc,crop,pad,renderer,v,faces,info):
    gt=load_json(SCENE/'scene_gt.json')[str(frame)][0]; K=np.array(load_json(SCENE/'scene_camera.json')[str(frame)]['cam_K']).reshape(3,3)
    gi=load_json(SCENE/'scene_gt_info.json')[str(frame)][0]; R=np.array(gt['cam_R_m2c']).reshape(3,3); t=np.array(gt['cam_t_m2c']);box=pad(np.array(gi['bbox_visib'],float),1.5)
    rgb=cv2.imread(str(SCENE/'rgb'/f'{frame:06d}.jpg')); mask=cv2.imread(str(SCENE/'mask_visib'/f'{frame:06d}_000000.png'),0)
    n,f,valid,cam=renderer.render(v,faces,R,t,K,*mask.shape,.5); yy,xx=np.nonzero(valid)
    lo=np.array([info['min_'+a] for a in 'xyz']);size=np.array([info['size_'+a] for a in 'xyz']);outs=[]
    for folder,z in [('front',n),('back',f)]:
        zz=z[valid];xyz=(np.column_stack(((xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz))-t)@R
        q=np.rint(np.clip((xyz-lo)/size,0,1)*255).astype(np.uint8);a=np.zeros((*mask.shape,3),np.uint8);a[valid]=q
        stored=cv2.imread(str(BASE/'labels'/folder/f'{frame:06d}_000000.png'));assert np.array_equal(a,stored)
        out=crop(stored,box,128,cv2.INTER_NEAREST);outs.append(torch.tensor(enc(None,out).transpose(2,0,1),dtype=torch.float32))
    xc=crop(rgb,box,256,cv2.INTER_LINEAR);x=torch.tensor(np.ascontiguousarray(xc.transpose(2,0,1)),dtype=torch.float32)/255
    x=(x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
    mc=crop(mask,box,128,cv2.INTER_NEAREST);return (x,torch.tensor(mc/255,dtype=torch.float32),*outs)

def main(folder):
    cv2.setNumThreads(1);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    frameids=[43,75,235,247];adapter=BBAdapter(frameids);enc,crop,pad=functions();module=explicit_module()
    watched=[BB/'BB_obj04/scene_gt_obj04.json',BB/'BB_obj04/scene_camera_obj04.json',BASE/'data/models/obj_000004.ply',RUN/'checkpoints/best_val_loss.pt',RUN/'checks/split.json',RUN/'code/network_model_explicit.py']
    watched += [adapter.rgb_paths[i] for i in frameids]
    watched += [BB/'BB_obj04/poses_obj04'/f'pose_{i:06d}.npy' for i in frameids]
    for i in (0,10): watched += [SCENE/'rgb'/f'{i:06d}.jpg',SCENE/'mask_visib'/f'{i:06d}_000000.png',BASE/'labels/front'/f'{i:06d}_000000.png',BASE/'labels/back'/f'{i:06d}_000000.png']
    before={str(p):sha(p) for p in watched}; result={'status':'RUNNING','gpu_used':False,'original_input_sha256_before':before}; real=[]
    result['synthetic']=synthetic(module)
    decode=original_function('HccePose/network_model.py','HccePose_BF_Net.hcce_decode',{'torch':torch})
    raw=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    decoded=decode(SimpleNamespace(powers=None),torch.tensor(enc(None,raw)[None]>.5,dtype=torch.float32))[0].numpy()
    assert np.array_equal(decoded,raw);result['hcce_roundtrip']='all 256 byte values x/y/z exact';print('synthetic and HCCE roundtrip PASS',flush=True)
    examples={}
    for i in frameids:
        s=adapter.load(i);t=adapter.tensors(s);ys,xs=np.nonzero(s['cad']);idx=np.linspace(0,len(xs)-1,5,dtype=int)
        hits=adapter.renderer.independent_rays(s['camera_vertices'],adapter.faces,s['K'],list(zip(ys[idx],xs[idx])),.5)
        err=max(max(abs(s['near'][y,x]-a),abs(s['far'][y,x]-b)) for y,x,a,b in hits);assert err<1e-5
        assert s['rgb'].shape==(720,1280,3) and t['rgb'].shape==(3,256,256) and t['front'].shape==(24,128,128)
        posefile=np.load(BB/'BB_obj04/poses_obj04'/f'pose_{i:06d}.npy',allow_pickle=False)
        assert np.allclose(posefile[:3,:3],s['R'],atol=1e-9) and np.allclose(posefile[:3,3],s['t'],atol=1e-9)
        assert s['bbox'].tolist()==list(map(float,cv2.boundingRect(s['cad'].astype('uint8'))))
        assert np.allclose(s['box'],pad(s['bbox'],1.5));assert np.isfinite(s['K']).all() and np.isfinite(s['R']).all() and np.isfinite(s['t']).all()
        for name in ('u_front','u_back','q_pos','q_neg'):assert not t[name].any()
        if i==43:
            prior=Path('/RESEARCH_DATA/HCCEPose/bb_real_preflight_20261108')
            assert np.array_equal(s['front'],cv2.imread(str(prior/'43_front_pseudo.png')))
            assert np.array_equal(s['back'],cv2.imread(str(prior/'43_back_pseudo.png')))
            assert np.array_equal(s['cad'].astype('uint8')*255,cv2.imread(str(prior/'43_rendered_full_pseudo_mask.png'),0))
        # A single crop must preserve pixel positions across RGB-like labels and gates.
        marker=s['cad'].astype('uint8')*255; c1=crop(marker,s['box'],128,cv2.INTER_NEAREST);c2=crop(np.repeat(marker[:,:,None],3,axis=2),s['box'],128,cv2.INTER_NEAREST)
        assert np.array_equal(c1,c2[:,:,0]);assert np.isfinite(t['front'].numpy()).all() and np.isfinite(t['back'].numpy()).all()
        row={'frame':i,'state':adapter.rgb_paths[i].parent.name,'group':s['group_id'],'reference':s['reference_id'],'bbox_xywh':s['bbox'].tolist(),'cad_pixels':int(s['cad'].sum()),'valid_u_front_pixels':int(s['gates']['u_front'].sum()),'valid_u_back_pixels':int(s['gates']['u_back'].sum()),'valid_fraction_cad':0.0,'q_pos':0,'q_neg':0,'reason':s['gates']['reason'],'temporal_difference':s['gates']['temporal_difference'],'independent_ray_max_error':float(err)}
        real.append(row);examples[i]=(s,t)
        with (folder/f'{i:06d}_CAD_PSEUDO_NOT_VISIBLE_GT.npz').open('xb') as f:
            np.savez_compressed(f,cad_amodal=s['cad'].astype(np.uint8),front_xyz=s['front'],back_xyz=s['back'],
                u_front=s['gates']['u_front'].astype(np.uint8),u_back=s['gates']['u_back'].astype(np.uint8),
                q_pos=s['gates']['q_pos'].astype(np.uint8),q_neg=s['gates']['q_neg'].astype(np.uint8),K=s['K'],bbox=s['bbox'])
        print('BB',i,'PASS, weak supervision 0',flush=True)
    result['bb_samples']=real
    result['same_pose_group_count']=len(adapter.group_map)
    eligible={g:[i for i in v if i in frameids] for g,v in adapter.group_map.items()};eligible={g:v for g,v in eligible.items() if v}
    picked=balanced_ids(eligible,12,42);assert set(picked)<=set(frameids);result['group_sampling_smoke']=picked
    assert {0,10} <= set(load_json(RUN/'checks/split.json')['train_ids'])
    for i in (0,10):
        examples['pbr'+str(i)]=pbr(i,enc,crop,pad,adapter.renderer,adapter.v,adapter.faces,adapter.info);print('PBR',i,'bitwise PASS',flush=True)
    # Build the actual baseline network without downloading pretrained weights; CPU only.
    lo=torch.tensor([adapter.info['min_'+a] for a in 'xyz']);size=torch.tensor([adapter.info['size_'+a] for a in 'xyz']);torch.manual_seed(42)
    net=build_network(module,lo,size).cpu().train(); criterion=module.HccePose_Loss();opt=torch.optim.Adam(net.parameters(),lr=.0002)
    x0=examples['pbr0'];x1=examples['pbr10']; pb=tuple(torch.stack([x0[j],x1[j],x0[j]]) for j in range(4));rb={k:v.unsqueeze(0) for k,v in examples[43][1].items()}
    try:mixed_step(net,opt,criterion,pb,rb)
    except RuntimeError as e:assert 'NO-GO' in str(e)
    else:raise AssertionError('empty real gates must block production')
    result['cpu_one_step']=mixed_step(net,opt,criterion,pb,rb,allow_empty_for_smoke=True)
    assert result['cpu_one_step']['real_xyz']==0 and result['cpu_one_step']['real_mask']==0 and not torch.cuda.is_initialized()
    after={str(p):sha(p) for p in watched};assert after==before;result['source_integrity']={'files_checked':len(before),'all_sha256_unchanged':True};result['status']='PASS_SMOKE_REAL_SUPERVISION_NO_GO';result['bb_trainable_count_with_reliable_gates']=0;result['bb_frames_evaluated']=4;result['bb100_trainable_count']='not measured';result['notes']='temporal difference is diagnostic only, no validated occlusion model, all real gates empty';return result

if __name__=='__main__':
    folder=HERE/'runs'/('smoke_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    try:r=main(folder)
    except Exception as e:r={'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()};print(r['traceback'],flush=True)
    with (folder/'results.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('RESULT',folder/'results.json',r['status'],flush=True)
    if r['status']=='FAIL':sys.exit(1)
