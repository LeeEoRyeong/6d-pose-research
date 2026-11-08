"""PBR train만으로 4 step 기능 검증. 학습 성능·실사 추론 검증 아님."""
import csv,json,os,time,traceback
import numpy as np
import cv2
import torch
from common import ROOT,SCENE,load_json,write_json,original_function
from probe_env import load_network,build_network,gpu_busy

def main():
    assert load_json(ROOT/'checks/environment.json')['cuda_test']=='PASS'
    assert load_json(ROOT/'checks/labels_v2_100.json')['status']=='PASS'
    busy=gpu_busy()
    if busy:
        write_json(ROOT/'checks/smoke_skip.json',{'status':'SKIP','reason':'다른 GPU compute 작업 발견','processes':busy});return
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);cv2.setNumThreads(1)
    data=load_json(ROOT/'data/dataset.json');ids=data['split']['train'][:4]
    assert not set(ids)&set(data['split']['val'])
    gtinfo=load_json(SCENE/'scene_gt_info.json')
    enc=original_function('HccePose/bop_loader.py','train_bop_dataset_back_front.hcce_encode',{'np':np})
    pad=original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np})
    crop=original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2})
    inputs=[]
    for i in ids:
        box=pad(np.array(gtinfo[str(i)][0]['bbox_visib'],dtype=float),1.5)
        rgb=crop(cv2.imread(str(SCENE/'rgb'/f'{i:06d}.jpg')),box,256,cv2.INTER_LINEAR)
        mask=crop(cv2.imread(str(SCENE/'mask_visib'/f'{i:06d}_000000.png'),0),box,128,cv2.INTER_NEAREST)
        front=crop(cv2.imread(str(ROOT/'labels/front'/f'{i:06d}_000000.png')),box,128,cv2.INTER_NEAREST)
        back=crop(cv2.imread(str(ROOT/'labels/back'/f'{i:06d}_000000.png')),box,128,cv2.INTER_NEAREST)
        x=torch.tensor(np.ascontiguousarray(rgb.transpose(2,0,1)),dtype=torch.float32)/255
        mean=torch.tensor([.406,.456,.485])[:,None,None];std=torch.tensor([.225,.224,.229])[:,None,None]
        inputs.append(((x-mean)/std,torch.tensor(mask/255,dtype=torch.float32),torch.tensor(enc(None,front).transpose(2,0,1),dtype=torch.float32),torch.tensor(enc(None,back).transpose(2,0,1),dtype=torch.float32)))
    info=load_json(ROOT/'data/models/models_info.json')['4'];minimum=torch.tensor([info['min_x'],info['min_y'],info['min_z']],device='cuda');size=torch.tensor([info['size_x'],info['size_y'],info['size_z']],device='cuda')
    module,compat=load_network()
    class ScalarCompatibleNumpy:
        def __getattr__(self,name):return getattr(np,name)
        def mean(self,*args,**kwargs):
            value=np.mean(*args,**kwargs)
            return value.item() if isinstance(value,np.generic) else value
    # 원본 파일을 바꾸지 않고 이 모듈의 scalar 평균 반환형만 정규화한다.
    module.np=ScalarCompatibleNumpy()
    compat.append('Loss 이동평균 np scalar를 Python scalar로 변환; 수식·가중치·원본 소스 불변')
    net=build_network(module,minimum,size).cuda().train();criterion=module.HccePose_Loss()
    optimizer=torch.optim.Adam(net.parameters(),lr=.0002);scaler=torch.amp.GradScaler('cuda')
    rows=[];param=next(net.parameters());torch.cuda.reset_peak_memory_stats()
    log=ROOT/'logs/smoke_steps_compat.jsonl'
    with log.open('x',encoding='utf8') as stream:
        for step in range(4):
            busy=gpu_busy(os.getpid())
            if busy:raise RuntimeError('다른 GPU 작업이 시작되어 즉시 중단: '+str(busy))
            pair=[(step*2)%4,(step*2+1)%4];batch=[torch.stack([inputs[k][col] for k in pair]).cuda() for col in range(4)]
            x,m,f,b=batch;optimizer.zero_grad(set_to_none=True);before=param.detach().clone();start=time.monotonic()
            with torch.amp.autocast('cuda',dtype=torch.float16):
                pm,pc=net(x);losses=criterion(pc[:,:24],pc[:,24:],pm,f,b,m)
                total=3*losses['Front_L1Losses'].sum()+3*losses['Back_L1Losses'].sum()+losses['mask_loss']
            assert torch.isfinite(total),'non-finite loss'
            scaler.scale(total).backward();scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'))
            assert torch.isfinite(grad) and grad>0,'non-finite/zero gradient'
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            delta=float((param.detach()-before).abs().max().item());assert delta>0 and scaler.get_scale()>=old_scale,'optimizer step가 overflow로 생략됨'
            row={'step':step+1,'frame_ids':[ids[k] for k in pair],'batch':2,'loss_total':float(total.item()),'loss_front_sum':float(losses['Front_L1Losses'].sum().item()),'loss_back_sum':float(losses['Back_L1Losses'].sum().item()),'loss_mask':float(losses['mask_loss'].item()),'gradient_norm':float(grad.item()),'max_first_parameter_delta':delta,'amp_scale':scaler.get_scale(),'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'mask_shape':list(pm.shape),'code_shape':list(pc.shape)}
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    path=ROOT/'checkpoints/smoke_compat_step4.pt';assert not path.exists()
    state={k:v.detach().cpu() for k,v in net.state_dict().items()}
    checkpoint={'model_state_dict':state,'iteration_step':4,'object_id':4,'training_frames':ids,'scope':'4 optimizer steps only, random initialization, no optimizer state','seed':0}
    torch.save(checkpoint,path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    restored=build_network(module,minimum.cpu(),size.cpu());result=restored.load_state_dict(loaded['model_state_dict'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items())
    report={'status':'PASS','optimizer_steps_completed':len(rows),'training_frame_ids':ids,'validation_frames_used_for_optimization':0,'real_frames_used_for_optimization':0,'batch_size':2,'crop':256,'backbone':'original ResNet-34 OS8 + ASPP','initialization':'random; pretrained download disabled only during constructor via local patch','original_network_file':str(module.__file__),'compatibility_shims':compat,'amp':'float16 with GradScaler','augmentation':'disabled for functional smoke; deterministic padded crop','checkpoint':str(path),'checkpoint_bytes':path.stat().st_size,'checkpoint_strict_reload':'PASS; all state tensors exactly equal','optimizer_state_saved':False,'parameter_count':sum(p.numel() for p in net.parameters()),'rows':rows,'warning':'학습 완료·Pose 성능·실사 추론 성능을 의미하지 않는 기능 검증'}
    write_json(ROOT/'checks/smoke_result_compat.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(ROOT/'checks/smoke_failure_compat.json',{'status':'FAIL','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
