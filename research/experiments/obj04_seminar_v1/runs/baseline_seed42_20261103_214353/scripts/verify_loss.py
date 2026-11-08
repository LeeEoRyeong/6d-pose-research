import json,time,traceback
import numpy as np
import torch
from support import RUN,BASE,load_module,explicit_module,write_json,guard,OtherGpuJob

class Proxy:
    def __getattr__(self,name):return getattr(np,name)
    def mean(self,*args,**kwargs):
        v=np.mean(*args,**kwargs);return v.item() if isinstance(v,np.generic) else v

def total(parts):return 3*parts['Front_L1Losses'].sum()+3*parts['Back_L1Losses'].sum()+parts['mask_loss']

def compare(device,dtype,iterations):
    original=load_module(RUN/'code/network_model_original_snapshot.py','legacy_'+device+str(dtype));original.np=Proxy()
    fixed=explicit_module();a=original.HccePose_Loss();b=fixed.HccePose_Loss()
    torch.manual_seed(42);max_loss=max_grad=max_history=0.
    for step in range(iterations):
        if device=='cuda':guard()
        f=torch.randn(2,24,8,8,device=device,dtype=dtype);back=torch.randn_like(f);m=torch.randn(2,1,8,8,device=device,dtype=dtype)
        if step%7==0:m.fill_(-2)
        if step%11==0:m.fill_(2)
        gf=torch.rand(2,24,8,8,device=device);gb=torch.rand_like(gf);gm=(torch.rand(2,8,8,device=device)>.4).float()
        pa=[v.detach().clone().requires_grad_(True) for v in (f,back,m)];pb=[v.detach().clone().requires_grad_(True) for v in (f,back,m)]
        la=a(*pa,gf,gb,gm);lb=b(*pb,gf,gb,gm)
        max_loss=max(max_loss,max(float((la[k]-lb[k]).abs().max()) for k in la))
        total(la).backward();total(lb).backward()
        max_grad=max(max_grad,max(float((x.grad-y.grad).abs().max()) for x,y in zip(pa,pb)))
        for attr in ('weight_front_error_ratio','weight_back_error_ratio','current_front_error_ratio','current_back_error_ratio'):
            max_history=max(max_history,max(float((x-y).abs().max()) for x,y in zip(getattr(a,attr),getattr(b,attr))))
        for attr in ('Front_error_list','Back_error_list'):
            for x,y in zip(getattr(a,attr),getattr(b,attr)):
                for xx,yy in zip(x,y):assert np.array_equal(np.asarray(xx.data),np.asarray(yy.data))
    assert max_loss==max_grad==max_history==0
    return {'device':device,'prediction_dtype':str(dtype),'iterations':iterations,'history_window':100,'max_loss_difference':max_loss,'max_gradient_difference':max_grad,'max_weight_and_error_difference':max_history,'history_arrays_exactly_equal':True}

def raw_probe():
    raw=load_module(RUN/'code/network_model_original_snapshot.py','raw_original_loss');fixed=explicit_module();a=raw.HccePose_Loss();b=fixed.HccePose_Loss()
    torch.manual_seed(17);args=[torch.randn(2,24,8,8,requires_grad=True),torch.randn(2,24,8,8,requires_grad=True),torch.randn(2,1,8,8,requires_grad=True)];truth=[torch.rand(2,24,8,8),torch.rand(2,24,8,8),(torch.rand(2,8,8)>.5).float()]
    others=[x.detach().clone().requires_grad_(True) for x in args]
    x=a(*args,*truth);y=b(*others,*truth);total(x).backward();total(y).backward()
    assert all(torch.equal(x[k],y[k]) for k in x) and all(torch.equal(p.grad,q.grad) for p,q in zip(args,others))
    error=None
    try:a(*args,*truth)
    except TypeError as exc:error=str(exc)
    return {'first_forward_and_grad_exact':True,'raw_second_forward_error':error,'limitation':'원본은 둘째 이동평균 갱신에서 오류이므로 원본 무변환 120회 직접 비교는 불가능. 기존 proxy 대 명시적 float 복사본을 별도로 비교함'}

def main():
    torch.set_num_threads(2);start=time.monotonic();result={'raw':raw_probe(),'comparisons':[compare('cpu',torch.float32,120)]}
    guard(force=True)
    assert torch.cuda.is_available()
    result['comparisons'].append(compare('cuda',torch.float32,120))
    result['comparisons'].append(compare('cuda',torch.float16,120))
    result.update(status='PASS',seconds=time.monotonic()-start,scope='2곳의 float32/float16 np scalar→Python float 변환만 변경. window100 초과·zero/full/부분 Mask·loss·history·gradient 동일 확인')
    write_json(RUN/'checks/loss_compatibility.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(RUN/'checks/loss_compatibility_failure.json',{'status':'FAIL','message':str(exc),'traceback':traceback.format_exc()});raise
