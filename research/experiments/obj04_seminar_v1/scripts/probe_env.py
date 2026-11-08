import importlib.util,json,os,subprocess,sys,traceback
from pathlib import Path
from common import ROOT,SOURCE,write_json

def gpu_busy(allow_pid=None):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    rows=[x.strip() for x in raw.splitlines() if x.strip()]
    foreign=[x for x in rows if allow_pid is None or int(x.split(',')[0])!=allow_pid]
    return foreign

def load_network():
    import torch,torchvision
    compat=[]
    import torchvision.ops.misc as misc
    if not hasattr(misc,'ConvNormActivation'):
        misc.ConvNormActivation=misc.Conv2dNormActivation
        compat.append('프로세스 내부 ConvNormActivation 별칭만 추가; 기본 ResNet 분기에서 해당 모듈 미사용')
    spec=importlib.util.spec_from_file_location('seminar_original_network',SOURCE/'HccePose/network_model.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,compat

def build_network(mod,minimum,size):
    # 원본 소스는 그대로 import. 생성자 호출 동안만 pretrained 다운로드를 차단.
    import torchvision.models as models
    from unittest.mock import patch
    original=models.resnet34
    def random_resnet(*args,**kwargs):
        kwargs.pop('pretrained',None);kwargs['weights']=None
        return original(*args,**kwargs)
    with patch.object(models,'resnet34',side_effect=random_resnet):
        return mod.HccePose_BF_Net(efficientnet_key=None,input_channels=3,min_xyz=minimum,size_xyz=size)

def main():
    import torch,torchvision,numpy,cv2,PIL
    torch.set_num_threads(2)
    mod,compat=load_network()
    info={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__,'network_import':'PASS','source':str(SOURCE/'HccePose/network_model.py'),'compatibility_shims':compat,'gpu_compute_apps_before':gpu_busy()}
    if info['gpu_compute_apps_before']:
        info['cuda_test']='SKIP: 다른 compute 작업 존재'
    else:
        assert torch.cuda.is_available(),'CUDA unavailable'
        info.update(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),compiled_arches=torch.cuda.get_arch_list())
        x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4)
        y=x@x.T;torch.cuda.synchronize()
        assert torch.allclose(y.cpu(),x.cpu()@x.cpu().T)
        info.update(cuda_test='PASS',cuda_result_sum=float(y.sum().item()),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(ROOT/'checks/environment.json',info);print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(ROOT/'checks/environment_failure.json',{'status':'FAIL','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
