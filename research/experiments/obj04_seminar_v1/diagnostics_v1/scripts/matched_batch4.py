"""기존 Baseline과 동일한 추론 batch4로 수치 민감도를 분리. 기존 진단 출력 보존."""
import inspect
from pathlib import Path
import diagnose as d
ROOT=d.ROOT/'matched_batch4';ROOT.mkdir()
for name in ('checks','figures','tables'): (ROOT/name).mkdir()
d.ROOT=ROOT
src=inspect.getsource(d.predictions)
old="""    for frame in val:
        guard(True);s=samples[frame];x=s['tensors'][0][None].cuda();box=s['bbox'][None].cuda()
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
"""
new="""    cache={}
    for at in range(0,len(val),4):
        guard(True);ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();boxes=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,boxes)
        for j,frame in enumerate(ids):cache[frame]=({k:v[j:j+1].detach().clone() for k,v in out.items()},{k:v[j:j+1].detach().clone() for k,v in raw.items()})
    for frame in val:
        guard(True);s=samples[frame];out,raw=cache[frame]
"""
assert old in src
src=src.replace(old,new)
exec(compile(src,str(Path(__file__).resolve()),'exec'),d.__dict__)
d.predictions();d.summarize()
d.write(ROOT/'checks/execution.json',{'batch_size':4,'same_as_baseline':True,'precision':'CUDA AMP float16','single_image_diagnostic_preserved':str(ROOT.parent)})
