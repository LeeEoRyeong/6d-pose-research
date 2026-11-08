"""Read-only CPU audit of official ITODD obj04 checkpoint and CAD."""
import importlib.util,json,hashlib,sys
from pathlib import Path
import numpy as np,torch
import torchvision.ops.misc as misc
if not hasattr(misc,'ConvNormActivation'):misc.ConvNormActivation=misc.Conv2dNormActivation
ROOT=Path('/RESEARCH_DATA/HCCEPose'); EXP=Path('/RESEARCH_DATA/HCCEPose_experiments');HERE=Path(__file__).resolve().parent
p=HERE/'itodd_obj04_1_0000step50000.pt'; sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(p)=='224c0869e20be8c9825b77925837e98f0e4c31ae1006361cddd1b46e0cb48ffc'
with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):
 ck=torch.load(p,map_location='cpu',weights_only=True)
sd=ck['model_state_dict'];spec=importlib.util.spec_from_file_location('official_local_arch',ROOT/'HccePose/network_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
info=json.load((HERE/'official_models_info.json').open())['4'];local=json.load((EXP/'obj04_seminar_v1/data/models/models_info.json').open())['4']
net=module.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=torch.tensor([info['min_'+a] for a in 'xyz']),size_xyz=torch.tensor([info['size_'+a] for a in 'xyz']))
expected=net.state_dict(); missing=sorted(set(expected)-set(sd)); extra=sorted(set(sd)-set(expected)); mismatch=[{'key':k,'checkpoint':list(sd[k].shape),'network':list(expected[k].shape)} for k in sorted(set(expected)&set(sd)) if sd[k].shape!=expected[k].shape]
result={'checkpoint_sha256':sha(p),'checkpoint_bytes':p.stat().st_size,'keys':list(ck),'iteration_step':int(ck['iteration_step']),'best_score':float(ck['best_score']),'score_meaning':'not independently identified','checkpoint_tensor_count':len(sd),'model_tensor_count':len(expected),'missing_keys':missing,'unexpected_keys':extra,'shape_mismatch':mismatch,'strict_load':'NOT_RUN','model_backbone':'efficientnet_b4','input_channels':3,'source_commit':None,'official_model_sha256':sha(HERE/'official_obj_000004.ply'),'bb_model_sha256':sha(Path('/RESEARCH_DATA/HCCEPose_obj04_transfer/obj_000004.ply')),'baseline_model_sha256':sha(EXP/'obj04_seminar_v1/data/models/obj_000004.ply'),'official_model_info':info,'baseline_model_info':local}
if not (missing or extra or mismatch):
 try:
  ans=net.load_state_dict(sd,strict=True);result['strict_load']='PASS';result['load_result']=str(ans)
 except Exception as e:result['strict_load']='FAIL';result['load_error']=repr(e)
else:result['strict_load']='FAIL_KEY_OR_SHAPE_MISMATCH'
with (HERE/'checkpoint_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('official_model_info','baseline_model_info')},indent=2))
