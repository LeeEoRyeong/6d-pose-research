import sys,json,torch
from pathlib import Path
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
O=Path('/home/oylee/projects/HCCEPose_experiments/bb100_robotseg_phase16a2r_resume_20261110');sys.path.insert(0,str(O/'robotseg_source'))
import robotseg
from robotseg.modeling import memory_attention_with_structure as structure
assert structure._HAS_KORNIA, 'Kornia unavailable; would silently use fallback'
ck=torch.load('/home/oylee/projects/HCCEPose/robotseg.pt',map_location='cpu',weights_only=True)['model'];cfg=compose(config_name='configs/robotseg-infer',overrides=['++model._target_=robotseg.robotseg_video_predictor.RobotSegVideoPredictor']);OmegaConf.resolve(cfg);net=instantiate(cfg.model,_recursive_=True);expected=net.state_dict();missing=sorted(set(expected)-set(ck));extra=sorted(set(ck)-set(expected));mismatch=[{'key':k,'model':list(expected[k].shape),'checkpoint':list(ck[k].shape)} for k in sorted(set(expected)&set(ck)) if expected[k].shape!=ck[k].shape];out={'model_keys':len(expected),'checkpoint_keys':len(ck),'missing':missing,'extra':extra,'shape_mismatch':mismatch,'kornia':structure._HAS_KORNIA,'strict_load':'NOT_RUN','device':'cpu'}
if not (missing or extra or mismatch):
 net.load_state_dict(ck,strict=True);out['strict_load']='PASS'
(O/'strict_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v if k not in ('missing','extra','shape_mismatch') else len(v) for k,v in out.items()},indent=2));assert out['strict_load']=='PASS'
