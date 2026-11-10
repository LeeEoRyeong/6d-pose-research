from pathlib import Path
import json,sys,importlib.util,hashlib
import numpy as np, cv2, torch
import torchvision.ops.misc as misc
if not hasattr(misc,'ConvNormActivation'):misc.ConvNormActivation=misc.Conv2dNormActivation
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'bb_weak_supervision_20261108'))
import pipeline
from prepare import InspectionDataset
ckp=P/'itodd_official_audit_20261108/itodd_obj04_1_0000step50000.pt'
h=hashlib.sha256()
with ckp.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='224c0869e20be8c9825b77925837e98f0e4c31ae1006361cddd1b46e0cb48ffc'
with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(ckp,map_location='cpu',weights_only=True)
spec=importlib.util.spec_from_file_location('phase14_original_network','/home/oylee/projects/HCCEPose/HccePose/network_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
info=json.loads((P/'itodd_official_audit_20261108/official_models_info.json').read_text())['4']
net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=torch.tensor([info['min_'+a] for a in 'xyz']),size_xyz=torch.tensor([info['size_'+a] for a in 'xyz']))
r=net.load_state_dict(ck['model_state_dict'],strict=True);assert not r.missing_keys and not r.unexpected_keys
with (O/'checkpoint_check.json').open('x') as f:json.dump({'sha256':h.hexdigest(),'strict_load':'PASS','keys':len(ck['model_state_dict']),'device':'cpu'},f,indent=2)
del ck,net
import csv
with (O/'bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
d=InspectionDataset(rows); byid={int(r['frame']):i for i,r in enumerate(rows)}
enc,crop,pad=pipeline.functions();test=[]; (O/'target_previews').mkdir()
for frame in [43,235]:
 a=d[byid[frame]];cache=P/f'bb_weak_supervision_20261108/runs/smoke_20261108T134619_ebd13163/{frame:06d}_CAD_PSEUDO_NOT_VISIBLE_GT.npz';z=np.load(cache)
 assert np.array_equal(a['K'],z['K']);cad=z['cad_amodal']>0;assert cad.shape==a['image'].shape[:2] and cad.any()
 assert all(not np.any(z[k]) for k in ['u_front','u_back','q_pos','q_neg'])
 box=pad(z['bbox'],1.5);cc=crop(z['cad_amodal'],box,128,cv2.INTER_NEAREST);rgb=crop(a['image'],box,256,cv2.INTER_LINEAR);xyz=crop(z['front_xyz'],box,128,cv2.INTER_NEAREST);code=enc(None,xyz)
 assert rgb.shape==(256,256,3) and cc.shape==(128,128) and code.shape==(128,128,24)
 # visualization is CAD-derived potential target, never valid visible GT
 overlay=a['image'].copy();overlay[cad]=(0.55*overlay[cad]+0.45*np.array([0,0,255])).astype('uint8')
 panel=np.hstack([cv2.resize(overlay,(640,360)),cv2.resize(z['front_xyz'],(640,360)),cv2.resize(z['back_xyz'],(640,360))]);
 cv2.putText(panel,'CAD AMODAL / NOT VISIBLE GT',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
 cv2.putText(panel,'FRONT XYZ CAD',(650,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);cv2.putText(panel,'BACK XYZ CAD',(1290,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
 assert cv2.imwrite(str(O/'target_previews'/f'{frame:06d}_CAD_NOT_VISIBLE_GT.jpg'),panel)
 test.append({'frame':frame,'support_pixels':int(cad.sum()),'nonzero_gates':0,'rgb_crop':list(rgb.shape),'encoded_front':list(code.shape)})
with (O/'target_checks.json').open('x') as f:json.dump({'tests':test,'source':'existing CPU-rendered CAD caches','training_target':'NOT VERIFIED; not visible GT'},f,indent=2)
print('strict load PASS; two cached CPU rendering crops and HCCE encoding PASS')
