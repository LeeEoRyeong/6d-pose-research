from pathlib import Path
import sys,json,hashlib,csv,collections
import numpy as np,cv2
from targets import DiagnosticTargets,P
O=Path(__file__).resolve().parent
# Reuse actual unmodified CPU projection and original HCCE crop/encoding code.
a=DiagnosticTargets([43,235]);checks=[]
for i,frame in enumerate([43,235]):
 s=a[i];t=s['diagnostic_tensors'];assert s['frame']==frame and s['cad_amodal'].shape==(720,1280)
 assert t['rgb'].shape==(3,256,256) and t['front'].shape==t['back'].shape==(24,128,128)
 assert all(t[k].sum().item()==0 for k in ('u_front','u_back','q_pos','q_neg'))
 for v in t.values():assert bool(v.isfinite().all())
 assert s['cad_amodal'].sum()>0 and s['front_uncanonicalized'].dtype==np.uint8
 try:a.training_sample(i);raise AssertionError('unprotected training path')
 except RuntimeError:pass
 checks.append({'frame':frame,'cad_pixels':int(s['cad_amodal'].sum()),'supervision_gate_pixels':0,'rgb_crop':[3,256,256],'front_back_code':[24,128,128],'training_guard':'PASS'})
# Re-render two PBR inputs from CAD, compare to existing uncanonicalized PBR PNGs.
ren=a.adapter.renderer;v,faces=ren.mesh();scene=P/'realistic_final100/train_pbr/000000';gt=json.loads((scene/'scene_gt.json').read_text());camera=json.loads((scene/'scene_camera.json').read_text());info=a.adapter.info;lo=np.array([info['min_'+x] for x in 'xyz']);size=np.array([info['size_'+x] for x in 'xyz']);pbr=[]
for frame in (0,10):
 item=gt[str(frame)][0];R=np.array(item['cam_R_m2c']).reshape(3,3);t=np.array(item['cam_t_m2c']);K=np.array(camera[str(frame)]['cam_K']).reshape(3,3);im=cv2.imread(str(scene/'rgb'/f'{frame:06d}.jpg'));H,W=im.shape[:2];near,far,m,_=ren.render(v,faces,R,t,K,H,W,.5);y,x=np.nonzero(m)
 for name,z in [('front',near),('back',far)]:
  depth=z[m];xyz=(np.c_[(x-K[0,2])/K[0,0]*depth,(y-K[1,2])/K[1,1]*depth,depth]-t)@R;q=np.rint(np.clip((xyz-lo)/size,0,1)*255).astype('uint8');out=np.zeros((H,W,3),np.uint8);out[m]=q;ref=cv2.imread(str(P/f'obj04_seminar_v1/labels/{name}/{frame:06d}_000000.png'));assert ref is not None and np.array_equal(out,ref)
 pbr.append(frame)
# Confirm all static candidate masks, numeric labels and files have consistent geometry.
with (O/'target_quality.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==100
for r in rows:
 mask=cv2.imread(str(O/r['amodal_mask']),0);front=cv2.imread(str(O/r['front_uncanonicalized']));back=cv2.imread(str(O/r['back_uncanonicalized']));assert mask.shape==(720,1280) and front.shape==back.shape==(720,1280,3) and set(np.unique(mask))<={0,255} and int((mask>0).sum())==int(r['amodal_pixels']);assert np.all(front[mask==0]==0) and np.all(back[mask==0]==0);assert int(r['valid_supervision_pixels'])==0
result={'bb_diagnostic_loader':checks,'pbr_front_back_pixel_equality_uncanonicalized':pbr,'all_100_mask_xyz_format_range_and_zero_background':'PASS','empty_gate_training_block':'PASS','official_symmetry_target_compatibility':'NOT VERIFIED','training_pipeline':'NOT IMPLEMENTED'}
(O/'target_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
