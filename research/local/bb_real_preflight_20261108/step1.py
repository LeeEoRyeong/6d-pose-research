"""Read-only source audit and single-frame pseudo-label experiment; never edits source data."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EXP = Path('/RESEARCH_DATA/HCCEPose_experiments')
BB = EXP / 'real_bb100_20261108'
OLD = EXP / 'obj04_seminar_v1'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD / 'scripts'))
spec = importlib.util.spec_from_file_location('legacy_xyz', OLD / 'scripts/generate_xyz_v2.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
load = lambda p: json.loads(p.read_text())

def main():
    gt = load(BB / 'BB_obj04/scene_gt_obj04.json')
    cameras = load(BB / 'BB_obj04/scene_camera_obj04.json')
    images = list((BB / 'rgb_undistorted').rglob('*.jpg'))
    by_id = {}
    for p in images:
        by_id.setdefault(int(p.stem), []).append(p)
    ids = set(map(int, gt))
    missing = sorted(i for i in ids if len(by_id.get(i, [])) != 1)
    pose_mismatch = []
    for i in sorted(ids):
        p = BB / 'BB_obj04/poses_obj04' / f'pose_{i:06d}.npy'
        if not p.exists():
            pose_mismatch.append(i)
            continue
        pose = np.load(p, allow_pickle=False)
        item = gt[str(i)][0]
        if not (np.allclose(pose[:3,:3].ravel(), item['cam_R_m2c'], atol=1e-9) and np.allclose(pose[:3,3], item['cam_t_m2c'], atol=1e-9)):
            pose_mismatch.append(i)
    assert set(gt) == set(cameras) and not missing and not pose_mismatch
    v, faces = legacy.mesh()
    info = load(OLD / 'data/models/models_info.json')['4']
    item = gt['43'][0]
    R = np.array(item['cam_R_m2c']).reshape(3,3)
    t = np.array(item['cam_t_m2c'])
    K = np.array(cameras['43']['cam_K']).reshape(3,3)
    rgb = cv2.imread(str(by_id[43][0])); assert rgb is not None
    h,w = rgb.shape[:2]
    near,far,valid,cam = legacy.render(v,faces,R,t,K,h,w,.5)
    assert valid.any() and (far[valid] >= near[valid]).all()
    yy,xx = np.nonzero(valid)
    minimum = np.array([info['min_x'],info['min_y'],info['min_z']])
    size = np.array([info['size_x'],info['size_y'],info['size_z']])
    stats = []
    for name,z in [('front',near),('back',far)]:
        zz=z[valid]
        xyz=(np.column_stack(((xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz))-t)@R
        norm=(xyz-minimum)/size
        assert np.isfinite(norm).all()
        quant=np.rint(np.clip(norm,0,1)*255).astype(np.uint8)
        label=np.zeros((h,w,3),np.uint8); label[valid]=quant
        assert cv2.imwrite(str(OUT/f'43_{name}_pseudo.png'),label)
        restored=quant/255*size+minimum
        p=restored@R.T+t; uv=p@K.T; uv=uv[:,:2]/uv[:,2:]
        stats.append({'name':name,'xyz_min':xyz.min(axis=0).tolist(),'xyz_max':xyz.max(axis=0).tolist(),'max_normalized_clip':float(np.maximum(-norm,norm-1).clip(0).max()),'reprojection_p99_px':float(np.quantile(np.linalg.norm(uv-np.column_stack((xx,yy)),axis=1),.99))})
    mask=(valid*255).astype(np.uint8); assert cv2.imwrite(str(OUT/'43_rendered_full_pseudo_mask.png'),mask)
    overlay=rgb.copy(); overlay[valid]=(0.55*rgb[valid]+0.45*np.array([0,200,0])).astype(np.uint8)
    assert cv2.imwrite(str(OUT/'43_pseudo_overlay.png'),overlay)
    idx=np.linspace(0,len(yy)-1,10,dtype=int)
    hits=legacy.independent_rays(cam,faces,K,list(zip(yy[idx],xx[idx])),.5)
    ray_error=max(max(abs(near[y,x]-n),abs(far[y,x]-f)) for y,x,n,f in hits)
    result={'status':'NOT_PASSED_visible_mask_unverified','frame':43,'rgb':str(by_id[43][0]),'gt_count':len(gt),'rgb_count':len(images),'unique_gt_rgb_mismatches':missing,'pose_file_mismatches':pose_mismatch,'model_sha256':hashlib.sha256((OLD/'data/models/obj_000004.ply').read_bytes()).hexdigest(),'R':R.tolist(),'t_mm_as_provided':t.tolist(),'K':K.tolist(),'resolution_wh':[w,h],'rendered_pixels':int(valid.sum()),'rendered_fraction':float(valid.mean()),'bbox_xywh':list(map(int,cv2.boundingRect(mask))),'ray_minmax_max_error_mm':float(ray_error),'nan_inf_valid':0,'labels':stats,'warning':'Full CAD rendering is pseudo mask, not verified visible mask.'}
    (OUT/'step1_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == '__main__': main()
