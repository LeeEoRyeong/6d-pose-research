"""CPU 전용 좌표 채널·bit 분포 및 GT 채널 치환 Oracle. 실제 모델 성능 아님."""
import diagnose as d
import numpy as np,csv,json
O=d.ROOT/'channel_controls';O.mkdir()
for name in ('tables','checks','figures'):(O/name).mkdir()
P=d.ROOT/'supplement/cache';d.ROOT=O
stats=[]
for frame in d.VAL:
    z=np.load(P/f'{frame:06d}.npz');mask=z['gtmask']&z['valid'];K=np.array(d.CAM[str(frame)]['cam_K']).reshape(3,3);g=d.GT[str(frame)][0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c'])
    for branch in ('front','back'):
        gt=z[branch+'_gt'];pred=z[branch+'_pred'];other=z[('back' if branch=='front' else 'front')+'_gt']
        for i,a in enumerate('xyz'):
            stats.append({'frame':frame,'branch':branch,'axis':a,'gt_mean':float(gt[mask,i].mean()),'pred_mean':float(pred[mask,i].mean()),'gt_upper_half_fraction':float(np.mean(gt[mask,i]>=d.LO[i]+d.SIZE[i]/2)),'pred_upper_half_fraction':float(np.mean(pred[mask,i]>=d.LO[i]+d.SIZE[i]/2)),'direct_branch_xyz_mean_error':float(np.linalg.norm(pred[mask]-gt[mask],axis=1).mean()),'other_branch_xyz_mean_error':float(np.linalg.norm(pred[mask]-other[mask],axis=1).mean())})
    for name,axis in [('predFront_GTvalidMask_control',None),('GTreplace_X_Oracle',0),('GTreplace_Y_Oracle',1),('GTreplace_Z_Oracle',2)]:
        xyz=z['front_pred'].copy()
        if axis is not None:xyz[...,axis]=z['front_gt'][...,axis]
        d.pose(name,frame,mask,xyz,z['uv'],K,R,t)
d.table('channel_distribution.csv',stats);d.table('GT_channel_replacement_Oracle.csv',d.ALL);d.summarize()
print(json.dumps(d.S.load_json(O/'checks/summary.json'),ensure_ascii=False,indent=2))
