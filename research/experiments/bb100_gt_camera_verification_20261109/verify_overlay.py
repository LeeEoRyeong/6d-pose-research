import sys,json,cv2,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;E=P.parent;B=E/'real_bb100_20261108/BB_obj04';sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry
v,faces,_,_=symmetry.load();g=json.load((B/'scene_gt_obj04.json').open());cam=json.load((B/'scene_camera_obj04.json').open());prior=Path('/home/oylee/projects/HCCEPose/bb_visible_audit_20261108_v1/pseudo_masks');z=[]
for i in [31,43,67,75,99,221,235,247,340,379]:
 a=g[str(i)][0];R=np.array(a['cam_R_m2c']).reshape(3,3);t=np.array(a['cam_t_m2c']);K=np.array(cam[str(i)]['cam_K']).reshape(3,3);pc=v@R.T+t;uv=pc@K.T;uv=uv[:,:2]/uv[:,2:];m=np.zeros((720,1280),np.uint8)
 for poly in np.rint(uv[faces]).astype('int32'):cv2.fillConvexPoly(m,poly,255)
 ref=cv2.imread(str(prior/f'{i:06d}_amodal.png'),0)>0;ours=m>0;inter=int(np.logical_and(ref,ours).sum());union=int(np.logical_or(ref,ours).sum());z.append({'frame':i,'iou_to_existing_cpu_renderer':inter/union,'ours_area':int(ours.sum()),'renderer_area':int(ref.sum()),'scope':'synthetic CAD rasterization agreement, not visible GT or pose validation'})
json.dump(z,(P/'overlay_rasterization_check.json').open('x'),indent=2);print('min IoU',min(x['iou_to_existing_cpu_renderer'] for x in z),'mean',np.mean([x['iou_to_existing_cpu_renderer'] for x in z]))
