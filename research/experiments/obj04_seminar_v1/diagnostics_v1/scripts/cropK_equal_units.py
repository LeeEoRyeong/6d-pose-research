"""Crop-K 대조에서 1.5 원본 px 임계값도 같은 affine 배율로 변환하는 CPU 확인."""
import diagnose as d
import numpy as np
O=d.ROOT/'cropK_equal_units';O.mkdir()
for n in ('checks','tables','figures'):(O/n).mkdir()
P=d.ROOT/'supplement/cache';d.ROOT=O
for frame in d.VAL:
    z=np.load(P/f'{frame:06d}.npz');box=d.PAD(np.array(d.GI[str(frame)][0]['bbox_visib'],float),1.5);scale=128/box[2];A=np.array([[scale,0,-box[0]*scale],[0,scale,-box[1]*scale],[0,0,1]])
    K=np.array(d.CAM[str(frame)]['cam_K']).reshape(3,3);g=d.GT[str(frame)][0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c'])
    yy,xx=np.indices((128,128));uv=np.stack([xx,yy],-1).astype(np.float32)
    row=d.pose('A2_cropK_equal_threshold_Oracle',frame,z['gtmask'],z['front_gt'],uv,A@K,R,t,threshold=1.5*scale);row['reprojection_unit']='crop pixels';row['scale_crop_per_original']=float(scale)
d.table('cropK_equal_threshold_Oracle.csv',d.ALL);d.summarize()
