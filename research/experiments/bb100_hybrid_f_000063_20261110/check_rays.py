from pathlib import Path
import sys,csv,json,cv2,numpy as np
P=Path('/home/oylee/projects/HCCEPose_experiments');O=P/'bb100_hybrid_f_000063_20261110';sys.path.insert(0,str(P/'bb_weak_supervision_20261108'));import pipeline
r=next(x for x in csv.DictReader((P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig')) if int(x['frame'])==63);T=np.load(r['pose_path']);K=np.asarray(json.loads(r['K']));renderer=pipeline.load_renderer();v,faces=renderer.mesh();camera=v@T[:3,:3].T+T[:3,3];tri=camera[faces];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;ss=-a;q=np.cross(ss,e1);num=np.sum(e2*q,axis=1);mask=cv2.imread(str(O/'H_CAD_closed_geometry.png'),0)>0;ys,xs=np.nonzero(mask);hits=[]
for x,y in zip(xs,ys):
 d=np.array([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],1.]);h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.sum(e1*h,axis=1);good=abs(det)>1e-10;inv=np.divide(1,det,out=np.zeros_like(det),where=good);u=np.sum(ss*h,axis=1)*inv;w=np.sum(q*d,axis=1)*inv;z=num*inv
 if np.any(good&(u>=-1e-7)&(w>=-1e-7)&(u+w<=1+1e-7)&(z>0)):hits.append([int(x),int(y)])
report={'closed_geometry_pixel_count':len(xs),'independent_triangle_ray_hit_count':len(hits),'hits':hits,'meaning':'no CAD triangle on pixel-center ray; does not prove RGB appearance or watertight topology'};(O/'independent_hole_rays.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='hits'}));assert not hits
