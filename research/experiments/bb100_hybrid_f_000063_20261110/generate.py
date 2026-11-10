from pathlib import Path
import cv2,numpy as np,csv,json,hashlib,sys
P=Path('/home/oylee/projects/HCCEPose_experiments');O=P/'bb100_hybrid_f_000063_20261110';OLD=P/'bb100_visible_gt_phase16a4_20261110';sys.path.insert(0,str(P/'bb_weak_supervision_20261108'));import pipeline

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 row=next(r for r in csv.DictReader((P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig')) if int(r['frame'])==63);impath=Path(row['image_path']);posepath=Path(row['pose_path']);paths={k:OLD/'candidates'/k/'000063.png' for k in 'ABCD'};state=json.loads(__import__('urllib.request',fromlist=['urlopen']).urlopen('http://127.0.0.1:7870/api/000063/state').read());accepted=Path(state['save_path']);tracked=[impath,posepath,accepted,*paths.values()];before={str(p):sha(p) for p in tracked};im=cv2.imread(str(impath));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);cad=cv2.imread(str(paths['A']),0)>0;d=cv2.imread(str(paths['D']),0)>0;T=np.load(posepath);K=np.asarray(json.loads(row['K']));renderer=pipeline.load_renderer();v,faces=renderer.mesh();R=T[:3,:3];t=T[:3,3];near,far,rendered,camera=renderer.render(v,faces,R,t,K,720,1280,.5);assert np.array_equal(cad,rendered)
 # Independent ray/triangle intersections: missing hits prove no CAD surface projects to sampled pixel ray.
 tri=camera[faces];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;ss=-a;q=np.cross(ss,e1);num=np.sum(e2*q,axis=1)
 def ray_hits(x,y):
  direction=np.array([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],1.]);h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.sum(e1*h,axis=1);good=np.abs(det)>1e-10;inv=np.divide(1.,det,out=np.zeros_like(det),where=good);u=np.sum(ss*h,axis=1)*inv;w=np.sum(q*direction,axis=1)*inv;z=num*inv;return bool(np.any(good&(u>=-1e-7)&(w>=-1e-7)&(u+w<=1+1e-7)&(z>0)))
 contours,hierarchy=cv2.findContours(cad.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);geometry=np.zeros(cad.shape,bool);used=np.zeros(cad.shape,bool);uncertain=np.zeros(cad.shape,bool);rows=[];kernel=np.ones((3,3),np.uint8)
 for idx,c in enumerate(contours):
  if hierarchy[0,idx,3]<0:continue
  mask=np.zeros(cad.shape,np.uint8);cv2.drawContours(mask,contours,idx,255,-1);region=(mask>0)&~cad
  if region.sum()<8:continue
  geometry|=region;core=cv2.erode(region.astype('uint8'),kernel)>0;ring=(cv2.dilate(region.astype('uint8'),np.ones((11,11),np.uint8))>0)&cad
  baseline=float(np.median(gray[ring])) if ring.any() else None
  bright=core&(gray>=min(255,baseline+40)) if baseline is not None else np.zeros_like(core)
  fraction=float(bright.sum()/core.sum()) if core.any() else 0
  qualifies=bool(core.sum()>=5 and baseline is not None and baseline<100 and fraction>=.8)
  chosen=bright if qualifies else np.zeros_like(core)
  # Conservatively drop any pixel with an independent CAD intersection.
  yy,xx=np.nonzero(chosen);hits=[(int(x),int(y)) for y,x in zip(yy,xx) if ray_hits(int(x),int(y))]
  for x,y in hits:chosen[y,x]=False
  used|=chosen;uncertain|=region&~chosen
  rows.append({'component':len(rows)+1,'bbox_xywh':list(map(int,cv2.boundingRect(region.astype('uint8')))),'cad_hole_pixels':int(region.sum()),'eroded_core_pixels':int(core.sum()),'surrounding_median_gray':baseline,'core_fraction_40_brighter':fraction,'qualifies_heuristic':qualifies,'independent_ray_hits_excluded':len(hits),'used_pixels':int(chosen.sum()),'marked_uncertain_pixels':int((region&~chosen).sum()),'reason':'RGB brightness heuristic only; CAD alignment and visible surface not independently verified'})
 F=d&~used;ref=cv2.imread(str(accepted),0);assert ref.shape==(720,1280) and set(np.unique(ref)).issubset({0,128,255});removed=d&used;conflict=removed&(ref==255);assert not (used&cad).any()
 for name,a in [('H_CAD_closed_geometry.png',geometry),('H_CAD_selected_core.png',used),('H_CAD_uncertain_review.png',uncertain),('F_HYBRID_000063_NOT_GT.png',F),('F_removed_from_D.png',removed),('F_removed_conflicts_accepted_63.png',conflict)]:
  path=O/name;assert not path.exists() and cv2.imwrite(str(path),a.astype('uint8')*255)
 # RGB + B/C/D/F and uncertainty comparisons at same crop, nearest-neighbour preserves exact pixels.
 x0,y0,x1,y1=350,190,590,390;B=cv2.imread(str(paths['B']),0)>0;C=cv2.imread(str(paths['C']),0)>0
 def overlay(m,color):
  a=im.copy();a[m]=(a[m]*.45+np.array(color)*.55).astype('uint8');return a
 marked=im.copy();marked[uncertain]=(marked[uncertain]*.45+np.array([0,200,255])*.55).astype('uint8');marked[used]=(marked[used]*.35+np.array([0,0,255])*.65).astype('uint8')
 views=[(im,'RGB'),(overlay(B,[255,0,220]),'B SAM NOT GT'),(overlay(C,[0,220,0]),'C CAD SAM NOT GT'),(overlay(d,[255,0,220]),'D SAM ROBOT NOT GT'),(overlay(F,[255,0,220]),'F HYBRID NOT GT'),(marked,'RED REMOVED / YELLOW REVIEW')]
 tiles=[]
 for v,title in views:
  z=cv2.resize(v[y0:y1,x0:x1],(720,600),interpolation=cv2.INTER_NEAREST);cv2.rectangle(z,(0,0),(720,30),(0,0,0),-1);cv2.putText(z,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA);tiles.append(z)
 cv2.imwrite(str(O/'000063_B_C_D_F_comparison.png'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
 with (O/'hole_components.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 assert before=={str(p):sha(p) for p in tracked}
 out={'frame':'000063','cad_holes_count':len(rows),'closed_hole_geometry_pixels':int(geometry.sum()),'selected_geometry_hole_pixels':int(used.sum()),'uncertain_hole_pixels':int(uncertain.sum()),'F_removed_pixels':int(removed.sum()),'F_removed_conflicting_accepted_foreground_pixels':int(conflict.sum()),'prior_accepted_status':state['status'],'prior_accepted_revision':state['revision'],'removed_nongap_actual_gt':'UNKNOWN: no independent pixel reference; existing accepted mask disagrees, not automatically adjudicated','closed_hole_pixel_ray_excluded_count':sum(x['independent_ray_hits_excluded'] for x in rows),'input_hashes_unchanged':True,'inputs':before,'selection':'3x3 erosion of enclosed CAD background; ring median<100, >=80% core pixels >= ring median+40; retain only bright core pixels with no CAD ray hit; other hole pixels uncertain. Heuristic is not calibrated.'}
 (O/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='inputs'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
