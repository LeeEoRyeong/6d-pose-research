import csv, hashlib, importlib.util, json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
EXP=Path('/RESEARCH_DATA/HCCEPose_experiments')
OLD=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')
BB=EXP/'real_bb100_20261108'
BASE=EXP/'obj04_seminar_v1'
SCENE=EXP/'realistic_final100/train_pbr/000000'
sys.path.insert(0,str(BASE/'scripts'))
spec=importlib.util.spec_from_file_location('legacy_xyz', BASE/'scripts/generate_xyz_v2.py')
legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
load=lambda p: json.loads(p.read_text())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def rot(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def pose(d): return np.array(d[0]['cam_R_m2c']).reshape(3,3),np.array(d[0]['cam_t_m2c'])
def thumb(p):
    x=cv2.imread(str(p)); assert x is not None
    return cv2.resize(x,(160,90),interpolation=cv2.INTER_AREA).astype(np.float32)

bg=load(BB/'BB_obj04/scene_gt_obj04.json'); bc=load(BB/'BB_obj04/scene_camera_obj04.json'); og=load(OLD/'scene_gt_obj04.json')
byid={int(p.stem):p for p in (BB/'rgb_undistorted').rglob('*.jpg') if str(int(p.stem)) in bg}
assert len(byid)==100
images=[]
for dataset,root,gt in [('old_raw',OLD/'image',og),('old_undistorted',OLD/'image_undistorted',og)]:
    for p in sorted(root.glob('*.jpg')): images.append({'dataset':dataset,'id':int(p.stem),'path':str(p),'sha256':sha(p)})
for i,p in sorted(byid.items()): images.append({'dataset':'bb_undistorted','id':i,'path':str(p),'sha256':sha(p)})
writecsv(OUT/'image_hashes.csv',images)
old=[x for x in images if x['dataset']=='old_undistorted']; new=[x for x in images if x['dataset']=='bb_undistorted']
cache={x['path']:thumb(x['path']) for x in old+new}
pairs=[]
for a in old:
    ra,ta=pose(og[str(a['id'])]); aa=cache[a['path']]
    for b in new:
        rb,tb=pose(bg[str(b['id'])]); delta=aa-cache[b['path']]
        pairs.append({'old_id':a['id'],'bb_id':b['id'],'bb_state':Path(b['path']).parent.name,'same_id':a['id']==b['id'],'sha_equal':a['sha256']==b['sha256'],'rgb_thumbnail_mae':float(np.abs(delta).mean()),'rotation_deg':rot(ra,rb),'translation_distance_model_units':float(np.linalg.norm(ta-tb))})
writecsv(OUT/'all_2700_pairs.csv',pairs)
closest=[]
for a in old:
    pp=[p for p in pairs if p['old_id']==a['id']]
    for criterion in ('rgb_thumbnail_mae','translation_distance_model_units','rotation_deg'):
        x=min(pp,key=lambda p:p[criterion]); closest.append({'criterion':criterion,**x})
writecsv(OUT/'nearest_pairs.csv',closest)
(OUT/'visualizations').mkdir(exist_ok=False);(OUT/'pseudo_masks').mkdir(exist_ok=False)
v,faces=legacy.mesh(); info=load(BASE/'data/models/models_info.json')['4']; minimum=np.array([info['min_x'],info['min_y'],info['min_z']]); size=np.array([info['size_x'],info['size_y'],info['size_z']])
ids=[31,43,67,75,99,221,235,247,340,379]
rows=[]
for i in ids:
    R,t=pose(bg[str(i)]); K=np.array(bc[str(i)]['cam_K']).reshape(3,3); rgb=cv2.imread(str(byid[i])); h,w=rgb.shape[:2]
    near,far,valid,cam=legacy.render(v,faces,R,t,K,h,w,.5); mask=valid.astype('uint8')*255
    assert cv2.imwrite(str(OUT/'pseudo_masks'/f'{i:06d}_amodal.png'),mask)
    x,y,bw,bh=cv2.boundingRect(mask); pad=45; box=(max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad));x0,y0,x1,y1=box
    overlay=rgb.copy(); overlay[valid]=(rgb[valid]*.60+np.array([0,180,0])*.40).astype('uint8')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,255,255),1)
    views=[rgb,overlay,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)]; titles=['RGB','CAD projection: green / outline: yellow','CAD pseudo mask (amodal)']
    canvas=Image.new('RGB',(1200,460),(25,25,25)); d=ImageDraw.Draw(canvas)
    for j,(im,title) in enumerate(zip(views,titles)):
        img=Image.fromarray(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));img.thumbnail((390,400),Image.Resampling.LANCZOS);canvas.paste(img,(j*400+(400-img.width)//2,45+(400-img.height)//2));d.text((j*400+8,12),title,fill='white')
    d.text((8,440),f'{i:06d}  {byid[i].parent.name}   no visible-mask ground truth',fill='white');canvas.save(OUT/'visualizations'/f'{i:06d}.png')
    yy,xx=np.nonzero(valid); p99=[]; clip=[]
    for z in [near,far]:
        zz=z[valid]; xyz=(np.column_stack(((xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz))-t)@R
        n=(xyz-minimum)/size; q=np.rint(np.clip(n,0,1)*255).astype('uint8'); rec=q/255*size+minimum; proj=(rec@R.T+t)@K.T; uv=proj[:,:2]/proj[:,2:];p99.append(float(np.quantile(np.linalg.norm(uv-np.column_stack((xx,yy)),axis=1),.99)));clip.append(float(np.maximum(-n,n-1).clip(0).max()))
    rows.append({'frame':i,'state':byid[i].parent.name,'pseudo_mask_pixels':int(valid.sum()),'bbox_xywh':str([x,y,bw,bh]),'front_reprojection_p99_px':p99[0],'back_reprojection_p99_px':p99[1],'max_clip':max(clip),'near_le_far':bool(np.all(near[valid]<=far[valid]))})
writecsv(OUT/'representative_render_metrics.csv',rows)
pgt=load(SCENE/'scene_gt.json'); pc=load(SCENE/'scene_camera.json'); tests=[]
for i in [0,10,40,70,99]:
    R,t=pose(pgt[str(i)]);K=np.array(pc[str(i)]['cam_K']).reshape(3,3); ref=cv2.imread(str(SCENE/'mask'/f'{i:06d}_000000.png'),0)>0;h,w=ref.shape
    near,far,valid,cam=legacy.render(v,faces,R,t,K,h,w,.5); yy,xx=np.nonzero(valid); diffs=[]
    for folder,z in [('front',near),('back',far)]:
        zz=z[valid];xyz=(np.column_stack(((xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz))-t)@R
        img=np.zeros((h,w,3),np.uint8);img[valid]=np.rint(np.clip((xyz-minimum)/size,0,1)*255).astype('uint8')
        stored=cv2.imread(str(BASE/'labels'/folder/f'{i:06d}_000000.png'),cv2.IMREAD_UNCHANGED); assert stored is not None
        diff=np.abs(img.astype('int16')-stored.astype('int16')); diffs.append({'label':folder,'exact':bool(np.array_equal(img,stored)),'different_channel_values':int(np.count_nonzero(diff)),'max_difference':int(diff.max())})
    hits=legacy.independent_rays(cam,faces,K,list(zip(yy[np.linspace(0,len(yy)-1,10,dtype=int)],xx[np.linspace(0,len(xx)-1,10,dtype=int)])),.5)
    tests.append({'frame':i,'full_mask_iou':float((valid&ref).sum()/(valid|ref).sum()),'ray_max_error_model_units':max(max(abs(near[y,x]-n),abs(far[y,x]-f)) for y,x,n,f in hits),'labels':diffs})
(OUT/'pbr_xyz_comparison.json').write_text(json.dumps(tests,indent=2)+'\n')
summary={'old_count':len(old),'bb_count':len(new),'exact_hash_pairs':sum(p['sha_equal'] for p in pairs),'same_id_pairs':[p for p in pairs if p['same_id']],'smallest_image_mae':sorted(pairs,key=lambda p:p['rgb_thumbnail_mae'])[:10],'smallest_translation':sorted(pairs,key=lambda p:p['translation_distance_model_units'])[:10],'pbr_tests':tests}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
