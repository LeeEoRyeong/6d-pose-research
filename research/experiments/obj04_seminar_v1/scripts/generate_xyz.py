"""CPU 라벨: 양면 삼각형의 최소/최대 카메라 깊이, 원본 HCCE 역투영·PNG 채널 규약."""
import argparse,json,time,traceback
import numpy as np
import cv2
from PIL import Image,ImageDraw
from common import ROOT,SCENE,write_json,load_json

def mesh():
    p=ROOT/'data/models/obj_000004.ply';lines=p.read_text().splitlines();n=lines.index('end_header')+1
    nv=int(next(x.split()[-1] for x in lines[:n] if x.startswith('element vertex ')))
    nf=int(next(x.split()[-1] for x in lines[:n] if x.startswith('element face ')))
    v=np.array([list(map(float,x.split()[:3])) for x in lines[n:n+nv]],dtype=np.float64)
    f=np.array([list(map(int,x.split()[1:])) for x in lines[n+nv:n+nv+nf]],dtype=np.int64)
    assert f.shape==(nf,3) and f.min()>=0 and f.max()<len(v)
    return v,f

def render(v,faces,R,t,K,H,W,offset):
    camera=v@R.T+t
    assert np.isfinite(camera).all() and camera[:,2].min()>0
    uv=camera@K.T;uv=uv[:,:2]/uv[:,2:]
    near=np.full((H,W),np.inf,dtype=np.float64);far=np.full((H,W),-np.inf,dtype=np.float64)
    for face in faces:
        a,b,c=uv[face];az,bz,cz=camera[face,2]
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        x0=max(0,int(np.floor(min(a[0],b[0],c[0])-offset)));x1=min(W-1,int(np.ceil(max(a[0],b[0],c[0])-offset)))
        y0=max(0,int(np.floor(min(a[1],b[1],c[1])-offset)));y1=min(H-1,int(np.ceil(max(a[1],b[1],c[1])-offset)))
        if x0>x1 or y0>y1:continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1];xx=xx+offset;yy=yy+offset
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        w=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        inside=(u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)
        z=1/(u/az+w/bz+(1-u-w)/cz)
        n=near[y0:y1+1,x0:x1+1];f=far[y0:y1+1,x0:x1+1]
        np.minimum(n,np.where(inside,z,np.inf),out=n)
        np.maximum(f,np.where(inside,z,-np.inf),out=f)
    valid=np.isfinite(near)&np.isfinite(far)
    return near,far,valid,camera

def independent_rays(camera,faces,K,pixels,offset):
    # 독립 Moller-Trumbore 교차로 min/max 깊이 확인. 정점 보간 코드와 다름.
    triangles=camera[faces];a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    hits=[]
    for y,x in pixels:
        d=np.array([(x+offset-K[0,2])/K[0,0],(y+offset-K[1,2])/K[1,1],1.])
        h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.sum(e1*h,axis=1)
        good=abs(det)>1e-10;inv=np.divide(1.,det,out=np.zeros_like(det),where=good)
        s=-a;u=np.sum(s*h,axis=1)*inv;q=np.cross(s,e1);w=np.sum(q*d,axis=1)*inv;z=np.sum(e2*q,axis=1)*inv
        mask=good&(u>=-1e-7)&(w>=-1e-7)&(u+w<=1+1e-7)&(z>0)
        assert mask.any(),'독립 ray 교차점 없음'
        hits.append((int(y),int(x),float(z[mask].min()),float(z[mask].max())))
    return hits

def panel(frame,front,back,mask):
    rgb=Image.open(SCENE/'rgb'/f'{frame:06d}.jpg').convert('RGB')
    bbox=Image.fromarray(mask.astype(np.uint8)*255).getbbox();x0,y0,x1,y1=bbox
    box=(max(0,x0-25),max(0,y0-25),min(rgb.width,x1+25),min(rgb.height,y1+25))
    images=[rgb,Image.fromarray(cv2.imread(str(SCENE/'mask_visib'/f'{frame:06d}_000000.png'),0)).convert('RGB'),Image.fromarray(front),Image.fromarray(back)]
    canvas=Image.new('RGB',(1280,360),'#202020');draw=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(images,['PBR RGB','Observed visible mask','Front XYZ (X=R,Y=G,Z=B)','Back XYZ (X=R,Y=G,Z=B)'])):
        im=im.crop(box);im.thumbnail((310,310));canvas.paste(im,(i*320+(320-im.width)//2,40+(310-im.height)//2));draw.text((i*320+6,12),title,fill='white')
    canvas.save(ROOT/'figures'/f'xyz_comparison_{frame:06d}.png')
    color=np.zeros((mask.shape[0],mask.shape[1],4),dtype=np.uint8);color[mask]=[0,255,80,100]
    overlay=Image.alpha_composite(rgb.convert('RGBA'),Image.fromarray(color,'RGBA')).convert('RGB')
    overlay.save(ROOT/'figures'/f'pbr_cad_overlay_{frame:06d}.png')

def generate(frame,offset,v,faces,gt,cams,info):
    start=time.monotonic();item=gt[str(frame)][0];R=np.array(item['cam_R_m2c']).reshape(3,3);t=np.array(item['cam_t_m2c']);K=np.array(cams[str(frame)]['cam_K']).reshape(3,3)
    ref=cv2.imread(str(SCENE/'mask'/f'{frame:06d}_000000.png'),0)>0;H,W=ref.shape
    near,far,valid,camera=render(v,faces,R,t,K,H,W,offset)
    assert valid.any() and np.all(far[valid]>=near[valid])
    iou=float(np.count_nonzero(valid&ref)/np.count_nonzero(valid|ref))
    depth=cv2.imread(str(SCENE/'depth'/f'{frame:06d}.png'),cv2.IMREAD_UNCHANGED).astype(float)*cams[str(frame)]['depth_scale']
    vis=cv2.imread(str(SCENE/'mask_visib'/f'{frame:06d}_000000.png'),0)>0
    depth_error=np.abs(near[valid&vis]-depth[valid&vis])
    yy,xx=np.nonzero(valid);indices=np.linspace(0,len(yy)-1,5,dtype=int)
    hits=independent_rays(camera,faces,K,list(zip(yy[indices],xx[indices])),offset)
    ray_error=max(max(abs(near[y,x]-n),abs(far[y,x]-f)) for y,x,n,f in hits)
    minimum=np.array([info['min_x'],info['min_y'],info['min_z']]);size=np.array([info['size_x'],info['size_y'],info['size_z']])
    outputs=[];max_clip=0.;reproj=[]
    for z in (near,far):
        zz=z[valid];xc=np.column_stack([(xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz])
        xyz=(xc-t)@R # 원본 rendering_bop_dataset_back_front와 같은 역변환
        assert np.isfinite(xyz).all()
        norm=(xyz-minimum)/size
        max_clip=max(max_clip,float(np.maximum(-norm,norm-1).clip(0).max()))
        quant=np.rint(np.clip(norm,0,1)*255).astype(np.uint8)
        restored=quant.astype(float)/255*size+minimum;rc=restored@R.T+t;uv=rc@K.T;uv=uv[:,:2]/uv[:,2:]
        error=np.linalg.norm(uv-np.column_stack([xx,yy]),axis=1);reproj.append(float(np.quantile(error,.99)))
        img=np.zeros((H,W,3),dtype=np.uint8);img[valid]=quant;outputs.append(img)
    stats={'frame':frame,'offset':offset,'mask_iou':iou,'depth_abs_median':float(np.median(depth_error)),'depth_abs_p95':float(np.quantile(depth_error,.95)),
           'ray_minmax_max_error':ray_error,'valid_pixels':int(valid.sum()),'back_minus_front_median':float(np.median((far-near)[valid])),
           'quantized_reprojection_p99_px':reproj,'max_normalized_clip':max_clip,'nan_inf_valid':0,'invalid_pixels_zero':True,'seconds':time.monotonic()-start}
    assert iou>=.97,stats
    assert stats['depth_abs_median']<=2.0 and stats['depth_abs_p95']<=3.0,stats
    assert ray_error<1e-5,stats
    assert max(reproj)<2.0 and max_clip<.02,stats
    for folder,img in zip(('front','back'),outputs):
        path=ROOT/'labels'/folder/f'{frame:06d}_000000.png'
        assert not path.exists()
        # 원본 cv2.imwrite/읽기 규약: 메모리 채널은 XYZ, PNG RGB 순서는 ZYX.
        assert cv2.imwrite(str(path),img)
        assert np.array_equal(cv2.imread(str(path)),img)
    if frame in (0,10,40):panel(frame,*outputs,valid)
    if frame==0:
        np.savez_compressed(ROOT/'checks/depth_minmax_000000.npz',front=np.where(valid,near,0).astype(np.float32),back=np.where(valid,far,0).astype(np.float32),mask=valid)
    write_json(ROOT/'labels/stats'/f'{frame:06d}.json',stats)
    return stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,choices=[1,10,100],required=True);args=parser.parse_args()
    for name in ('front','back','stats'):(ROOT/'labels'/name).mkdir(exist_ok=True)
    assert not (ROOT/'checks'/f'labels_{args.limit:03d}.json').exists()
    v,faces=mesh();gt=load_json(SCENE/'scene_gt.json');cams=load_json(SCENE/'scene_camera.json');info=load_json(ROOT/'data/models/models_info.json')['4']
    stats=[]
    for frame in range(args.limit):
        cached=ROOT/'labels/stats'/f'{frame:06d}.json'
        row=load_json(cached) if cached.exists() else generate(frame,.5,v,faces,gt,cams,info)
        for folder in ('front','back'):assert (ROOT/'labels'/folder/f'{frame:06d}_000000.png').is_file()
        stats.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(ROOT/'checks'/f'labels_{args.limit:03d}.json',{'status':'PASS','frames':args.limit,'renderer':'CPU numpy min/max depth, no culling, perspective correct','source_equivalence':'원본 front GL_LESS/back GL_GREATER의 의미와 동일. VisPy 실행·bitwise 동일성 미검증. 픽셀 중심 .5, XYZ 역투영은 원본 정수좌표 규약','stats':stats})
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
