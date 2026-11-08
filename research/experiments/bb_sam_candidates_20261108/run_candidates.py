"""Four-frame SAM 2.1 candidate inference. No GT claims and no model training."""
import hashlib,json,time,uuid,datetime,subprocess,os
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT=Path(__file__).resolve().parent
RGB=Path('/RESEARCH_DATA/HCCEPose_experiments/real_bb100_20261108/rgb_undistorted')
CAD=Path('/RESEARCH_DATA/HCCEPose/bb_visible_audit_20261108_v1/pseudo_masks')
CHECKPOINT=ROOT/'sam2.1_hiera_tiny.pt'
CONFIG='configs/sam2.1/sam2.1_hiera_t.yaml'
FRAMES=(43,75,235,247)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def gpu_apps():
    txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    return [x.strip() for x in txt.splitlines() if x.strip()]
def save_png(path,img):
    ok,data=cv2.imencode('.png',img);assert ok
    with path.open('xb') as f:f.write(data.tobytes())
def prompt_for(rgb,cad):
    x,y,w,h=map(int,cv2.boundingRect(cad));box=np.array([x,y,x+w-1,y+h-1],dtype=np.float32)
    distance=cv2.distanceTransform(cad,cv2.DIST_L2,5);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Dark interior heuristic limits some obvious bright gripper selections; it is NOT a confidence model.
    allowed=(distance>=10)&(gray<=90)
    if not allowed.any():return box,None,{'criterion':'no interior dark candidate; box-only','dark_pixel_threshold':90,'min_cad_boundary_distance_px':10}
    score=np.where(allowed,distance,-1);py,px=np.unravel_index(int(np.argmax(score)),score.shape)
    return box,np.array([[px,py]],dtype=np.float32),{'criterion':'largest CAD-boundary distance among gray<=90 pixels','gray_at_point':int(gray[py,px]),'cad_boundary_distance_px':float(distance[py,px]),'dark_pixel_threshold':90,'min_cad_boundary_distance_px':10}
def plot(rgb,cad,selected,box,point,outpath):
    h,w=cad.shape; x0,y0,x1,y1=map(int,box);pad=max(50,int(max(x1-x0,y1-y0)*.22)); crop=(max(0,x0-pad),max(0,y0-pad),min(w,x1+pad+1),min(h,y1+pad+1));l,t,r,b=crop
    marked=rgb.copy(); cv2.rectangle(marked,(x0,y0),(x1,y1),(255,220,0),2)
    if point is not None:cv2.drawMarker(marked,tuple(point[0].astype(int)),(255,0,255),cv2.MARKER_CROSS,14,2)
    cadview=np.repeat((cad>0)[:,:,None].astype('uint8')*255,3,axis=2);segview=np.repeat(selected[:,:,None].astype('uint8')*255,3,axis=2)
    over=rgb.copy();ix=selected>0;over[ix]=(rgb[ix]*.55+np.array([255,0,0])*.45).astype('uint8');cont,_=cv2.findContours(cad,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,cont,-1,(255,220,0),1)
    diff=np.zeros_like(rgb); both=(cad>0)&selected; onlycad=(cad>0)&~selected;onlysam=(cad==0)&selected
    diff[both]=[80,215,80];diff[onlycad]=[255,80,80];diff[onlysam]=[80,180,255]
    panels=[marked,cadview,segview,over,diff];labels=['RGB + box / positive point','CAD AMODAL (not GT)','SAM pseudo candidate','SAM overlay / CAD outline','CAD vs SAM: green/red/blue']
    canvas=Image.new('RGB',(2000,500),(24,28,36));draw=ImageDraw.Draw(canvas)
    for i,(panel,label) in enumerate(zip(panels,labels)):
        im=Image.fromarray(panel[t:b,l:r]);im.thumbnail((390,420));canvas.paste(im,(i*400+(400-im.width)//2,48+(420-im.height)//2));draw.text((i*400+6,13),label,fill='white')
    draw.text((8,480),'Green: intersection | Red: CAD only | Blue: SAM only | magenta cross: positive point | NO GT',fill='white')
    canvas.save(outpath,format='PNG')

def main():
    if gpu_apps():raise RuntimeError('기존 GPU compute 작업이 있어 추론을 시작하지 않습니다.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    inputs={}
    for i in FRAMES:
        found=list(RGB.glob(f'*/{i:06d}.jpg'));assert len(found)==1
        p=CAD/f'{i:06d}_amodal.png';assert p.is_file();inputs[i]=(found[0],p)
    watched={str(p):sha(p) for pair in inputs.values() for p in pair};watched[str(CHECKPOINT)]=sha(CHECKPOINT)
    out=ROOT/'results'/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    model=build_sam2(CONFIG,str(CHECKPOINT),device='cuda',mode='eval',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);rows=[]
    for frame in FRAMES:
        rgb_path,cad_path=inputs[frame];bgr=cv2.imread(str(rgb_path));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);cad=cv2.imread(str(cad_path),0)
        assert rgb.shape==(720,1280,3) and cad.shape==(720,1280) and set(np.unique(cad)).issubset({0,255})
        box,point,pinfo=prompt_for(rgb,cad);frame_start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks,scores,_=predictor.predict(point_coords=point,point_labels=np.ones(len(point),np.int32) if point is not None else None,box=box,multimask_output=True)
        torch.cuda.synchronize();best=int(np.argmax(scores));candidate=masks[best].astype(bool);amodal=cad>0
        assert masks.shape==(3,720,1280) and candidate.any() and np.isfinite(scores).all()
        for index,mask in enumerate(masks):save_png(out/f'{frame:06d}_pseudo_mask_candidate_{index}.png',mask.astype('uint8')*255)
        save_png(out/f'{frame:06d}_pseudo_mask_candidate_selected.png',candidate.astype('uint8')*255)
        plot(rgb,cad,candidate,box,point,out/f'{frame:06d}_comparison.png')
        inter=int((candidate&amodal).sum());union=int((candidate|amodal).sum());cadonly=int((amodal&~candidate).sum());samonly=int((candidate&~amodal).sum())
        rows.append({'frame':frame,'rgb_state':rgb_path.parent.name,'bbox_xyxy':box.astype(int).tolist(),'positive_point_xy':None if point is None else point[0].astype(int).tolist(),'point_heuristic':pinfo,'sam_predicted_quality_scores':[float(s) for s in scores],'selected_index_highest_predicted_quality':best,'cad_amodal_pixels':int(amodal.sum()),'sam_candidate_pixels':int(candidate.sum()),'intersection_pixels':inter,'cad_only_pixels':cadonly,'sam_only_pixels':samonly,'intersection_over_union_with_CAD_NOT_GT':inter/union,'inference_seconds':time.monotonic()-frame_start})
        print('frame',frame,'selected',best,'points',point,'areas',rows[-1]['cad_amodal_pixels'],rows[-1]['sam_candidate_pixels'],flush=True)
    after={str(p):sha(p) for pair in inputs.values() for p in pair};after[str(CHECKPOINT)]=sha(CHECKPOINT);assert watched==after
    metrics={'status':'CANDIDATES_ONLY_NOT_GT','model':'official SAM 2.1 hiera tiny','config':CONFIG,'checkpoint_sha256':watched[str(CHECKPOINT)],'source_commit':subprocess.check_output(['git','-C',str(ROOT/'sam2_source'),'rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'dtype':'bfloat16 autocast','peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds':time.monotonic()-start,'inputs_unchanged':True,'prompt_limitations':'bbox from CAD and positive point from dark CAD interior; no guarantee point is object or visible','scores_limitation':'SAM predicted quality is not measured accuracy; CAD IoU is not GT','frames':rows}
    with (out/'metrics.json').open('x',encoding='utf8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2);f.write('\n')
    print('RESULT',out,flush=True);print('GPU allocated/reserved',metrics['peak_gpu_allocated_bytes'],metrics['peak_gpu_reserved_bytes'],flush=True)
if __name__=='__main__':main()
