"""원본 소스/체크포인트를 읽는 GT Oracle 및 예측 좌표 병목 진단. 재학습 없음."""
import sys,json,time,datetime,os,subprocess,hashlib,csv,itertools,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TRAIN=BASE/'runs/baseline_seed42_20261103_214353/trained_only'
sys.path.insert(0,str(TRAIN/'scripts'))
import support as S
sys.path.insert(0,str(BASE/'scripts'));import generate_xyz_v2 as renderer
cv2.setNumThreads(1);torch.set_num_threads(2)
START=time.monotonic();END=START+50*60

def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'tables'/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def guard(gpu=False):
    if time.monotonic()>END:raise RuntimeError('50분 경과: 계산 중단 및 보고서 우선')
    if gpu:
        q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        other=[x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
        if other:raise RuntimeError('다른 GPU 작업 감지: '+str(other))
INFO=S.load_json(BASE/'data/models/models_info.json')['4'];LO=np.array([INFO['min_'+a] for a in 'xyz']);SIZE=np.array([INFO['size_'+a] for a in 'xyz']);V,F=renderer.mesh();DIAM=INFO['diameter']
ENC=S.original_function('HccePose/bop_loader.py','train_bop_dataset_back_front.hcce_encode',{'np':np})
DEC=S.original_function('HccePose/network_model.py','HccePose_BF_Net.hcce_decode',{'torch':torch})
CROP=S.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2})
PAD=S.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np})
def decoded(q):
    code=ENC(None,q)
    with torch.inference_mode():d=DEC(SimpleNamespace(powers=None),torch.tensor(code[None]>.5,dtype=torch.float32))[0].numpy()
    return d,code
class CVProxy:
    def __getattr__(self,n):return getattr(cv2,n)
    def solvePnPRansac(self,*a,**kw):
        self.points=np.array(a[0]);self.uv=np.array(a[1]);out=cv2.solvePnPRansac(*a,**kw);self.raw_inliers=out[3];return out
PROXY=CVProxy();SOLVE=S.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':PROXY,'itertools':itertools})
ALL=[];ORACLE=[];ERRORS=[];CAND=[];GEOM=[];RECS={};PRED={}
GT=S.load_json(S.SCENE/'scene_gt.json');CAM=S.load_json(S.SCENE/'scene_camera.json');GI=S.load_json(S.SCENE/'scene_gt_info.json');VAL=S.load_json(BASE/'data/dataset.json')['split']['val']

def project(x,R,t,K):
    xc=x@R.T+np.asarray(t).reshape(1,3);uv=xc@K.T;return uv[:,:2]/uv[:,2:]
def pose(stage,frame,mask,front,uv,K,R,t,back=None,bfu=None,threshold=1.5,seed=None):
    cv2.setRNGSeed(42+frame if seed is None else seed)
    n=int(mask.sum());row={'stage':stage,'frame':frame,'correspondences':n,'threshold_px':threshold,'solver_success':False,'valid_pose':False,'add':None,'add_pass':False,'rotation_deg':None,'translation':None,'reprojection_mean_px':None,'reprojection_inliers_mean_px':None,'inliers_actual':0,'inlier_ratio_actual':0.,'inliers_source_reported':0,'failure':None}
    PROXY.points=None;PROXY.raw_inliers=None
    try:
        args=(mask,front,uv,K) if back is None else (mask,back,uv,K,front)
        result=SOLVE(args,pnp_op=2,reprojectionError=threshold,iterationsCount=150,bfu=bfu)
        row['solver_success']=bool(result['success']);rp=np.asarray(result['rot']);tp=np.asarray(result['tvecs']);xc=V@rp.T+tp.reshape(1,3)
        if not result['success']:row['failure']='PnP 반환 실패'
        elif not(np.isfinite(rp).all() and np.isfinite(tp).all()):row['failure']='비유한 R/t'
        elif np.min(xc[:,2])<=0:row['failure']='CAD 음수 깊이'
        else:
            add=float(np.linalg.norm(xc-(V@R.T+t.reshape(1,3)),axis=1).mean());re=float(np.degrees(np.arccos(np.clip((np.trace(rp@R.T)-1)/2,-1,1))));te=float(np.linalg.norm(tp.reshape(3)-t.reshape(3)))
            pp=PROXY.points;uu=PROXY.uv;err=np.linalg.norm(project(pp,rp,tp,K)-uu,axis=1);ix=err<threshold
            raw=0 if PROXY.raw_inliers is None else len(PROXY.raw_inliers)
            row.update(valid_pose=True,add=add,add_pass=bool(add<DIAM*.1),rotation_deg=re,translation=te,reprojection_mean_px=float(err.mean()),reprojection_inliers_mean_px=float(err[ix].mean()) if ix.any() else None,inliers_actual=int(ix.sum()),inlier_ratio_actual=float(ix.mean()),inliers_ransac_raw=raw,inliers_source_reported=int(np.asarray(result['inliers']).shape[0]))
            RECS[(stage,frame)]={'R':rp.tolist(),'t':tp.reshape(3).tolist()}
    except Exception as e:row['failure']=type(e).__name__+': '+str(e)
    ALL.append(row);return row

def overlay(stage,frame,row):
    if (stage,frame) not in RECS:return
    im=cv2.imread(str(S.SCENE/'rgb'/f'{frame:06d}.jpg'));r=RECS[(stage,frame)];k=np.array(CAM[str(frame)]['cam_K']).reshape(3,3)
    for color,rt in [((0,255,0),(np.array(GT[str(frame)][0]['cam_R_m2c']).reshape(3,3),np.array(GT[str(frame)][0]['cam_t_m2c']))),((255,0,255),(np.array(r['R']),np.array(r['t'])))]:
        uv=project(V,*rt,k);polys=[np.clip(uv[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in F];cv2.polylines(im,polys,True,color,1,cv2.LINE_AA)
    cv2.putText(im,f'{stage} {frame:06d} GT=green / PnP=magenta ADD={row["add"]:.4f}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    path=ROOT/'figures'/f'{stage}_{frame:06d}_overlay.png';assert not path.exists();cv2.imwrite(str(path),im)

def oracle():
    # 全 0..255 RGB axis exhaustive original encode/decode on CPU, no network.
    q=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2);d,_=decoded(q)
    write(ROOT/'checks/hcce_exhaustive.json',{'codes_per_axis':256,'max_abs_uint8_error':float(np.abs(d-q).max()),'threshold':'>0.5','axis_order':'cv2 channel0=X,1=Y,2=Z'})
    for frame in VAL:
        guard();g=GT[str(frame)][0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c']);K=np.array(CAM[str(frame)]['cam_K']).reshape(3,3)
        uv=project(V,R,t,K);row=pose('A0_exact_CAD',frame,np.ones((1,len(V)),bool),V[None],uv[None],K,R,t);ORACLE.append(row)
        full=cv2.imread(str(BASE/'labels/front'/f'{frame:06d}_000000.png'));vis=cv2.imread(str(S.SCENE/'mask_visib'/f'{frame:06d}_000000.png'),0)>0;valid=full.any(axis=2);support=vis&valid
        yy,xx=np.nonzero(support);raw=full[support][None];dd,code=decoded(raw);xyz=dd/255*SIZE+LO;uv=np.stack([xx,yy],-1)[None]
        direct=raw.astype(float)/255*SIZE+LO
        row=pose('A1_full_HCCE',frame,np.ones((1,len(xx)),bool),xyz,uv,K,R,t);row['hcc_uint8_max_error']=float(np.abs(dd-raw).max());row['decode_vs_direct_xyz_max']=float(np.abs(xyz-direct).max());ORACLE.append(row)
        # Original affine crop onto 128 target. Exact inverse-affine grid matches inference convention x/N*w+x0.
        box=PAD(np.array(GI[str(frame)][0]['bbox_visib'],float),1.5);qc=CROP(full,box,128,cv2.INTER_NEAREST);mc=CROP(vis.astype(np.uint8),box,128,cv2.INTER_NEAREST)>0;vc=CROP(valid.astype(np.uint8),box,128,cv2.INTER_NEAREST)>0
        dc,_=decoded(qc);xyz=dc/255*SIZE+LO;grid=np.stack(np.meshgrid(np.arange(128),np.arange(128)),axis=-1);coord=grid/128*box[2:]+box[:2]
        row=pose('A2_crop_GTmask',frame,mc,xyz,coord,K,R,t);row['invalid_label_pixels']=int((mc&~vc).sum());ORACLE.append(row)
        row=pose('A2_crop_valid',frame,mc&vc,xyz,coord,K,R,t);ORACLE.append(row)
        # Source-nearest pixel coordinate control separates resampling from label/PnP error.
        yy,xx=np.indices(vis.shape);pixel=np.stack([xx,yy],-1).astype(np.float32);nearest=CROP(pixel,box,128,cv2.INTER_NEAREST)
        row=pose('A2_nearest_pixel_control',frame,mc&vc,xyz,nearest,K,R,t);ORACLE.append(row)
        err=np.linalg.norm(project(xyz[mc&vc],R,t,K)-coord[mc&vc],axis=1)
        GEOM.append({'frame':frame,'K':K.tolist(),'bbox':box.tolist(),'image_size':[1280,720],'output_size':[128,128],'A2_source_pixel_shift_max_px':float(np.linalg.norm(nearest[mc&vc]-coord[mc&vc],axis=1).max()),'label_GT_pose_reprojection_mean_px':float(err.mean()),'label_GT_pose_reprojection_p95_px':float(np.quantile(err,.95)),'missing_label_GTmask_pixels':int((mc&~vc).sum())})
        if frame in (40,90):
            for st in ['A0_exact_CAD','A1_full_HCCE','A2_crop_GTmask']:overlay(st,frame,next(x for x in ORACLE if x['stage']==st and x['frame']==frame))
        print(json.dumps({'A_frame':frame,'A0_ADD':ORACLE[-5]['add'],'A1_ADD':ORACLE[-4]['add'],'A2_ADD':ORACLE[-3]['add']},ensure_ascii=False),flush=True)
    table('GT_XYZ_Oracle_PnP.csv',[x for x in ORACLE if x['stage']=='A0_exact_CAD']);table('GT_HCCE_Roundtrip.csv',[x for x in ORACLE if x['stage']!='A0_exact_CAD']);write(ROOT/'checks/geometry.json',GEOM);write(ROOT/'checks/oracle_results.json',ORACLE)

def xyz_panel(frame,rgb,gtf,gtb,pf,pb,mask):
    canvas=Image.new('RGB',(1280,300*2),'#202020');d=ImageDraw.Draw(canvas)
    for row,(gt,pr,title) in enumerate([(gtf,pf,'Front'),(gtb,pb,'Back')]):
        error=np.linalg.norm(pr-gt,axis=-1);heat=cv2.applyColorMap(np.clip(error/50*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);heat[~mask]=0
        a=np.clip((gt-LO)/SIZE*255,0,255).astype(np.uint8);b=np.clip((pr-LO)/SIZE*255,0,255).astype(np.uint8);a[~mask]=0;b[~mask]=0
        # Display diagnostic XYZ as X=R/Y=G/Z=B; these are not photorealistic renders.
        ims=[Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)),Image.fromarray(a),Image.fromarray(b),Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB))]
        for col,(im,label) in enumerate(zip(ims,['RGB','GT XYZ','Pred XYZ','3D Error 0..50 native units'])):
            canvas.paste(im.resize((300,256),Image.Resampling.NEAREST),(col*320,row*300+36));d.text((col*320+4,row*300+9),f'{frame:06d} {title} {label}',fill='white')
    canvas.save(ROOT/'figures'/f'XYZ_GT_prediction_{frame:06d}.png')

def predictions():
    guard(True);samples,train,val=S.dataset();module=S.explicit_module();device='cuda'
    net=S.build_network(module,torch.tensor(LO,dtype=torch.float32,device=device),torch.tensor(SIZE,dtype=torch.float32,device=device)).cuda()
    ck=torch.load(TRAIN/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);assert ck['epoch']==155;net.load_state_dict(ck['model_state_dict']);net.eval()
    raw={}
    def hook(m,args,out):raw['mask']=out[0].detach().clone();raw['code']=out[1].detach().clone()
    handle=net.net.register_forward_hook(hook)
    for frame in val:
        guard(True);s=samples[frame];x=s['tensors'][0][None].cuda();box=s['bbox'][None].cuda()
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
        pf=out['pred_front_code_obj'][0].float().cpu().numpy();pb=out['pred_back_code_obj'][0].float().cpu().numpy();uv=out['coord_2d_image'][0].float().cpu().numpy();pm=out['pred_mask'][0].cpu().numpy()>0
        logits=raw['code'][0].float().cpu().numpy().transpose(1,2,0);ml=raw['mask'][0,0].float().cpu().numpy();maskconf=1/(1+np.exp(-ml));bitconf=np.minimum(np.abs(np.tanh(logits[...,:24]/2)).mean(axis=-1),np.abs(np.tanh(logits[...,24:]/2)).mean(axis=-1))
        box=s['bbox'].numpy();qf=CROP(cv2.imread(str(BASE/'labels/front'/f'{frame:06d}_000000.png')),box,128,cv2.INTER_NEAREST);qb=CROP(cv2.imread(str(BASE/'labels/back'/f'{frame:06d}_000000.png')),box,128,cv2.INTER_NEAREST)
        gf=qf.astype(float)/255*SIZE+LO;gb=qb.astype(float)/255*SIZE+LO;gm=s['tensors'][1].numpy()>.5;valid=qf.any(-1)&qb.any(-1)
        masks={'GTmask':gm&valid,'PredMask_validLabel':pm&valid,'intersection':gm&pm&valid,'GTmask_eroded':(cv2.erode(gm.astype(np.uint8),np.ones((3,3),np.uint8))>0)&valid}
        for branch,gt,pr,q,bit in [('front',gf,pf,qf,logits[...,:24]),('back',gb,pb,qb,logits[...,24:])]:
            enc=ENC(None,q);truth=enc>.5;predbit=bit>0
            for scope,mask in masks.items():
                n=int(mask.sum());e=pr[mask]-gt[mask];dist=np.linalg.norm(e,axis=1)
                row={'frame':frame,'branch':branch,'scope':scope,'n':n,'gt_mask_pixels':int(gm.sum()),'pred_mask_pixels':int(pm.sum()),'invalid_label_in_gt':int((gm&~valid).sum()),'invalid_label_in_pred':int((pm&~valid).sum()),'xyz_l2_mean':float(dist.mean()) if n else None,'xyz_l2_median':float(np.median(dist)) if n else None,'xyz_l2_p95':float(np.quantile(dist,.95)) if n else None,'bit_error_rate':float(np.mean(predbit[mask]!=truth[mask])) if n else None,'soft_hcce_MAE':float(np.abs(np.tanh(bit[mask])-(enc[mask]*2-1)).mean()) if n else None}
                for j,a in enumerate('xyz'):row[a+'_MAE']=float(np.abs(e[:,j]).mean()) if n else None
                ERRORS.append(row)
        if frame in (40,90):xyz_panel(frame,s['rgb_crop'],gf,gb,pf,pb,gm&valid)
        K=s['K'];R=s['R'];t=s['t'].reshape(3);n=int(pm.sum());rank=np.argsort(-(maskconf*bitconf)[pm],kind='stable');flat=np.flatnonzero(pm)
        candidates=[('front',pm,pf,None,None,1.5),('back',pm,pf,pb,'b',1.5),('mixed_bf',pm,pf,pb,'bf',1.5),('mixed_bfu',pm,pf,pb,'bfu',1.5),('front_GTmask_Oracle',gm,pf,None,None,1.5),('front_intersection_Oracle',pm&gm&valid,pf,None,None,1.5),('GTxyz_PredMask_Oracle',pm,gf,None,None,1.5),('front_threshold3',pm,pf,None,None,3.),('front_threshold5',pm,pf,None,None,5.)]
        for count in (256,1024):
            mask=np.zeros(pm.shape,bool);mask.reshape(-1)[flat[rank[:min(count,n)]]]=True;candidates.append((f'front_topConfidence{count}',mask,pf,None,None,1.5))
        for name,mask,front,back,mode,thr in candidates:
            guard();row=pose(name,frame,mask,front,uv,K,R,t,back,mode,thr);CAND.append(row)
            if name=='front' and frame in (40,90):overlay(name,frame,row)
        PRED[frame]={'mask_iou':float(np.count_nonzero(pm&gm)/np.count_nonzero(pm|gm)),'front_xyz_error':next(x['xyz_l2_mean'] for x in ERRORS if x['frame']==frame and x['branch']=='front' and x['scope']=='GTmask')}
        print(json.dumps({'B_C_frame':frame,'front_GTmask_error':PRED[frame]['front_xyz_error'],'front_ADD':next(x['add'] for x in CAND if x['stage']=='front' and x['frame']==frame)},ensure_ascii=False),flush=True)
    handle.remove();del net;torch.cuda.empty_cache();table('Predicted_XYZ_error.csv',ERRORS);table('PnP_candidates_per_frame.csv',CAND);write(ROOT/'checks/prediction_results.json',{'errors':ERRORS,'candidates':CAND,'frames':PRED})


def summarize():
    summaries=[]
    for name in dict.fromkeys(r['stage'] for r in ALL):
        rows=[r for r in ALL if r['stage']==name];good=[r for r in rows if r['valid_pose']]
        s={'stage':name,'n':len(rows),'valid':len(good),'ADD_pass':sum(r['add_pass'] for r in rows),'ADD_rate':sum(r['add_pass'] for r in rows)/len(rows),'inliers_mean':float(np.mean([r['inliers_actual'] for r in rows])),'inlier_ratio_mean':float(np.mean([r['inlier_ratio_actual'] for r in rows])),'source_inlier_count_mismatches':sum(r['inliers_actual']!=r['inliers_source_reported'] for r in rows)}
        for key in ['add','rotation_deg','translation','reprojection_mean_px']:
            arr=[r[key] for r in good if r[key] is not None];s[key+'_mean']=float(np.mean(arr)) if arr else None;s[key+'_median']=float(np.median(arr)) if arr else None
        summaries.append(s)
    table('stage_summary.csv',summaries);table('PnP_candidate_comparison.csv',[r for r in summaries if not r['stage'].startswith('A')]);write(ROOT/'checks/summary.json',{'stages':summaries,'elapsed_seconds':time.monotonic()-START});write(ROOT/'checks/returned_poses.json',[{'stage':a,'frame':b,**r} for (a,b),r in RECS.items()])

def main():
    write(ROOT/'checks/run_info.json',{'start':datetime.datetime.now(datetime.timezone.utc).isoformat(),'budget_minutes':60,'GPU_training':False,'checkpoint':str(TRAIN/'checkpoints/best_val_loss.pt'),'checkpoint_sha256':hashlib.sha256((TRAIN/'checkpoints/best_val_loss.pt').read_bytes()).hexdigest(),'image_coordinates':'full1280x720,K unchanged; crop inverse x=x0+w*j/128,y=y0+h*i/128','native_unit':'PBR metadata mm interpretation; no physical measurement','candidate_confidence':'mean bit margin from pre-threshold raw logits times mask probability; NOT calibrated confidence'})
    oracle()
    try:predictions()
    except Exception as e:write(ROOT/'checks/prediction_failure.json',{'failure':str(e),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True)
    summarize()
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'checks/fatal_failure.json',{'failure':str(e),'traceback':traceback.format_exc()});raise
