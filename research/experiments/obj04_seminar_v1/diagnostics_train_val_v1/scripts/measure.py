"""Train80/Val20 동일조건 좌표·HCCE plane·PnP 진단. 재학습 없음."""
from pathlib import Path
import sys,json,time,datetime,subprocess,os,hashlib,csv,traceback
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;PREV=BASE/'diagnostics_v1'
sys.path.insert(0,str(PREV/'scripts'));import diagnose as D
D.ROOT=ROOT
START=time.monotonic();STOP=START+35*60

def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def gpu_guard():
    if time.monotonic()>STOP:raise RuntimeError('35분 경과: 계산 중단, 보고서 우선')
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    other=[r for r in out.splitlines() if r.strip() and int(r.split(',')[0])!=os.getpid()]
    if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def decode_axis(bits):
    binary=np.bitwise_xor.accumulate(bits.astype(np.uint8),axis=-1)
    return (binary.astype(np.int32)*np.array([128,64,32,16,8,4,2,1])).sum(axis=-1)
def scalar(v):return float(v) if np.isfinite(v) else None

def main():
    gpu_guard();torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoint=D.TRAIN/'checkpoints/best_val_loss.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limit_minutes':45,'calculation_stop_minutes':35,'checkpoint':str(checkpoint),'checkpoint_sha256':sha,'epoch':155,'batch_size':4,'AMP':'CUDA float16','crop':'GT visible BBox pad1.5, RGB256 / code128','scope':'GTmask & nonzero Front and Back label validity, same as diagnostics_v1 supplement','network_mode':'eval + inference_mode, no optimizer/no training','plane_counterfactual':'GT hard plane substitution, reference FP32 decoder, diagnostic Oracle only','collapse_flag':'aggregate pred positive <=0.01 or >=0.99; GT positive also shown','plane_confidence':'sigmoid(abs(raw)) heuristic, regression code outputs not calibrated probabilities'}
    save(ROOT/'checks/config.json',config)
    samples,train,val=D.S.dataset();assert len(train)==80 and len(val)==20;assert not set(train)&set(val);assert not {i//10 for i in train}&{i//10 for i in val}
    save(ROOT/'checks/split.json',{'train':train,'val':val,'train_blocks':sorted({i//10 for i in train}),'val_blocks':sorted({i//10 for i in val}),'real_frames':0})
    module=D.S.explicit_module();net=D.S.build_network(module,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ck['epoch']==155;net.load_state_dict(ck['model_state_dict'],strict=True);net.eval();raw={}
    def hook(m,args,out):raw['code']=out[1].detach().clone()
    handle=net.net.register_forward_hook(hook)
    xyz=[];planes=[];poses=[];frames=[];hist={};dist=[];reference_max_uint8_error=0.;amp_vs_fp32_max=0.
    edges={a:np.linspace(D.LO[i],D.LO[i]+D.SIZE[i],65) for i,a in enumerate('xyz')}
    # Fixed split order and ascending frame IDs keep Val batches identical to prior evaluation.
    for split,ids in [('train',train),('val',val)]:
        for at in range(0,len(ids),4):
            gpu_guard();batch=ids[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in batch]).cuda();box=torch.stack([samples[i]['bbox'] for i in batch]).cuda()
            with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
            for j,frame in enumerate(batch):
                s=samples[frame];bb=s['bbox'].numpy();gtmask=s['tensors'][1].numpy()>.5;predmask=out['pred_mask'][j].cpu().numpy()>0
                fullF=cv2.imread(str(BASE/'labels/front'/f'{frame:06d}_000000.png'));fullB=cv2.imread(str(BASE/'labels/back'/f'{frame:06d}_000000.png'))
                valid=D.CROP((fullF.any(-1)&fullB.any(-1)).astype(np.uint8),bb,128,cv2.INTER_NEAREST)>0
                qf=D.CROP(fullF,bb,128,cv2.INTER_NEAREST);qb=D.CROP(fullB,bb,128,cv2.INTER_NEAREST)
                mask=gtmask&valid;intersection=gtmask&predmask&valid;eroded=cv2.erode(gtmask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
                logits=raw['code'][j].float().cpu().numpy().transpose(1,2,0);uv=out['coord_2d_image'][j].float().cpu().numpy()
                iou=float(np.count_nonzero(gtmask&predmask)/max(np.count_nonzero(gtmask|predmask),1))
                cameraZangle=float(np.degrees(np.arccos(np.clip(s['R'][2,2],-1,1))));vis=float(D.GI[str(frame)][0]['visib_fract'])
                frames.append({'split':split,'frame':frame,'block':frame//10,'mask_iou':iou,'GTmask_pixels':int(gtmask.sum()),'valid_GTmask_pixels':int(mask.sum()),'pred_mask_pixels':int(predmask.sum()),'invalid_label_GTmask_pixels':int((gtmask&~valid).sum()),'object_Z_vs_camera_Z_deg':cameraZangle,'camera_tz':float(s['t'][2,0]),'distance':float(np.linalg.norm(s['t'])),'visib_fract':vis})
                for branch,q,key,lg in [('front',qf,'pred_front_code_obj',logits[...,:24]),('back',qb,'pred_back_code_obj',logits[...,24:])]:
                    pred=out[key][j].float().cpu().numpy();gt=q.astype(float)/255*D.SIZE+D.LO;target=D.ENC(None,q);truth=target>.5;bits=lg>0
                    # Verify efficient NumPy reference against the original torch decoder each frame.
                    refq=np.stack([decode_axis(bits[...,k*8:(k+1)*8]) for k in range(3)],axis=-1)
                    with torch.inference_mode():srcq=D.DEC(type('DecodeState',(),{'powers':None})(),torch.tensor(bits[None],dtype=torch.float32))[0].numpy()
                    reference_max_uint8_error=max(reference_max_uint8_error,float(np.max(np.abs(refq-srcq))))
                    refxyz=refq.astype(float)/255*D.SIZE+D.LO;amp_vs_fp32_max=max(amp_vs_fp32_max,float(np.max(np.abs(pred-refxyz))))
                    for scope,ma in [('GTmask',mask),('PredMask_validLabel',predmask&valid),('intersection',intersection),('GTmask_eroded',eroded)]:
                        delta=pred[ma]-gt[ma];length=np.linalg.norm(delta,axis=-1);n=int(ma.sum())
                        rr={'split':split,'frame':frame,'block':frame//10,'branch':branch,'scope':scope,'n':n,'mask_iou':iou,'xyz_l2':float(length.mean()),'xyz_l2_median':float(np.median(length)),'xyz_l2_p95':float(np.quantile(length,.95)),'x_MAE':float(np.abs(delta[:,0]).mean()),'y_MAE':float(np.abs(delta[:,1]).mean()),'z_MAE':float(np.abs(delta[:,2]).mean()),'raw_target_MAE':float(np.abs(lg[ma]-(target[ma]*2-1)).mean()),'hard_bit_error':float(np.mean(bits[ma]!=truth[ma]))}
                        xyz.append(rr)
                    for axis,a in enumerate('xyz'):
                        valsgt=gt[mask,axis];valspred=pred[mask,axis]
                        dist.append({'split':split,'frame':frame,'branch':branch,'axis':a,'n':len(valsgt),'GT_mean':float(valsgt.mean()),'GT_std':float(valsgt.std()),'GT_min':float(valsgt.min()),'GT_max':float(valsgt.max()),'GT_q25':float(np.quantile(valsgt,.25)),'GT_median':float(np.median(valsgt)),'GT_q75':float(np.quantile(valsgt,.75)),'pred_mean':float(valspred.mean()),'pred_std':float(valspred.std())})
                        k=f'{split}_{branch}_{a}';h=hist.setdefault(k,{'GT':np.zeros(64,np.int64),'pred':np.zeros(64,np.int64)});h['GT']+=np.histogram(valsgt,edges[a])[0];h['pred']+=np.histogram(valspred,edges[a])[0]
                        bits_axis=bits[...,axis*8:(axis+1)*8];gt_axis=truth[...,axis*8:(axis+1)*8];n=int(mask.sum());referr=np.abs(refxyz[mask,axis]-gt[mask,axis]);actualerr=np.abs(pred[mask,axis]-gt[mask,axis])
                        for plane in range(8):
                            gtbit=gt_axis[mask,plane];prbit=bits_axis[mask,plane];ll=lg[mask,axis*8+plane];prob=1/(1+np.exp(-np.clip(ll,-80,80)));wrong=prbit!=gtbit
                            fixed=bits_axis.copy();fixed[...,plane]=gt_axis[...,plane];fixedq=decode_axis(fixed);fixedcoord=fixedq.astype(float)/255*D.SIZE[axis]+D.LO[axis];fixederr=np.abs(fixedcoord[mask]-gt[mask,axis])
                            pos=int(gtbit.sum());neg=n-pos;tp=int(np.count_nonzero(gtbit&prbit));tn=int(np.count_nonzero(~gtbit&~prbit));confidence=np.maximum(prob,1-prob)
                            rr={'split':split,'frame':frame,'branch':branch,'axis':a,'plane':plane,'n':n,'decoder_binary_weight':2**(7-plane),'maximum_cascade_code_change':2**(8-plane)-1,'maximum_cascade_native_change':float((2**(8-plane)-1)/255*D.SIZE[axis]),'hard_bit_error':float(wrong.mean()),'GT_positive_fraction':float(gtbit.mean()),'pred_positive_fraction':float(prbit.mean()),'GT_positive_count':pos,'GT_negative_count':neg,'TP':tp,'TN':tn,'raw_mean':float(ll.mean()),'raw_std':float(ll.std()),'abs_raw_margin_mean':float(np.abs(ll).mean()),'sigmoid_mean':float(prob.mean()),'confidence_mean':float(confidence.mean()),'confidence_wrong_mean':float(confidence[wrong].mean()) if wrong.any() else None,'wrong_count':int(wrong.sum()),'raw_target_MAE':float(np.abs(ll-(target[mask,axis*8+plane]*2-1)).mean()),'axis_actual_MAE':float(actualerr.mean()),'axis_reference_MAE':float(referr.mean()),'axis_MAE_after_GT_plane_Oracle':float(fixederr.mean()),'Oracle_MAE_reduction':float(referr.mean()-fixederr.mean()),'axis_error_if_bit_wrong':float(actualerr[wrong].mean()) if wrong.any() else None,'axis_error_if_bit_right':float(actualerr[~wrong].mean()) if (~wrong).any() else None}
                            planes.append(rr)
                    gpu_guard();pose=D.pose(branch,frame,predmask,pred,uv,s['K'],s['R'],s['t'].reshape(3));pose.update(split=split,branch=branch,block=frame//10);poses.append(pose)
                print(json.dumps({'split':split,'frame':frame,'mask_iou':iou,'front_xyz':next(r['xyz_l2'] for r in xyz if r['frame']==frame and r['branch']=='front' and r['scope']=='GTmask'),'ADD':poses[-2]['add']},ensure_ascii=False),flush=True)
    handle.remove();del net;torch.cuda.empty_cache()
    assert reference_max_uint8_error==0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
    csvout(ROOT/'tables/xyz_per_frame.csv',xyz);csvout(ROOT/'tables/plane_per_frame.csv',planes);csvout(ROOT/'tables/pose_per_frame.csv',poses);csvout(ROOT/'tables/frame_metadata.csv',frames);csvout(ROOT/'tables/GT_distribution_per_frame.csv',dist)
    save(ROOT/'checks/histograms.json',{'edges':{a:v.tolist() for a,v in edges.items()},'counts':{k:{q:v.tolist() for q,v in h.items()} for k,h in hist.items()},'normalization':'plot divides counts by sum within split, branch and GT/pred source'})
    # Unweighted frame means primary; n-weighted alternatives retained and explicitly named.
    xa=[]
    for split in ('train','val'):
        for branch in ('front','back'):
            for scope in ('GTmask','PredMask_validLabel','intersection','GTmask_eroded'):
                rr=[r for r in xyz if r['split']==split and r['branch']==branch and r['scope']==scope];o={'split':split,'branch':branch,'scope':scope,'frames':len(rr),'pixels':sum(r['n'] for r in rr)}
                for key in ('mask_iou','xyz_l2','x_MAE','y_MAE','z_MAE','raw_target_MAE','hard_bit_error'):
                    o[key]=float(np.mean([r[key] for r in rr]));o[key+'_pixel_weighted']=float(np.average([r[key] for r in rr],weights=[r['n'] for r in rr]))
                xa.append(o)
    csvout(ROOT/'train_val_xyz_comparison.csv',xa)
    pa=[]
    for split in ('train','val'):
        for branch in ('front','back'):
            for axis in 'xyz':
                for plane in range(8):
                    rr=[r for r in planes if r['split']==split and r['branch']==branch and r['axis']==axis and r['plane']==plane];o={'split':split,'branch':branch,'axis':axis,'plane':plane,'frames':len(rr),'pixels':sum(r['n'] for r in rr),'decoder_binary_weight':rr[0]['decoder_binary_weight'],'maximum_cascade_code_change':rr[0]['maximum_cascade_code_change'],'maximum_cascade_native_change':rr[0]['maximum_cascade_native_change']}
                    for key in ('hard_bit_error','GT_positive_fraction','pred_positive_fraction','raw_mean','raw_std','abs_raw_margin_mean','sigmoid_mean','confidence_mean','raw_target_MAE','axis_actual_MAE','axis_reference_MAE','axis_MAE_after_GT_plane_Oracle','Oracle_MAE_reduction'):
                        o[key]=float(np.mean([r[key] for r in rr]));o[key+'_pixel_weighted']=float(np.average([r[key] for r in rr],weights=[r['n'] for r in rr]))
                    o['GT_positive_total']=sum(r['GT_positive_count'] for r in rr);o['GT_negative_total']=sum(r['GT_negative_count'] for r in rr);o['TP_total']=sum(r['TP'] for r in rr);o['TN_total']=sum(r['TN'] for r in rr)
                    o['balanced_accuracy_micro']=(o['TP_total']/o['GT_positive_total']+o['TN_total']/o['GT_negative_total'])/2 if o['GT_positive_total'] and o['GT_negative_total'] else None
                    o['aggregate_pred_collapse_flag']=bool(o['pred_positive_fraction_pixel_weighted']<=.01 or o['pred_positive_fraction_pixel_weighted']>=.99);o['frames_pred_single_side']=sum(r['pred_positive_fraction']<=.01 or r['pred_positive_fraction']>=.99 for r in rr)
                    pa.append(o)
    csvout(ROOT/'hcce_plane_error.csv',pa)
    ps=[]
    for split in ('train','val'):
        for branch in ('front','back'):
            rr=[r for r in poses if r['split']==split and r['branch']==branch];good=[r for r in rr if r['valid_pose']];o={'split':split,'branch':branch,'frames':len(rr),'PnP_valid':len(good),'PnP_valid_rate':len(good)/len(rr),'ADD_pass':sum(r['add_pass'] for r in rr),'ADD_rate':sum(r['add_pass'] for r in rr)/len(rr),'inlier_ratio':float(np.mean([r['inlier_ratio_actual'] for r in rr])),'inliers_mean':float(np.mean([r['inliers_actual'] for r in rr]))}
            for key in ('add','rotation_deg','translation','reprojection_mean_px'):
                o[key+'_mean']=float(np.mean([r[key] for r in good])) if good else None;o[key+'_median']=float(np.median([r[key] for r in good])) if good else None
            ps.append(o)
    csvout(ROOT/'train_val_pose_metrics.csv',ps)
    block=[]
    for b in range(10):
        for branch in ('front','back'):
            rr=[r for r in xyz if r['block']==b and r['branch']==branch and r['scope']=='GTmask'];pp=[r for r in poses if r['block']==b and r['branch']==branch]
            block.append({'block':b,'split':rr[0]['split'],'branch':branch,'frames':len(rr),'xyz_l2':float(np.mean([r['xyz_l2'] for r in rr])),'z_MAE':float(np.mean([r['z_MAE'] for r in rr])),'mask_iou':float(np.mean([r['mask_iou'] for r in rr])),'ADD_pass':sum(r['add_pass'] for r in pp),'ADD_rate':sum(r['add_pass'] for r in pp)/len(pp)})
    csvout(ROOT/'tables/block_metrics.csv',block)
    relations=[]
    meta={f['frame']:f for f in frames}
    for split in ('train','val'):
        rr=[r for r in xyz if r['split']==split and r['branch']=='front' and r['scope']=='GTmask'];po={p['frame']:p for p in poses if p['split']==split and p['branch']=='front'};err=np.array([r['xyz_l2'] for r in rr])
        for label,values in [('MaskIoU',[r['mask_iou'] for r in rr]),('object_Z_angle',[meta[r['frame']]['object_Z_vs_camera_Z_deg'] for r in rr]),('visib_fract',[meta[r['frame']]['visib_fract'] for r in rr]),('tz',[meta[r['frame']]['camera_tz'] for r in rr]),('PnP_inlier_ratio',[po[r['frame']]['inlier_ratio_actual'] for r in rr]),('ADD',[po[r['frame']]['add'] for r in rr])]:
            good=[(a,b) for a,b in zip(err,values) if a is not None and b is not None];a,b=np.array(good).T
            relations.append({'split':split,'predictor':label,'n':len(good),'pearson_xyz_error':scalar(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None,'predictor_min':float(min(b)),'predictor_max':float(max(b)),'note':'descriptive; correlated generation blocks; no causal claim'})
    csvout(ROOT/'tables/relationships.csv',relations)
    # Verify Val reproducibility versus previous diagnosis framewise, not merely rounded summary.
    previous=D.S.load_json(PREV/'matched_batch4/checks/prediction_results.json')['candidates'];pd={r['frame']:r for r in previous if r['stage']=='front'}
    previous_errors=list(csv.DictReader((PREV/'supplement/tables/Predicted_XYZ_errors_raw_target.csv').open()));prevxyz={(int(r['frame']),r['branch'],r['scope']):r for r in previous_errors}
    maxadd=max(abs(r['add']-pd[r['frame']]['add']) for r in poses if r['split']=='val' and r['branch']=='front')
    maxxyz=max(abs(r['xyz_l2']-float(prevxyz[(r['frame'],r['branch'],r['scope'])]['xyz_l2_mean'])) for r in xyz if r['split']=='val')
    save(ROOT/'checks/completion.json',{'status':'PASS','frames':100,'train':80,'val':20,'epoch':ck['epoch'],'inference_batch':4,'AMP':'float16','retraining':False,'max_reference_decoder_uint8_difference':reference_max_uint8_error,'max_AMP_vs_FP32_xyz_element_difference':amp_vs_fp32_max,'Val_front_ADD_max_difference_vs_previous':maxadd,'Val_XYZ_max_difference_vs_previous':maxxyz,'elapsed_seconds':time.monotonic()-START,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256_unchanged':True,'xyz_summary':xa,'pose_summary':ps})
    print(json.dumps({'status':'PASS','pose_summary':ps,'GTmask_xyz':[x for x in xa if x['scope']=='GTmask'],'Val_ADD_delta':maxadd,'Val_xyz_delta':maxxyz},ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'checks/failure.json',{'reason':str(e),'traceback':traceback.format_exc()});raise
