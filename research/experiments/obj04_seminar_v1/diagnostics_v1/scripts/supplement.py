"""원본 batch4 재현에 맞춘 HCCE raw/bit·채널 오차, GT Back/ray·crop-K 대조 및 후보선택 진단."""
import diagnose as d
import json,os,collections
from pathlib import Path
import numpy as np,cv2,torch
R=d.ROOT;B=R/'matched_batch4';O=R/'supplement';O.mkdir()
for name in ('checks','tables','figures'): (O/name).mkdir()
d.ROOT=O
samples,train,val=d.S.dataset();m=d.S.explicit_module();d.guard(True)
net=d.S.build_network(m,torch.tensor(d.LO,dtype=torch.float32,device='cuda'),torch.tensor(d.SIZE,dtype=torch.float32,device='cuda')).cuda();ck=torch.load(d.TRAIN/'checkpoints/best_val_loss.pt',weights_only=True,map_location='cpu');net.load_state_dict(ck['model_state_dict']);net.eval();raw={}
def hook(m,a,o):raw['code']=o[1].detach().clone()
handle=net.net.register_forward_hook(hook)
bits=[];xyzrows=[];geometry=[];control=[];cache=O/'cache';cache.mkdir()
for at in range(0,len(val),4):
    d.guard(True);ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();box=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,box)
    for j,frame in enumerate(ids):
        s=samples[frame];bb=s['bbox'].numpy();gm=s['tensors'][1].numpy()>.5;pm=out['pred_mask'][j].cpu().numpy()>0;uv=out['coord_2d_image'][j].float().cpu().numpy();logits=raw['code'][j].float().cpu().numpy().transpose(1,2,0)
        ff=cv2.imread(str(d.BASE/'labels/front'/f'{frame:06d}_000000.png'));bf=cv2.imread(str(d.BASE/'labels/back'/f'{frame:06d}_000000.png'));fullvalid=ff.any(-1)&bf.any(-1);qf=d.CROP(ff,bb,128,cv2.INTER_NEAREST);qb=d.CROP(bf,bb,128,cv2.INTER_NEAREST);valid=d.CROP(fullvalid.astype(np.uint8),bb,128,cv2.INTER_NEAREST)>0
        arrays={};Rgt=s['R'];t=s['t'].reshape(3);K=s['K']
        for branch,q,pred,lg in [('front',qf,out['pred_front_code_obj'][j].float().cpu().numpy(),logits[...,:24]),('back',qb,out['pred_back_code_obj'][j].float().cpu().numpy(),logits[...,24:])]:
            gt=q.astype(float)/255*d.SIZE+d.LO;enc=d.ENC(None,q);truth=enc>.5;pb=lg>0;mask=gm&valid;arrays[branch+'_gt']=gt;arrays[branch+'_pred']=pred
            for axis in range(3):
                for bit in range(8):bits.append({'frame':frame,'branch':branch,'axis':'xyz'[axis],'bit_index':bit,'binary_weight':2**(7-bit),'n':int(mask.sum()),'hard_bit_error':float(np.mean(pb[mask,axis*8+bit]!=truth[mask,axis*8+bit])),'raw_target_MAE':float(np.abs(lg[mask,axis*8+bit]-(enc[mask,axis*8+bit]*2-1)).mean())})
            for name,ma in [('GTmask',gm&valid),('PredMask_validLabel',pm&valid),('intersection',gm&pm&valid),('GTmask_eroded',cv2.erode(gm.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid)]:
                e=pred[ma]-gt[ma];dist=np.linalg.norm(e,axis=1);x={'frame':frame,'branch':branch,'scope':name,'n':int(ma.sum()),'xyz_l2_mean':float(dist.mean()),'xyz_l2_median':float(np.median(dist)),'xyz_l2_p95':float(np.quantile(dist,.95)),'hard_bit_error_rate':float(np.mean(pb[ma]!=truth[ma])),'raw_output_target_MAE':float(np.abs(lg[ma]-(enc[ma]*2-1)).mean())}
                x.update({a+'_MAE':float(np.abs(e[:,k]).mean()) for k,a in enumerate('xyz')});xyzrows.append(x)
        np.savez_compressed(cache/f'{frame:06d}.npz',**arrays,gtmask=gm,predmask=pm,valid=valid,uv=uv,logits=logits)
        row=d.pose('A2_Back_GTmask_Oracle',frame,gm,arrays['back_gt'],uv,K,Rgt,t);control.append(row)
        # Analytical affine K' and crop grid use unchanged camera rays; no learned data.
        A=np.array([[128/bb[2],0,-128*bb[0]/bb[2]],[0,128/bb[3],-128*bb[1]/bb[3]],[0,0,1.]])
        grid=np.stack(np.meshgrid(np.arange(128),np.arange(128)),axis=-1).astype(np.float32)
        row=d.pose('A2_cropK_control_Oracle',frame,gm,arrays['front_gt'],grid,A@K,Rgt,t);control.append(row)
        # Independent Moller-Trumbore rays against actual CAD, sampled away from silhouettes.
        fullvis=cv2.imread(str(d.S.SCENE/'mask_visib'/f'{frame:06d}_000000.png'),0)>0;interior=(cv2.erode((fullvis&fullvalid).astype(np.uint8),np.ones((7,7),np.uint8))>0)
        yy,xx=np.nonzero(interior);ix=np.linspace(0,len(yy)-1,20,dtype=int);hits=d.renderer.independent_rays(d.V@Rgt.T+t,d.F,K,list(zip(yy[ix],xx[ix])),.5)
        errF=[];errB=[]
        for y,x,n,f in hits:
            gf=ff[y,x].astype(float)/255*d.SIZE+d.LO;gb=bf[y,x].astype(float)/255*d.SIZE+d.LO
            errF.append(abs((gf@Rgt.T+t)[2]-n));errB.append(abs((gb@Rgt.T+t)[2]-f))
        # Numerical RE floor due to finite precision GT rotation: project GT onto SO(3) for reporting control only.
        u,_,vt=np.linalg.svd(Rgt);proper=u@vt
        a0=next(a for a in d.S.load_json(R/'checks/returned_poses.json') if a['stage']=='A0_exact_CAD' and a['frame']==frame)
        re=float(np.degrees(np.arccos(np.clip((np.trace(np.array(a0['R'])@proper.T)-1)/2,-1,1))))
        geometry.append({'frame':frame,'independent_CAD_rays':len(hits),'front_cameraZ_error_mean':float(np.mean(errF)),'front_cameraZ_error_max':float(max(errF)),'back_cameraZ_error_mean':float(np.mean(errB)),'back_cameraZ_error_max':float(max(errB)),'GT_R_orthogonality_max':float(np.abs([REDACTED_EMAIL](3)).max()),'A0_RE_with_SO3_GT_deg':re,'note':'+0.5 ray, quantized label XYZ from original integer inverse projection; 20 interior pixels/frame'})
handle.remove();del net;torch.cuda.empty_cache()
d.table('HCCE_bit_channel_errors.csv',bits);d.table('Predicted_XYZ_errors_raw_target.csv',xyzrows);d.table('GT_back_cropK_control.csv',control);d.table('independent_CAD_ray_check.csv',geometry)
# Candidate selector ablations use only fixed 4 source modes. GT selection marked Oracle.
rows=list(d.S.load_json(B/'checks/prediction_results.json')['candidates']);selectors=[]
for frame in val:
    c=[r for r in rows if r['frame']==frame and r['stage'] in ('front','back','mixed_bf','mixed_bfu')]
    for name,key in [('source_maxReportedInliers',lambda r:r['inliers_source_reported']),('actual_maxUsedInliers',lambda r:r['inliers_actual']),('GT_minADD_Oracle',lambda r:-r['add'] if r['valid_pose'] else -1e30)]:
        chosen=max(c,key=key);selectors.append({**chosen,'stage':name,'chosen_candidate':chosen['stage']})
d.table('PnP_candidate_selectors.csv',selectors);d.ALL.extend(selectors);d.summarize()
# Aggregate per-frame means (unweighted); preserve per-pixel counts for weighted summaries.
errorstats=[]
for branch in ('front','back'):
    for scope in ('GTmask','PredMask_validLabel','intersection','GTmask_eroded'):
        rr=[r for r in xyzrows if r['branch']==branch and r['scope']==scope]
        out={'branch':branch,'scope':scope,'frames':len(rr),'pixels':sum(r['n'] for r in rr)}
        for k in ('xyz_l2_mean','xyz_l2_median','xyz_l2_p95','x_MAE','y_MAE','z_MAE','hard_bit_error_rate','raw_output_target_MAE'):
            out[k+'_frame_mean']=float(np.mean([r[k] for r in rr]));out[k+'_pixel_weighted']=float(np.average([r[k] for r in rr],weights=[r['n'] for r in rr]))
        errorstats.append(out)
d.write(O/'checks/error_summary.json',errorstats)
front=[r for r in xyzrows if r['branch']=='front' and r['scope']=='GTmask'];pose={r['frame']:r for r in rows if r['stage']=='front'}
a=np.array([r['xyz_l2_mean'] for r in front]);b=np.array([pose[r['frame']]['add'] for r in front]);rho=float(np.corrcoef(a,b)[0,1]);d.write(O/'checks/relationship.json',{'n':len(a),'pearson_FrontGTmask_error_vs_ADD':rho,'scope':'descriptive on same Val20, no causal inference'})
print(json.dumps({'xyz_error_summary':errorstats,'correlation':rho,'controls':d.S.load_json(O/'checks/summary.json')},ensure_ascii=False,indent=2))
