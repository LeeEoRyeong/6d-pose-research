import itertools,json,time,traceback
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image,ImageDraw
from support import RUN,BASE,SCENE,CONFIG,load_json,write_json,explicit_module,build_network,original_function,guard,OtherGpuJob,Deadline,dataset,batch

def mesh():
    lines=(BASE/'data/models/obj_000004.ply').read_text().splitlines();at=lines.index('end_header')+1
    n=int(next(x.split()[-1] for x in lines[:at] if x.startswith('element vertex ')));nf=int(next(x.split()[-1] for x in lines[:at] if x.startswith('element face ')))
    return np.array([list(map(float,x.split()[:3])) for x in lines[at:at+n]]),np.array([list(map(int,x.split()[1:])) for x in lines[at+n:at+n+nf]])

def aggregate(rows):
    good=[r for r in rows if r['valid_pose']]
    def stats(key):
        a=[r[key] for r in good]
        return {'n':len(a),'mean':float(np.mean(a)) if a else None,'median':float(np.median(a)) if a else None,'p95':float(np.quantile(a,.95)) if a else None}
    return {'n':len(rows),'mask_iou_mean':float(np.mean([r['mask_iou'] for r in rows])),'opencv_success_count':sum(r['opencv_success'] for r in rows),'pnp_valid_count':len(good),'pnp_valid_rate':len(good)/len(rows),'add_0_1d_pass_count':sum(r['add_0_1d_pass'] for r in rows),'add_0_1d_rate_all_frames':sum(r['add_0_1d_pass'] for r in rows)/len(rows),'add_error_conditional':stats('add_error'),'rotation_deg_conditional':stats('rotation_deg'),'translation_native_units_conditional':stats('translation_error'),'failures_counted_in_denominator':True,'error_distribution_scope':'유효 PnP만, n을 함께 표시; 실패 오차를0으로 채우지 않음'}

def mask_figure(tag,rows,samples,pred_masks):
    chosen=[40,45,90,95];canvas=Image.new('RGB',(1024,288*4),'#202020');draw=ImageDraw.Draw(canvas)
    lookup={r['frame']:r for r in rows}
    for row,frame in enumerate(chosen):
        s=samples[frame];rgb=Image.fromarray(cv2.cvtColor(s['rgb_crop'],cv2.COLOR_BGR2RGB));gt=s['tensors'][1].numpy()>.5;pr=pred_masks[frame]
        color=np.zeros((128,128,3),dtype=np.uint8);color[gt&pr]=[0,220,60];color[gt&~pr]=[240,40,40];color[~gt&pr]=[40,70,255]
        ims=[rgb,Image.fromarray(gt.astype(np.uint8)*255),Image.fromarray(pr.astype(np.uint8)*255),Image.fromarray(color)]
        for col,(im,title) in enumerate(zip(ims,['RGB crop','GT Mask','Prediction','Green=TP Red=FN Blue=FP'])):
            canvas.paste(im.resize((256,256),Image.Resampling.NEAREST),(col*256,row*288+32));draw.text((col*256+3,row*288+4),f'{frame:06d} {title}',fill='white')
        draw.text((520,row*288+18),f'IoU={lookup[frame]["mask_iou"]:.4f}',fill='white')
    canvas.save(RUN/'figures'/f'{tag}_mask_gt_prediction.png')

def pose_overlay(tag,record,samples,vertices,faces):
    frame=record['frame'];s=samples[frame];bgr=cv2.imread(str(SCENE/'rgb'/f'{frame:06d}.jpg'))
    gtmask=cv2.imread(str(SCENE/'mask'/f'{frame:06d}_000000.png'),0)
    contours,_=cv2.findContours(gtmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(bgr,contours,-1,(0,255,0),1)
    R=np.array(record['pred_R']);t=np.array(record['pred_t']).reshape(1,3);xyz=vertices@R.T+t;uv=xyz@s['K'].T;uv=uv[:,:2]/uv[:,2:]
    polygons=[np.clip(uv[f],-100000,100000).astype(np.int32).reshape(-1,1,2) for f in faces]
    cv2.polylines(bgr,polygons,True,(255,0,255),1,cv2.LINE_AA)
    cv2.putText(bgr,f'{tag} frame {frame:06d} | GT contour green / PnP CAD wire magenta',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr,f'ADD={record["add_error"]:.3f}  RE={record["rotation_deg"]:.2f}deg  TE={record["translation_error"]:.3f}',(12,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    path=RUN/'figures'/f'{tag}_pnp_overlay_{frame:06d}.png';assert not path.exists();assert cv2.imwrite(str(path),bgr)
    return str(path)

@torch.inference_mode()
def evaluate_checkpoint(tag,path,net,samples,val,solve,vertices,faces,diameter):
    ck=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict(ck['model_state_dict'],strict=True);net.eval()
    rows=[];masks={}
    for at in range(0,len(val),4):
        guard();ids=val[at:at+4];x,m,f,b=batch(samples,ids);boxes=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
        with torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(x,boxes)
        predicted=out['pred_mask'].cpu().numpy()>0;coords=out['coord_2d_image'].float().cpu().numpy();front=out['pred_front_code_obj'].float().cpu().numpy()
        for j,frame in enumerate(ids):
            guard();gt=samples[frame]['tensors'][1].numpy()>.5;pr=predicted[j];masks[frame]=pr
            row={'frame':frame,'mask_iou':float(np.count_nonzero(pr&gt)/max(np.count_nonzero(pr|gt),1)),'predicted_mask_pixels':int(pr.sum()),'opencv_success':False,'valid_pose':False,'add_0_1d_pass':False,'add_error':None,'rotation_deg':None,'translation_error':None,'failure_reason':None}
            try:
                cv2.setRNGSeed(42+frame)
                result=solve((pr,front[j],coords[j],samples[frame]['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
                row['opencv_success']=bool(result['success'])
                R=result['rot'];t=result['tvecs'];camera=vertices@R.T+t.reshape(1,3)
                if not result['success']:row['failure_reason']='source solver success=False'
                elif not (np.isfinite(R).all() and np.isfinite(t).all()):row['failure_reason']='non-finite pose'
                elif camera[:,2].min()<=0:row['failure_reason']='CAD 일부가 카메라 뒤에 있음'
                else:
                    Rgt=samples[frame]['R'];tgt=samples[frame]['t'];truth=vertices@Rgt.T+tgt.reshape(1,3)
                    add=float(np.linalg.norm(camera-truth,axis=1).mean());rotation=float(np.degrees(np.arccos(np.clip((np.trace(R@Rgt.T)-1)/2,-1,1))));translation=float(np.linalg.norm(t.reshape(3,1)-tgt))
                    assert all(np.isfinite(v) for v in (add,rotation,translation))
                    row.update(valid_pose=True,add_error=add,rotation_deg=rotation,translation_error=translation,add_0_1d_pass=add<.1*diameter,pred_R=R.tolist(),pred_t=t.reshape(3).tolist())
            except Exception as exc:row['failure_reason']=type(exc).__name__+': '+str(exc)
            rows.append(row)
    mask_figure(tag,rows,samples,masks)
    good=[r for r in rows if r['valid_pose']];preferred=[r for r in good if r['frame'] in (40,90)]
    examples=(preferred+[r for r in good if r not in preferred])[:2]
    overlays=[pose_overlay(tag,r,samples,vertices,faces) for r in examples]
    result={'tag':tag,'checkpoint':str(path),'epoch':ck['epoch'],'optimizer_steps':ck['optimizer_steps'],'metrics':aggregate(rows),'rows':rows,'overlays':overlays,'overlay_selection':'우선40/90; 유효 Pose 없으면 순서상 최초2개. 오차가 작은 표본을 고른 것이 아님','scope':'GT visible BBox pad1.5 oracle crop; front-only source PnP RANSAC; PBR val20, detector/real evaluation 아님'}
    write_json(RUN/'checks'/f'{tag}_pose.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2));return result

def main():
    guard(force=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    summary=load_json(RUN/'checks/training_summary.json');samples,train,val=dataset();info=load_json(BASE/'data/models/models_info.json')['4']
    module=explicit_module();net=build_network(module,torch.tensor([info['min_x'],info['min_y'],info['min_z']],device='cuda'),torch.tensor([info['size_x'],info['size_y'],info['size_z']],device='cuda')).cuda()
    solve=original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});v,f=mesh();results={}
    for tag,key in [('best_loss','best_val_loss'),('best_mask','best_val_mask_iou_secondary'),('last','last')]:results[tag]=evaluate_checkpoint(tag,Path(summary[key]['checkpoint']),net,samples,val,solve,v,f,info['diameter'])
    write_json(RUN/'checks/validation_pose_summary.json',{k:{x:y for x,y in v.items() if x!='rows'} for k,v in results.items()})
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(RUN/'checks/pose_evaluation_failure.json',{'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()});raise
