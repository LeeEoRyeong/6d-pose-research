"""저장 checkpoint만 PBR Val20 평가. 실사 이미지, 학습 optimizer 미사용."""
from pathlib import Path
import csv,json,sys,os,time,hashlib,traceback,subprocess,math,collections
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;OLD=BASE/'runs/baseline_seed42_20261103_214353/trained_only';ROT=BASE/'agent_run_v1';sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D
D.ROOT=ROOT;START=time.monotonic();STOP=START+15*60

def guard():
 if time.monotonic()>=STOP:raise TimeoutError('15분 경과: 분석 중지, 보고서 시간 확보')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def js(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def h(p):
 z=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):z.update(c)
 return z.hexdigest()
def scalar(x):
 x=float(x);return x if math.isfinite(x) else None

def history():
 out=[]
 for model,epochfile,stepfile,cfgfile in [('baseline',OLD/'logs/epoch_metrics.jsonl',OLD/'logs/train_steps.jsonl',OLD/'config.json'),('rotation',ROT/'logs/rotation_epochs.jsonl',ROT/'logs/rotation_steps.jsonl',ROT/'stage3/config.json')]:
  cfg=json.loads(cfgfile.read_text());stepstats=collections.defaultdict(lambda:{'losses':[],'amp_skip':0})
  for line in stepfile.open():
   x=json.loads(line);stepstats[x['epoch']]['losses'].append(x['loss']);stepstats[x['epoch']]['amp_skip']+=not x['optimizer_step_applied']
  for line in epochfile.open():
   x=json.loads(line);epoch=x['epoch'];ss=stepstats.get(epoch,{'losses':[],'amp_skip':0});out.append({'model':model,'epoch':epoch,'optimizer_steps':x['optimizer_steps'],'train_optimization_loss':x.get('train_optimization_loss'),'train_evaluation_loss':x['train_eval']['loss'],'validation_loss':x['val']['loss'],'train_mask_iou':x['train_eval']['mask_iou_mean'],'val_mask_iou':x['val']['mask_iou_mean'],'mask_loss_historical':'N/A','front_HCCE_loss_historical':'N/A','back_HCCE_loss_historical':'N/A','learning_rate_config':cfg['lr'],'learning_rate_step_log':'N/A','amp_skip_in_epoch':ss['amp_skip'],'amp_skip_since_previous_evaluation':sum(stepstats[e]['amp_skip'] for e in stepstats if (0 if epoch==0 else max(0,epoch-5))<e<=epoch),'source':'original epoch and step JSONL; component loss and per-step LR not logged'})
 csvout(ROOT/'epoch_loss_history.csv',out);return out

def main():
 hist=history();inventory=[]
 for model,path in [('baseline',OLD/'checkpoints'),('rotation',ROT/'stage3/checkpoints')]:
  for p in sorted(path.glob('*.pt')):inventory.append({'model':model,'file':str(p),'filename':p.name,'bytes':p.stat().st_size,'sha256':h(p),'status':'미평가','epoch':None,'optimizer_steps':None,'error':None})
 assert len(inventory)==9
 # Partial progress is flushed per checkpoint, so a time or GPU interruption cannot erase completed results.
 js(ROOT/'checks/initial_inventory.json',inventory)
 guard();torch.set_num_threads(2);cv2.setNumThreads(1);samples,train,val=D.S.dataset();assert len(val)==20
 labels={}
 for i in val:
  box=samples[i]['bbox'].numpy();f=cv2.imread(str(BASE/'labels/front'/f'{i:06d}_000000.png'));b=cv2.imread(str(BASE/'labels/back'/f'{i:06d}_000000.png'));valid=D.CROP((f.any(-1)&b.any(-1)).astype(np.uint8),box,128,cv2.INTER_NEAREST)>0
  labels[i]={'front':D.CROP(f,box,128,cv2.INTER_NEAREST).astype(float)/255*D.SIZE+D.LO,'back':D.CROP(b,box,128,cv2.INTER_NEAREST).astype(float)/255*D.SIZE+D.LO,'valid':valid}
 mod=D.S.explicit_module();net=D.S.build_network(mod,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda();net.eval();raw={}
 def hook(m,a,o):raw['mask']=o[0].detach().clone();raw['code']=o[1].detach().clone()
 hookref=net.net.register_forward_hook(hook);aggregate=[];geoms=[];frame_metrics=[];fail=None
 for idx,item in enumerate(inventory):
  try:
   guard();ck=torch.load(item['file'],map_location='cpu',weights_only=True);item['epoch']=int(ck['epoch']);item['optimizer_steps']=int(ck.get('optimizer_steps',0));net.load_state_dict(ck['model_state_dict'],strict=True);net.eval();criterion=mod.HccePose_Loss();metrics=[];components=[]
   for at in range(0,len(val),4):
    guard();ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();bbox=torch.stack([samples[i]['bbox'] for i in ids]).cuda();gm=torch.stack([samples[i]['tensors'][1] for i in ids]).cuda();front_gt=torch.stack([samples[i]['tensors'][2] for i in ids]).cuda();back_gt=torch.stack([samples[i]['tensors'][3] for i in ids]).cuda()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
     out=net.inference_batch(x,bbox);code=raw['code'];masklog=raw['mask'];parts=criterion(code[:,:24],code[:,24:],masklog,front_gt,back_gt,gm);total=D.S.loss_value(parts)
     # Keep weights from the just-computed predicted-mask history fixed while changing only gate for counterfactual HCCE contribution.
     predgate=(torch.sigmoid(masklog).detach().round().clamp(0,1));gtgate=(gm[:,None]>.5).to(predgate.dtype);cf=[]
     for branch in range(2):
      pred=code[:,branch*24:(branch+1)*24];target=front_gt if branch==0 else back_gt;weights=criterion.weight_front_error_ratio if branch==0 else criterion.weight_back_error_ratio;value=0.
      for axis in range(3):
       weight=torch.exp(torch.minimum(weights[axis],.51-weights[axis])*3);loss=((pred[:,axis*8:(axis+1)*8]-(target[:,axis*8:(axis+1)*8]*2-1)).abs()*gtgate).mean([0,2,3]);value=value+torch.sum(loss*weight)/torch.sum(weight)
      cf.append(float(value)*3)
    components.append({'n':len(ids),'loss_total':float(total),'mask_loss':float(parts['mask_loss']),'front_weighted_loss':float(parts['Front_L1Losses'].sum())*3,'back_weighted_loss':float(parts['Back_L1Losses'].sum())*3,'counterfactual_front_gt_gate_fixed_weights':cf[0],'counterfactual_back_gt_gate_fixed_weights':cf[1],'pred_gate_fraction':float(predgate.float().mean()),'gt_gate_fraction':float(gtgate.float().mean()),'GT_foreground_missed_fraction':float(((gtgate>0)&(predgate==0)).sum()/gtgate.sum().clamp_min(1))})
    bits=(torch.sigmoid(code)>.5).cpu().numpy().transpose(0,2,3,1)
    for j,frame in enumerate(ids):
     s=samples[frame];pm=out['pred_mask'][j].cpu().numpy()>0;gtmask=s['tensors'][1].numpy()>.5;ma=gtmask&labels[frame]['valid'];uv=out['coord_2d_image'][j].float().cpu().numpy();iou=float(np.count_nonzero(pm&gtmask)/max(np.count_nonzero(pm|gtmask),1));entry={'model':item['model'],'checkpoint':item['filename'],'epoch':item['epoch'],'frame':frame,'mask_iou':iou}
     for b,key,off,tgt in [('front','pred_front_code_obj',0,front_gt),('back','pred_back_code_obj',24,back_gt)]:
      pred=out[key][j].float().cpu().numpy();target=labels[frame][b];entry[b+'_xyz_l2']=float(np.linalg.norm(pred[ma]-target[ma],axis=-1).mean());truth=tgt[j].permute(1,2,0).cpu().numpy()>.5;entry[b+'_z0_hard_error']=float(np.mean(bits[j,...,off+16][ma]!=truth[...,16][ma]))
      pts=pred[pm].astype(np.float64);n=len(pts)
      if n:
       centered=pts-pts.mean(axis=0);cov=(centered.T@centered)/n;ev=np.maximum(np.linalg.eigvalsh(cov)[::-1],0.);v=ev/ev.sum() if ev.sum()>0 else np.zeros(3);vv=v[v>0];rank=float(np.exp(-(vv*np.log(vv)).sum())) if len(vv) else 0.;sd=pts.std(axis=0);unique=len(np.unique(pts,axis=0))
      else:ev=np.zeros(3);rank=0.;sd=np.zeros(3);unique=0
      po=D.pose('chk_'+b,frame,pm,pred,uv,s['K'],s['R'],s['t'].reshape(3))
      if po['valid_pose'] and not all(po[k] is not None and math.isfinite(float(po[k])) for k in ('add','rotation_deg','translation','reprojection_mean_px')):
       po.update(valid_pose=False,add_pass=False,add=None,rotation_deg=None,translation=None,reprojection_mean_px=None,failure='비유한 오류량')
      geo={'model':item['model'],'checkpoint':item['filename'],'epoch':item['epoch'],'frame':frame,'branch':b,'n':n,'unique_xyz':unique,'unique_fraction':unique/n if n else 0.,'x_std':float(sd[0]),'y_std':float(sd[1]),'z_std':float(sd[2]),'cov_eigenvalue_1':float(ev[0]),'cov_eigenvalue_2':float(ev[1]),'cov_eigenvalue_3':float(ev[2]),'effective_rank':rank,'PnP_valid':po['valid_pose'],'PnP_failure':po['failure'],'PnP_add':po['add'],'PnP_add_pass':po['add_pass'],'PnP_inlier_ratio':po['inlier_ratio_actual']};geoms.append(geo)
      for k in ('valid_pose','add','add_pass','rotation_deg','translation','inlier_ratio_actual'):entry[b+'_'+k]=po[k]
     metrics.append(entry);frame_metrics.append(entry)
   n=sum(x['n'] for x in components);weighted=lambda k:sum(x[k]*x['n'] for x in components)/n
   outrow={'model':item['model'],'checkpoint':item['filename'],'epoch':item['epoch'],'optimizer_steps':item['optimizer_steps'],'frames':len(metrics),'val_loss_recomputed':weighted('loss_total'),'mask_loss_recomputed':weighted('mask_loss'),'front_HCCE_weighted_loss_recomputed':weighted('front_weighted_loss'),'back_HCCE_weighted_loss_recomputed':weighted('back_weighted_loss'),'front_HCCE_GTgate_fixedweights_counterfactual':weighted('counterfactual_front_gt_gate_fixed_weights'),'back_HCCE_GTgate_fixedweights_counterfactual':weighted('counterfactual_back_gt_gate_fixed_weights'),'pred_mask_fraction':weighted('pred_gate_fraction'),'GT_mask_fraction':weighted('gt_gate_fraction'),'GT_foreground_missed_fraction_batchmean':weighted('GT_foreground_missed_fraction'),'mask_iou':float(np.mean([r['mask_iou'] for r in metrics]))}
   for b in ('front','back'):
    good=[r for r in metrics if r[b+'_valid_pose']];group=[x for x in geoms if x['model']==item['model'] and x['checkpoint']==item['filename'] and x['branch']==b];outrow.update({b+'_xyz_l2':float(np.mean([r[b+'_xyz_l2'] for r in metrics])),b+'_z0_hard_error':float(np.mean([r[b+'_z0_hard_error'] for r in metrics])),b+'_PnP_valid':len(good),b+'_ADD_pass':sum(r[b+'_add_pass'] for r in metrics),b+'_ADD_rate':sum(r[b+'_add_pass'] for r in metrics)/len(metrics)})
    for k in ('add','rotation_deg','translation'):outrow[b+'_'+k+'_mean_valid']=float(np.mean([r[b+'_'+k] for r in good])) if good else None
    outrow[b+'_x_std_mean']=float(np.mean([r['x_std'] for r in group]));outrow[b+'_y_std_mean']=float(np.mean([r['y_std'] for r in group]));outrow[b+'_z_std_mean']=float(np.mean([r['z_std'] for r in group]));outrow[b+'_unique_xyz_mean']=float(np.mean([r['unique_xyz'] for r in group]));outrow[b+'_effective_rank_mean']=float(np.mean([r['effective_rank'] for r in group]));outrow[b+'_inlier_ratio_mean']=float(np.mean([r['PnP_inlier_ratio'] for r in group]))
   aggregate.append(outrow);item['status']='평가 완료';del ck
   js(ROOT/'checks'/f'checkpoint_{idx:02d}.json',{'inventory':item,'summary':outrow,'components_by_batch':components});print(json.dumps(outrow,ensure_ascii=False),flush=True)
  except (TimeoutError,RuntimeError) as exc:
   if isinstance(exc,TimeoutError) or '다른 GPU' in str(exc):fail=str(exc);item['status']='중단';item['error']=str(exc);break
   item['status']='오류';item['error']=str(exc);print(traceback.format_exc(),flush=True)
  except Exception as exc:item['status']='오류';item['error']=str(exc);print(traceback.format_exc(),flush=True)
 hookref.remove();del net;torch.cuda.empty_cache();csvout(ROOT/'checkpoint_inventory.csv',inventory);csvout(ROOT/'checkpoint_comparison.csv',aggregate);csvout(ROOT/'correspondence_geometry.csv',geoms);csvout(ROOT/'tables/checkpoint_frame_metrics.csv',frame_metrics)
 js(ROOT/'checks/evaluation_status.json',{'status':'PASS' if len(aggregate)==len(inventory) else 'PARTIAL','inventory_count':len(inventory),'evaluated':len(aggregate),'reason':fail,'elapsed_seconds':time.monotonic()-START,'real27_used':False,'retraining':False})
 print('FINISHED',len(aggregate),len(inventory),'reason',fail,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:js(ROOT/'checks/fatal_failure.json',{'reason':str(exc),'traceback':traceback.format_exc()});raise
