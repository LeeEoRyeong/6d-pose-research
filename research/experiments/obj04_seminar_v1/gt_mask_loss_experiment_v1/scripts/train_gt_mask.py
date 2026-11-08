"""GT∩branch support만 좌표 L1 gate로 바꾼 단일 학습. Dynamic weights는 원본 그대로."""
from pathlib import Path
import time,json,math,traceback,csv,hashlib,cv2,numpy as np,torch
from common_experiment import ROOT,BASE,OLD,D,guard,write,load_module,batch,model
from validate_loss import total

def snapshot(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def loss_history(criterion):return {name:[[[float(v) for v in item.data] for item in axis] for axis in getattr(criterion,name)] for name in ('Front_error_list','Back_error_list')}
def cp(name,net,epoch,steps,metrics,weights=None,optimizer=None,criterion=None):
 p=ROOT/'checkpoints'/name
 if p.exists():raise FileExistsError(p)
 obj={'model_state_dict':weights if weights is not None else snapshot(net),'epoch':epoch,'optimizer_steps':steps,'object_id':4,'metrics':metrics,'config':CONFIG,'torch_rng':torch.get_rng_state()}
 if optimizer is not None:obj['optimizer_state_dict']=optimizer.state_dict()
 if criterion is not None:obj['loss_history']=loss_history(criterion)
 torch.save(obj,p);return str(p)
CONFIG={'change':'좌표 L1 gate만 GT visible mask ∩ branch-specific reconstructed valid support','dynamic_weight_history':'predicted mask 유지','loss_reduction':'원본 batch-height-width 전체 mean, 추가 정규화 없음','seed':42,'batch_size':4,'crop':256,'label_resolution':128,'optimizer':'Adam','lr':0.0002,'AMP':'float16 GradScaler','max_epoch':195,'max_optimizer_steps':3900,'max_gpu_training_wall_seconds':2700,'augmentation':'none','pretrained':False,'train_frames':80,'val_frames':20,'real_frames_for_training_selection':0,'validation_interval_epochs':5,'early_stop_patience_checks':8,'early_stop_min_delta':0.0001,'official_selection':'Val total loss among epoch>=1','secondary_selection':'0.5*(Val Front XYZ L2+Val Back XYZ L2), GT valid support','pnp':'original front/back solve_PnP pnp_op2, reprojectionError=1.5px, 150iterations, seed42+frame','baseline_lr_and_train_order':'same seeded generator and exact initial state; no resume'}

def make_truth(samples,ids):
 out={}
 for i in ids:
  box=samples[i]['bbox'].numpy();out[i]={}
  for kind in ('front','back'):
   full=cv2.imread(str(BASE/'labels'/kind/f'{i:06d}_000000.png'));assert full is not None
   out[i][kind]=D.CROP(full,box,128,cv2.INTER_NEAREST).astype(float)/255*D.SIZE+D.LO
 return out

@torch.inference_mode()
def evaluate(net,module,samples,ids,truth):
 net.eval();criterion=module.HccePose_Loss();components=[];fr=[];pn=[];D.RECS.clear()
 for at in range(0,len(ids),4):
  guard();part=ids[at:at+4];x,m,f,b,vf,vb=batch(samples,part);bbox=torch.stack([samples[i]['bbox'] for i in part]).cuda()
  with torch.amp.autocast('cuda',dtype=torch.float16):
   pm,pc=net(x);loss=criterion(pc[:,:24],pc[:,24:],pm,f,b,m,vf,vb);z=total(loss);inference=net.inference_batch(x,bbox)
  assert torch.isfinite(z);source_hard=(torch.sigmoid(pc)>.5).cpu().numpy().transpose(0,2,3,1)
  components.append({'n':len(part),'total':float(z),'mask':float(loss['mask_loss']),'front':float(3*loss['Front_L1Losses'].sum()),'back':float(3*loss['Back_L1Losses'].sum()),'pred_gate_pixels':int((torch.sigmoid(pm).round()>0).sum()),'GT_pixels':int((m>.5).sum()),'GT_missed':int(((m>.5)&(torch.sigmoid(pm[:,0]).round()==0)).sum()),'front_gate_pixels':int(((m>.5)&vf).sum()),'back_gate_pixels':int(((m>.5)&vb).sum())})
  for j,i in enumerate(part):
   predmask=inference['pred_mask'][j].cpu().numpy()>0;uv=inference['coord_2d_image'][j].float().cpu().numpy();gtmask=m[j].cpu().numpy()>.5;iou=float(np.count_nonzero(predmask&gtmask)/max(np.count_nonzero(predmask|gtmask),1));row={'frame':i,'mask_iou':iou}
   for kind,key,v,offset,target in [('front','pred_front_code_obj',vf,0,f),('back','pred_back_code_obj',vb,24,b)]:
    point=inference[key][j].float().cpu().numpy();valid=gtmask&v[j].cpu().numpy();assert valid.any();err=point[valid]-truth[i][kind][valid];truthbits=target[j].permute(1,2,0).cpu().numpy()>.5
    row[kind+'_XYZ_L2']=float(np.linalg.norm(err,axis=1).mean());row[kind+'_Z0_error']=float(np.mean(source_hard[j,...,offset+16][valid]!=truthbits[...,16][valid]))
    for k,axis in enumerate('xyz'):row[kind+'_'+axis+'_MAE']=float(np.abs(err[:,k]).mean())
    result=D.pose(kind,i,predmask,point,uv,samples[i]['K'],samples[i]['R'],samples[i]['t'].reshape(3))
    if result['valid_pose'] and not all(result[k] is not None and math.isfinite(float(result[k])) for k in ('add','rotation_deg','translation')):result.update(valid_pose=False,add_pass=False,add=None,rotation_deg=None,translation=None,failure='비유한 평가 오차')
    pn.append({'frame':i,'branch':kind,**result})
   fr.append(row)
 D.ALL.clear();D.RECS.clear();n=sum(x['n'] for x in components);w=lambda k:sum(x[k]*x['n'] for x in components)/n
 out={'n':n,'total_loss':w('total'),'mask_loss':w('mask'),'front_HCCE_loss_3x':w('front'),'back_HCCE_loss_3x':w('back'),'mask_iou':float(np.mean([v['mask_iou'] for v in fr])),'pred_gate_pixels':sum(v['pred_gate_pixels'] for v in components),'GT_pixels':sum(v['GT_pixels'] for v in components),'GT_missed_fraction':sum(v['GT_missed'] for v in components)/max(1,sum(v['GT_pixels'] for v in components)),'front_gate_pixels':sum(v['front_gate_pixels'] for v in components),'back_gate_pixels':sum(v['back_gate_pixels'] for v in components)}
 for kind in ('front','back'):
  for key in ('XYZ_L2','Z0_error','x_MAE','y_MAE','z_MAE'):out[kind+'_'+key]=float(np.mean([v[kind+'_'+key] for v in fr]))
  ss=[v for v in pn if v['branch']==kind];good=[v for v in ss if v['valid_pose']];out[kind+'_PnP_valid']=len(good);out[kind+'_ADD_pass']=sum(v['add_pass'] for v in ss);out[kind+'_ADD_rate']=sum(v['add_pass'] for v in ss)/len(ss)
  for key in ('add','rotation_deg','translation'):out[kind+'_'+key+'_mean']=float(np.mean([v[key] for v in good])) if good else None
 return out

def main():
 for name in ['support_validation.json','loss_equivalence.json','mask_gate_validation.json','initialization.json','gpu_smoke.json']:
  assert json.loads((ROOT/'checks'/name).read_text())['status']=='PASS',name
 guard();torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False;samples,train,val=D.S.dataset();assert len(train)==80 and len(val)==20 and not set(train)&set(val) and not {i//10 for i in train}&{i//10 for i in val}
 # Verify actual epoch-1 seeded order without consuming the run's generator.
 firstlog=[]
 with (OLD/'logs/train_steps.jsonl').open() as f:
  for line in f:
   z=json.loads(line)
   if z['epoch']!=1:break
   firstlog.extend(z['batch_ids'])
 tester=torch.Generator().manual_seed(42);expected=[train[i] for i in torch.randperm(len(train),generator=tester).tolist()];assert expected==firstlog
 module=load_module();net=model(module,'cuda');reference=torch.load(OLD.parent/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);assert reference['epoch']==0;assert all(torch.equal(v.cpu(),reference['model_state_dict'][k]) for k,v in net.state_dict().items());del reference
 write(ROOT/'configs/experiment_config.json',{**CONFIG,'train_ids':train,'val_ids':val,'train_blocks':sorted({i//10 for i in train}),'val_blocks':sorted({i//10 for i in val}),'first_epoch_batch_order_equal_baseline':True})
 truth=make_truth(samples,train+val);criterion=module.HccePose_Loss();optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['lr']);scaler=torch.amp.GradScaler('cuda');generator=torch.Generator().manual_seed(42);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 initial_train=evaluate(net,module,samples,train,truth);initial_val=evaluate(net,module,samples,val,truth);init={'epoch':0,'optimizer_steps':0,'train':initial_train,'val':initial_val,'train_optimization_loss':None};write(ROOT/'checks/initial_metrics.json',init)
 bestloss=float('inf');bestxyz=float('inf');bestloss_state=None;bestxyz_state=None;bestlossrow=None;bestxyzrow=None;wait=0;steps=0;skips=0;attempts=0;completed=0;reason='max_epochs';write(ROOT/'checks/training_started.json',{'single_training_run':1,'first_epoch_order_baseline_match':True,'initial_state_baseline_match':True,'real_images_used':0})
 rows=[]
 with (ROOT/'logs/train_steps.jsonl').open('x') as st,(ROOT/'logs/evaluation.jsonl').open('x') as ev:
  ev.write(json.dumps(init)+'\n');ev.flush()
  for epoch in range(1,196):
   guard();net.train();order=[train[j] for j in torch.randperm(len(train),generator=generator).tolist()];losses=[]
   for at in range(0,80,4):
    guard()
    if time.monotonic()-start>CONFIG['max_gpu_training_wall_seconds']:reason='GPU_wall_limit';break
    ids=order[at:at+4];x,m,f,b,vf,vb=batch(samples,ids);optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net(x);parts=criterion(pc[:,:24],pc[:,24:],pm,f,b,m,vf,vb);loss=total(parts)
    if not torch.isfinite(loss):raise RuntimeError('비유한 training Loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'));scale=scaler.get_scale();scaler.step(optimizer);scaler.update();did=scaler.get_scale()>=scale;attempts+=1
    if did:steps+=1
    else:skips+=1
    assert steps<=3900;losses.append(float(loss));st.write(json.dumps({'epoch':epoch,'batch_ids':ids,'loss':float(loss),'mask_loss':float(parts['mask_loss']),'front_HCCE_loss_3x':float(parts['Front_L1Losses'].sum()*3),'back_HCCE_loss_3x':float(parts['Back_L1Losses'].sum()*3),'pred_gate_pixels':int((torch.sigmoid(pm).round()>0).sum()),'front_gate_pixels':int(((m>.5)&vf).sum()),'back_gate_pixels':int(((m>.5)&vb).sum()),'gradient_norm':float(grad) if torch.isfinite(grad) else None,'optimizer_step_applied':did,'optimizer_steps_total':steps})+'\n')
   completed=epoch;st.flush()
   if epoch%5==0 or epoch==1 or reason!='max_epochs':
    tr=evaluate(net,module,samples,train,truth);v=evaluate(net,module,samples,val,truth);score=.5*(v['front_XYZ_L2']+v['back_XYZ_L2']);row={'epoch':epoch,'optimizer_steps':steps,'attempted_steps':attempts,'amp_skipped_steps':skips,'train_optimization_loss':float(np.mean(losses)) if losses else None,'train':tr,'val':v,'val_XYZ_score':score,'elapsed_seconds':time.monotonic()-start};rows.append(row);ev.write(json.dumps(row,allow_nan=False)+'\n');ev.flush();print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    if v['total_loss']<bestloss-CONFIG['early_stop_min_delta']:bestloss=v['total_loss'];bestlossrow=row;bestloss_state=snapshot(net);wait=0
    else:wait+=1
    if score<bestxyz:bestxyz=score;bestxyzrow=row;bestxyz_state=snapshot(net)
    if wait>=CONFIG['early_stop_patience_checks']:reason='early_stop_val_loss';break
   if epoch%50==0:cp(f'epoch_{epoch:04d}.pt',net,epoch,steps,rows[-1] if rows else init,optimizer=optimizer,criterion=criterion)
   if reason!='max_epochs':break
 assert bestlossrow is not None and bestxyzrow is not None
 official=cp('best_val_loss.pt',net,bestlossrow['epoch'],bestlossrow['optimizer_steps'],bestlossrow,weights=bestloss_state);secondary=cp('best_val_xyz.pt',net,bestxyzrow['epoch'],bestxyzrow['optimizer_steps'],bestxyzrow,weights=bestxyz_state);last=cp('last.pt',net,completed,steps,rows[-1],optimizer=optimizer,criterion=criterion)
 fields=['epoch','optimizer_steps','attempted_steps','amp_skipped_steps','train_optimization_loss','val_XYZ_score','elapsed_seconds']+[f'{split}_{k}' for split in ('train','val') for k in rows[0][split]]
 with (ROOT/'tables/train_history.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({**{k:r[k] for k in fields[:7]},**{f'{split}_{k}':v for split in ('train','val') for k,v in r[split].items()}})
 result={'status':'PASS','single_training_run':1,'epoch_completed':completed,'optimizer_steps':steps,'optimizer_attempts':attempts,'amp_skipped_steps':skips,'stop_reason':reason,'wall_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'official':{'epoch':bestlossrow['epoch'],'steps':bestlossrow['optimizer_steps'],'checkpoint':official,'metrics':bestlossrow},'secondary':{'epoch':bestxyzrow['epoch'],'steps':bestxyzrow['optimizer_steps'],'checkpoint':secondary,'metrics':bestxyzrow},'last_checkpoint':last,'same_selected_epoch':bestlossrow['epoch']==bestxyzrow['epoch'],'real_images_training_selection':0}
 write(ROOT/'checks/training_summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'checks/training_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc(),'no_automatic_retry':True});raise
