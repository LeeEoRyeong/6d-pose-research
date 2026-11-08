"""동기 in-plane 회전 하나만 변경한 단일 GPU 학습. 타 GPU 작업 감지 시 중단."""
from pathlib import Path
import os,time,sys,json,subprocess,traceback,importlib.util
import numpy as np,torch
from augmentation import ROOT,BASE,D,Augmentation
OLD=BASE/'runs/baseline_seed42_20261103_214353/trained_only';STAGE=ROOT/'stage3';DEADLINE=1793714609+75*60

def guard(force=False):
 if time.time()>=DEADLINE:raise RuntimeError('75분 경과: 새 GPU 작업 중단, 보고서 우선')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def checkpoint(name,net,epoch,steps,optimizer=None,criterion=None,metrics=None,weights=None):
 path=STAGE/'checkpoints'/name
 if path.exists():raise FileExistsError(path)
 value={'model_state_dict':weights if weights is not None else old.snapshot(net),'epoch':epoch,'optimizer_steps':steps,'object_id':4,'config':config,'metrics':metrics,'initialization':'random seed42, baseline-equal initialization; not resumed','torch_rng':torch.get_rng_state()}
 if optimizer is not None:value['optimizer_state_dict']=optimizer.state_dict()
 if criterion is not None:value['loss_history']=old.loss_history(criterion)
 torch.save(value,path);return str(path)
for p in [ROOT/'checks/stage1_gate.json',ROOT/'stage2/analysis_summary.json',ROOT/'checks/pbr_repro_gate.json',ROOT/'checks/augmentation_alignment.json',ROOT/'checks/augmentation_cpu_smoke.json']:
 assert json.loads(p.read_text())['status']=='PASS',p
config={'object_id':4,'seed':42,'batch_size':4,'crop':256,'optimizer':'Adam','lr':.0002,'amp':'float16 GradScaler','max_epochs':195,'early_stop_validation_checks':8,'early_stop_min_delta':.0001,'validation_every_epochs':5,'checkpoint_every_epochs':50,'angle_distribution':'uniform[-20,+20] degrees per train sample','angle_rng':'independent NumPy default_rng(42); baseline torch randperm RNG preserved','M_256_128':'same normalized 2D affine; RGB linear, mask/XYZ/support nearest; HCCE encode after rotation','loss':'baseline explicit float-cast-compatible HccePose_Loss, same weights, unchanged','validity':'transformed, monitored; not separately inserted into baseline loss','selection':'epoch>=1, Val Loss improvement min_delta .0001, patience8 checks, as baseline','evaluation':'same unaugmented Train/Val and same BBox, batch4, same PnP as baseline','real_training_samples':0,'experiment_runs_authorized':1}

def main():
 guard();(STAGE/'checkpoints').mkdir();torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False
 samples,train,val=D.S.dataset();aug=Augmentation(samples,train);assert len(train)==80 and len(val)==20 and not set(train)&set(val)
 mod=D.S.explicit_module();minimum=torch.tensor(D.LO,dtype=torch.float32,device='cuda');size=torch.tensor(D.SIZE,dtype=torch.float32,device='cuda');net=D.S.build_network(mod,minimum,size).cuda()
 # The diagnostic attempt's epoch0 snapshot records the original baseline initial parameters.
 baseline0=torch.load(OLD.parent/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);assert baseline0['epoch']==0
 state=net.state_dict();base=baseline0['model_state_dict'];assert state.keys()==base.keys();bad=[k for k in state if not torch.equal(state[k].cpu(),base[k])];assert not bad,('baseline 초기 state 불일치',bad[:10]);del baseline0,base
 save(ROOT/'checks/rotation_initialization.json',{'status':'PASS','baseline_epoch0_reference':str(OLD.parent/'checkpoints/best_val_loss.pt'),'all_model_state_tensors_exact':True,'resumed':False})
 criterion=mod.HccePose_Loss();optimizer=torch.optim.Adam(net.parameters(),lr=config['lr']);scaler=torch.amp.GradScaler('cuda');generator=torch.Generator().manual_seed(42);anglerng=np.random.default_rng(42);torch.cuda.reset_peak_memory_stats()
 spec=importlib.util.spec_from_file_location('baseline_evaluate',OLD/'scripts/train_baseline.py');global old;old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.guard=guard
 start=time.monotonic();init_train=old.evaluate(net,mod,samples,train);init_val=old.evaluate(net,mod,samples,val);initial={'epoch':0,'optimizer_steps':0,'train_eval':init_train,'val':init_val,'elapsed_seconds':time.monotonic()-start};save(STAGE/'initial_metrics.json',initial);save(STAGE/'config.json',config)
 best=float('inf');best_epoch=0;best_steps=0;best_state=old.snapshot(net);best_metrics=initial;best_iou=init_val['mask_iou_mean'];iou_epoch=0;iou_steps=0;iou_state=old.snapshot(net);iou_metrics=initial
 wait=0;steps=0;skips=0;completed=0;reason='max_epochs';last_record=initial;save(ROOT/'checks/training_started.json',{'status':'STARTED','single_gpu_training_run':1,'real_training_frames':0})
 with (ROOT/'logs/rotation_epochs.jsonl').open('x') as epo,(ROOT/'logs/rotation_steps.jsonl').open('x') as st:
  epo.write(json.dumps(initial)+'\n');epo.flush()
  for epoch in range(1,196):
   guard();net.train();order=[train[i] for i in torch.randperm(len(train),generator=generator).tolist()];losses=[]
   for at in range(0,len(order),4):
    guard();ids=order[at:at+4];angles=anglerng.uniform(-20.,20.,len(ids));x,m,f,b=aug.batch(ids,angles,'cuda');optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net(x);loss=D.S.loss_value(criterion(pc[:,:24],pc[:,24:],pm,f,b,m))
    if not torch.isfinite(loss):raise RuntimeError('비유한 training loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'));scale=scaler.get_scale();scaler.step(optimizer);scaler.update();did=scaler.get_scale()>=scale
    if did:steps+=1
    else:skips+=1
    losses.append(float(loss));st.write(json.dumps({'epoch':epoch,'batch_ids':ids,'angles_deg':angles.tolist(),'loss':float(loss),'gradient_norm':float(grad) if torch.isfinite(grad) else None,'optimizer_step_applied':did,'optimizer_steps_total':steps,'amp_scale':scaler.get_scale()})+'\n')
   completed=epoch;st.flush()
   if epoch%5==0 or epoch==1:
    train_eval=old.evaluate(net,mod,samples,train);v=old.evaluate(net,mod,samples,val);row={'epoch':epoch,'optimizer_steps':steps,'train_optimization_loss':float(np.mean(losses)),'train_eval':train_eval,'val':v,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};last_record=row;epo.write(json.dumps(row)+'\n');epo.flush();print(json.dumps(row),flush=True)
    if v['loss']<best-config['early_stop_min_delta']:best=v['loss'];best_epoch=epoch;best_steps=steps;best_state=old.snapshot(net);best_metrics=row;wait=0
    else:wait+=1
    if v['mask_iou_mean']>best_iou:best_iou=v['mask_iou_mean'];iou_epoch=epoch;iou_steps=steps;iou_state=old.snapshot(net);iou_metrics=row
    if wait>=config['early_stop_validation_checks']:reason='early_stop_val_loss';break
   if epoch%50==0:checkpoint(f'epoch_{epoch:04d}.pt',net,epoch,steps,optimizer,criterion,last_record)
 last={'train_eval':old.evaluate(net,mod,samples,train),'val':old.evaluate(net,mod,samples,val)}
 best_path=checkpoint('best_val_loss.pt',net,best_epoch,best_steps,metrics=best_metrics,weights=best_state);iou_path=checkpoint('best_val_mask_iou.pt',net,iou_epoch,iou_steps,metrics=iou_metrics,weights=iou_state);last_path=checkpoint('last.pt',net,completed,steps,optimizer,criterion,last)
 summary={'status':'PASS','completed_epochs':completed,'successful_optimizer_steps':steps,'amp_skipped_steps':skips,'stop_reason':reason,'training_wall_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'best_val_loss':{'epoch':best_epoch,'steps':best_steps,'metrics':best_metrics,'checkpoint':best_path},'best_val_mask_iou_secondary':{'epoch':iou_epoch,'steps':iou_steps,'metrics':iou_metrics,'checkpoint':iou_path},'last':{'epoch':completed,'metrics':last,'checkpoint':last_path},'training_runs':1,'real_training_frames':0}
 save(ROOT/'checks/rotation_training.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(ROOT/'checks/rotation_training_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc(),'single_attempt_only':True});raise
