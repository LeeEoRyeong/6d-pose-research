import json,time,traceback
import numpy as np
import torch
from support import RUN,BASE,CONFIG,load_json,write_json,explicit_module,build_network,guard,OtherGpuJob,Deadline,dataset,batch,loss_value

@torch.inference_mode()
def evaluate(net,module,samples,ids):
    net.eval();criterion=module.HccePose_Loss();losses=[];ious=[];inter=union=0
    for at in range(0,len(ids),4):
        guard();part=ids[at:at+4];x,m,f,b=batch(samples,part)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            pm,pc=net(x);parts=criterion(pc[:,:24],pc[:,24:],pm,f,b,m);loss=loss_value(parts)
        assert torch.isfinite(loss)
        losses.extend([float(loss)]*len(part))
        pred=pm[:,0]>0;gt=m>.5;ii=(pred&gt).sum((1,2));uu=(pred|gt).sum((1,2))
        ious.extend((ii/uu.clamp_min(1)).cpu().tolist());inter+=int(ii.sum());union+=int(uu.sum())
    return {'loss':float(np.mean(losses)),'mask_iou_mean':float(np.mean(ious)),'mask_iou_global':inter/max(union,1),'n':len(ids)}

def snapshot(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def loss_history(criterion):
    return {name:[[[float(v) for v in item.data] for item in axis] for axis in getattr(criterion,name)] for name in ('Front_error_list','Back_error_list')}

def checkpoint(name,net,epoch,steps,optimizer=None,criterion=None,metrics=None,weights=None):
    path=RUN/'checkpoints'/name
    if path.exists():raise FileExistsError(path)
    value={'model_state_dict':weights if weights is not None else snapshot(net),'epoch':epoch,'optimizer_steps':steps,'object_id':4,'config':CONFIG,'metrics':metrics,'initialization':'random seed42, not resumed from smoke','torch_rng':torch.get_rng_state()}
    if optimizer is not None:value['optimizer_state_dict']=optimizer.state_dict()
    if criterion is not None:value['loss_history']=loss_history(criterion)
    torch.save(value,path);return str(path)

def main():
    assert load_json(RUN/'checks/loss_compatibility.json')['status']=='PASS'
    assert load_json(BASE/'checks/labels_v2_100.json')['status']=='PASS'
    guard(force=True);torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    samples,train,val=dataset()
    write_json(RUN/'checks/split.json',{'train_ids':train,'val_ids':val,'train_blocks':sorted({i//10 for i in train}),'val_blocks':sorted({i//10 for i in val}),'overlap':[],'real_frames_used':0,'sample_preparation':'기존 라벨·crop 함수 재사용. 고정 pad1.5 GT visible bbox; augmentation 없음'})
    info=load_json(BASE/'data/models/models_info.json')['4'];minimum=torch.tensor([info['min_x'],info['min_y'],info['min_z']],device='cuda');size=torch.tensor([info['size_x'],info['size_y'],info['size_z']],device='cuda')
    module=explicit_module();net=build_network(module,minimum,size).cuda();criterion=module.HccePose_Loss()
    optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['lr']);scaler=torch.amp.GradScaler('cuda')
    generator=torch.Generator().manual_seed(42);steps=0;skipped=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    initial_train=evaluate(net,module,samples,train);initial_val=evaluate(net,module,samples,val)
    initial={'epoch':0,'optimizer_steps':0,'train_eval':initial_train,'val':initial_val,'elapsed_seconds':time.monotonic()-start}
    records=[initial];print(json.dumps(initial),flush=True)
    write_json(RUN/'checks/initial_metrics.json',initial)
    best=float('inf');best_epoch=0;best_steps=0;best_state=snapshot(net);best_metrics=initial
    best_iou=initial_val['mask_iou_mean'];iou_epoch=0;iou_steps=0;iou_state=snapshot(net);iou_metrics=initial
    wait=0;stop_reason='max_epochs';completed_epoch=0
    with (RUN/'logs/epoch_metrics.jsonl').open('x') as epochs_file,(RUN/'logs/train_steps.jsonl').open('x') as step_file:
        epochs_file.write(json.dumps(initial)+'\n');epochs_file.flush()
        for epoch in range(1,CONFIG['max_epochs']+1):
            guard();net.train();order=[train[i] for i in torch.randperm(len(train),generator=generator).tolist()];epoch_losses=[]
            for at in range(0,len(order),4):
                guard()
                if time.monotonic()-start>=CONFIG['max_gpu_training_seconds']:stop_reason='training_wall_time_limit';break
                ids=order[at:at+4];x,m,f,b=batch(samples,ids);optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    pm,pc=net(x);parts=criterion(pc[:,:24],pc[:,24:],pm,f,b,m);loss=loss_value(parts)
                if not torch.isfinite(loss):raise RuntimeError('비유한 training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'));scale=scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                did_step=scaler.get_scale()>=scale
                if did_step:steps+=1
                else:skipped+=1
                epoch_losses.append(float(loss));row={'epoch':epoch,'batch_ids':ids,'loss':float(loss),'gradient_norm':float(grad) if torch.isfinite(grad) else None,'optimizer_step_applied':did_step,'optimizer_steps_total':steps,'amp_scale':scaler.get_scale()}
                step_file.write(json.dumps(row)+'\n')
            completed_epoch=epoch;step_file.flush()
            if epoch%CONFIG['validation_every_epochs']==0 or epoch==1 or stop_reason=='training_wall_time_limit':
                train_eval=evaluate(net,module,samples,train);v=evaluate(net,module,samples,val)
                row={'epoch':epoch,'optimizer_steps':steps,'train_optimization_loss':float(np.mean(epoch_losses)) if epoch_losses else None,'train_eval':train_eval,'val':v,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                records.append(row);epochs_file.write(json.dumps(row)+'\n');epochs_file.flush();print(json.dumps(row),flush=True)
                if v['loss']<best-CONFIG['early_stop_min_delta']:
                    best=v['loss'];best_epoch=epoch;best_steps=steps;best_state=snapshot(net);best_metrics=row;wait=0
                else:wait+=1
                if v['mask_iou_mean']>best_iou:
                    best_iou=v['mask_iou_mean'];iou_epoch=epoch;iou_steps=steps;iou_state=snapshot(net);iou_metrics=row
                if wait>=CONFIG['early_stop_validation_checks']:stop_reason='early_stop_val_loss';break
            if epoch%CONFIG['checkpoint_every_epochs']==0:checkpoint(f'epoch_{epoch:04d}.pt',net,epoch,steps,optimizer,criterion,records[-1])
            if stop_reason=='training_wall_time_limit':break
    last_metrics={'train_eval':evaluate(net,module,samples,train),'val':evaluate(net,module,samples,val)}
    best_path=checkpoint('best_val_loss.pt',net,best_epoch,best_steps,metrics=best_metrics,weights=best_state)
    iou_path=checkpoint('best_val_mask_iou.pt',net,iou_epoch,iou_steps,metrics=iou_metrics,weights=iou_state)
    last_path=checkpoint('last.pt',net,completed_epoch,steps,optimizer,criterion,last_metrics)
    summary={'status':'PASS','completed_epochs':completed_epoch,'successful_optimizer_steps':steps,'amp_skipped_steps':skipped,'stop_reason':stop_reason,'training_wall_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'initial':initial,'best_val_loss':{'epoch':best_epoch,'optimizer_steps':best_steps,'metrics':best_metrics,'checkpoint':best_path},'best_val_mask_iou_secondary':{'epoch':iou_epoch,'optimizer_steps':iou_steps,'metrics':iou_metrics,'checkpoint':iou_path},'last':{'epoch':completed_epoch,'metrics':last_metrics,'checkpoint':last_path},'selection':'primary: minimum PBR val loss among trained epochs>=1; secondary: maximum PBR val MaskIoU (exploratory, same validation set)','evaluation_loss_policy':'매 평가마다 새 Loss history. train/val history는 학습 Loss history와 공유하지 않음','real_frames_used':0,'parameter_count':sum(p.numel() for p in net.parameters())}
    write_json(RUN/'checks/training_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except (OtherGpuJob,Deadline) as exc:
        write_json(RUN/'checks/training_stopped.json',{'status':'STOPPED','reason':str(exc),'traceback':traceback.format_exc()});raise SystemExit(2)
    except Exception as exc:
        write_json(RUN/'checks/training_failure.json',{'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()});raise
