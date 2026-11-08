"""CPU/GPU에서 게이트 단일 변경의 동일성, 유효영역, history 및 gradient 검증."""
import json,traceback,torch,numpy as np,importlib.util
from common_experiment import ROOT,BASE,OLD,D,guard,write,load_module,batch,model

def values(out):return torch.cat([out['mask_loss'].reshape(1),out['Front_L1Losses'],out['Back_L1Losses']])
def total(out):return out['mask_loss']+3*out['Front_L1Losses'].sum()+3*out['Back_L1Losses'].sum()
def synthetic(device,old,new):
 torch.manual_seed(1234);h=w=8;shape=(2,24,h,w);dtype=torch.float16 if device=='cuda' else torch.float32
 gm=torch.zeros((2,h,w),device=device);gm[:,1:7,1:7]=1;vf=torch.ones_like(gm);vb=torch.ones_like(gm);vf[:,2,2]=0;vb[:,3,3]=0;validones=torch.ones_like(gm)
 tf=torch.rand(shape,device=device);tb=torch.rand(shape,device=device);pf=torch.randn(shape,device=device,dtype=dtype,requires_grad=True);pb=torch.randn(shape,device=device,dtype=dtype,requires_grad=True);pm=torch.where(gm[:,None]>0,torch.tensor(20.,device=device,dtype=dtype),torch.tensor(-20.,device=device,dtype=dtype))
 c0=old.HccePose_Loss();c1=new.HccePose_Loss();diffs=[];weightdiff=[]
 for iteration in range(4):
  a=c0(pf,pb,pm,tf,tb,gm);z=c1(pf,pb,pm,tf,tb,gm,validones,validones);diffs.append(float((values(a)-values(z)).abs().max()))
  for key in ('weight_front_error_ratio','weight_back_error_ratio'):
   for x,y in zip(getattr(c0,key),getattr(c1,key)):weightdiff.append(float((x-y).abs().max()))
 assert max(diffs)<2e-6 and max(weightdiff)==0,(diffs,weightdiff)
 ga=torch.autograd.grad(total(a),(pf,pb),retain_graph=True);gb=torch.autograd.grad(total(z),(pf,pb),retain_graph=True);gdiff=max(float((x-y).abs().max()) for x,y in zip(ga,gb));assert gdiff<2e-6,gdiff
 # Always-missed predicted foreground. Different branch supports prove gate independence.
 pm_missing=torch.full_like(pm,-20.);old_c=old.HccePose_Loss();new_c=new.HccePose_Loss();old_out=old_c(pf,pb,pm_missing,tf,tb,gm);out=new_c(pf,pb,pm_missing,tf,tb,gm,vf,vb)
 assert float(old_out['Front_L1Losses'].sum())==0 and float(old_out['Back_L1Losses'].sum())==0
 assert float(out['Front_L1Losses'].sum())>0 and float(out['Back_L1Losses'].sum())>0 and float(out['mask_loss'])==float(old_out['mask_loss'])
 fg,bg=torch.autograd.grad(out['Front_L1Losses'].sum()+out['Back_L1Losses'].sum(),(pf,pb));gf=(gm>0)&(vf>0);gbb=(gm>0)&(vb>0)
 assert float(fg.permute(0,2,3,1)[~gf].abs().max())==0 and float(bg.permute(0,2,3,1)[~gbb].abs().max())==0
 assert float(fg.permute(0,2,3,1)[gf].abs().sum())>0 and float(bg.permute(0,2,3,1)[gbb].abs().sum())>0
 # Different gates must not alter dynamic weights/history, which still use the same predicted mask.
 for iteration in range(3):
  old_c(pf,pb,pm_missing,tf,tb,gm);new_c(pf,pb,pm_missing,tf,tb,gm,vf,vb)
 for key in ('weight_front_error_ratio','weight_back_error_ratio'):
  for x,y in zip(getattr(old_c,key),getattr(new_c,key)):assert torch.equal(x,y)
 for key in ('Front_error_list','Back_error_list'):
  for ax,ay in zip(getattr(old_c,key),getattr(new_c,key)):
   for x,y in zip(ax,ay):assert np.array_equal(np.asarray(x.data),np.asarray(y.data))
 return {'device':device,'identity_loss_max_abs_difference':max(diffs),'identity_gradient_max_abs_difference':gdiff,'predicted_weight_max_abs_difference':max(weightdiff),'masked_outside_gradient_zero':True,'valid_missing_gt_gradients_nonzero':True,'mask_loss_unchanged':True,'branch_validity_different_verified':True,'history_unchanged_under_different_gates':True,'old_missed_front':float(old_out['Front_L1Losses'].sum()),'new_missed_front':float(out['Front_L1Losses'].sum()),'new_missed_back':float(out['Back_L1Losses'].sum())}

def main():
 assert json.loads((ROOT/'checks/support_validation.json').read_text())['status']=='PASS'
 torch.set_num_threads(2);old=D.S.explicit_module();new=load_module();cpu=synthetic('cpu',old,new);guard();assert torch.cuda.is_available();gpu=synthetic('cuda',old,new)
 samples,train,val=D.S.dataset();guard();net=model(old,'cuda');ck=torch.load(OLD/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['model_state_dict']);net.eval();crit=old.HccePose_Loss();totals=[]
 with torch.inference_mode():
  for at in range(0,len(val),4):
   guard();ids=val[at:at+4];x,m,f,b,vf,vb=batch(samples,ids)
   with torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net(x);result=crit(pc[:,:24],pc[:,24:],pm,f,b,m);loss=total(result)
   totals.extend([float(loss)]*len(ids))
 logged=json.loads((OLD/'checks/training_summary.json').read_text())['best_val_loss']['metrics']['val']['loss'];actual=float(np.mean(totals));assert abs(actual-logged)<1e-6,(actual,logged)
 # Actual batch comparison; each Loss history starts from the same empty state.
 ids=val[:4];x,m,f,b,vf,vb=batch(samples,ids);old_c=old.HccePose_Loss();new_c=new.HccePose_Loss()
 with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net(x);p=old_c(pc[:,:24],pc[:,24:],pm,f,b,m);z=new_c(pc[:,:24],pc[:,24:],pm,f,b,m,vf,vb)
 observed={'frame_ids':ids,'old_total':float(total(p)),'new_total':float(total(z)),'old_mask':float(p['mask_loss']),'new_mask':float(z['mask_loss']),'old_front3':float(3*p['Front_L1Losses'].sum()),'new_front3':float(3*z['Front_L1Losses'].sum()),'old_back3':float(3*p['Back_L1Losses'].sum()),'new_back3':float(3*z['Back_L1Losses'].sum()),'pred_gate_pixels':int((torch.sigmoid(pm).round()>0).sum()),'GT_front_gate_pixels':int(((m>0.5)&vf).sum()),'GT_back_gate_pixels':int(((m>0.5)&vb).sum())}
 assert abs(observed['old_mask']-observed['new_mask'])<1e-7
 del net;torch.cuda.empty_cache();guard();torch.manual_seed(42);np.random.seed(42);net=model(new,'cuda');ref=torch.load(OLD.parent/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);assert ref['epoch']==0;bad=[k for k,v in net.state_dict().items() if not torch.equal(v.cpu(),ref['model_state_dict'][k])];assert not bad,bad[:10]
 initialization={'status':'PASS','seed':42,'all_initial_model_tensors_equal_baseline_epoch0':True,'reference':str(OLD.parent/'checkpoints/best_val_loss.pt'),'random_initialization':True,'pretrained':False,'resumed':False}
 # Independent GPU smoke: one forward/backward, no optimizer step and no persistent model state.
 net.train();ids=train[:4];x,m,f,b,vf,vb=batch(samples,ids);criterion=new.HccePose_Loss()
 with torch.amp.autocast('cuda',dtype=torch.float16):pm,pc=net(x);pc.retain_grad();out=criterion(pc[:,:24],pc[:,24:],pm,f,b,m,vf,vb);loss=total(out)
 assert torch.isfinite(loss);loss.backward();assert pc.grad is not None and torch.isfinite(pc.grad).all();gfront=float(pc.grad[:,:24].abs().sum());gback=float(pc.grad[:,24:].abs().sum());assert gfront>0 and gback>0
 smoke={'status':'PASS','device':'cuda','batch':4,'optimizer_steps':0,'loss':float(loss),'front_code_gradient_abs_sum':gfront,'back_code_gradient_abs_sum':gback,'model_discarded':True,'real_images_used':0}
 write(ROOT/'checks/loss_equivalence.json',{'status':'PASS','cpu':cpu,'gpu':gpu,'baseline_checkpoint_val_reproduction':{'epoch':155,'logged_val_loss':logged,'recomputed_val_loss':actual,'abs_diff':abs(actual-logged)},'actual_batch_comparison':observed})
 write(ROOT/'checks/mask_gate_validation.json',{'status':'PASS','CPU':cpu,'GPU':gpu,'real_support_source':'checks/support_validation.json','actual_batch_comparison':observed})
 write(ROOT/'checks/initialization.json',initialization);write(ROOT/'checks/gpu_smoke.json',smoke);print(json.dumps({'cpu':cpu,'gpu':gpu,'baseline_val':actual,'actual_batch':observed,'initialization':initialization,'smoke':smoke},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'checks/validation_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc()});raise
