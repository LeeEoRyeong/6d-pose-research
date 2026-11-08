"""GT 게이트 counterfactual의 수식이 원본 predicted gate에서 실제 Loss와 일치하는지 재확인."""
from pathlib import Path
import os,sys,json,subprocess,time,torch,numpy as np
R=Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'diagnostics_v1/scripts'));import diagnose as D
q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not [x for x in q.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
torch.set_num_threads(2);samples,train,val=D.S.dataset();ids=val[:4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();mask=torch.stack([samples[i]['tensors'][1] for i in ids]).cuda();f=torch.stack([samples[i]['tensors'][2] for i in ids]).cuda();b=torch.stack([samples[i]['tensors'][3] for i in ids]).cuda();results=[]
for model,path in [('baseline',B/'runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt'),('rotation',B/'agent_run_v1/stage3/checkpoints/best_val_loss.pt')]:
 m=D.S.explicit_module();net=D.S.build_network(m,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model_state_dict']);net.eval();crit=m.HccePose_Loss()
 with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
  pm,code=net(x);parts=crit(code[:,:24],code[:,24:],pm,f,b,mask);gate=torch.sigmoid(pm).detach().round().clamp(0,1)
  for branch,gt,wkey,part in [('front',f,'weight_front_error_ratio','Front_L1Losses'),('back',b,'weight_back_error_ratio','Back_L1Losses')]:
   pred=code[:,:24] if branch=='front' else code[:,24:];w=getattr(crit,wkey);v=0.
   for axis in range(3):
    weight=torch.exp(torch.minimum(w[axis],.51-w[axis])*3);loss=((pred[:,axis*8:(axis+1)*8]-(gt[:,axis*8:(axis+1)*8]*2-1)).abs()*gate).mean([0,2,3]);v=v+torch.sum(loss*weight)/torch.sum(weight)
   actual=float(parts[part].sum());calculated=float(v);difference=abs(actual-calculated);assert difference<1e-6,(model,branch,difference);results.append({'model':model,'branch':branch,'original_unweighted_sum':actual,'independent_expression':calculated,'absolute_difference':difference})
 del net;torch.cuda.empty_cache()
with (R/'checks/counterfactual_formula_check.json').open('x') as f:json.dump({'status':'PASS','first_batch_ids':ids,'predicted_gate_reexpression_checks':results,'scope':'single batch for each trained checkpoint; weights from same Loss history; no gradient or model update'},f,ensure_ascii=False,indent=2)
print(json.dumps(results,ensure_ascii=False,indent=2))
