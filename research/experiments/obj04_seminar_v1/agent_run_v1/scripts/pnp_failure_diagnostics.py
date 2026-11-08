"""기존 두 체크포인트의 PBR Val 입력 3D 대응점 분포만 비교한다. 학습·설정 탐색 없음."""
from pathlib import Path
import os,sys,time,subprocess,json,csv,traceback
import cv2,numpy as np,torch
R=Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'diagnostics_v1/scripts'));import diagnose as D

def guard():
 if time.time()>=1793714609+75*60:raise RuntimeError('보고서 시간 확보: 추가 추론 중단')
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);other=[x for x in p.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def main():
 guard();torch.set_num_threads(2);samples,train,val=D.S.dataset();paths={'baseline':B/'runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt','rotation':Path(json.loads((R/'checks/rotation_training.json').read_text())['best_val_loss']['checkpoint'])};records=[]
 for name,path in paths.items():
  guard();m=D.S.explicit_module();net=D.S.build_network(m,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda();ck=torch.load(path,map_location='cpu',weights_only=True);net.load_state_dict(ck['model_state_dict']);net.eval()
  for at in range(0,len(val),4):
   guard();ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();bbox=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):o=net.inference_batch(x,bbox)
   for j,i in enumerate(ids):
    pm=o['pred_mask'][j].cpu().numpy()>0;xyz=o['pred_front_code_obj'][j].float().cpu().numpy()[pm];n=len(xyz)
    if n:
     values=xyz.astype(float);sigma=np.linalg.svd(values-values.mean(0),compute_uv=False)/np.sqrt(n);uniq=np.unique(xyz,axis=0).shape[0];sd=np.std(values,axis=0)
    else:sigma=np.zeros(3);uniq=0;sd=np.zeros(3)
    records.append({'model':name,'frame':i,'correspondences':n,'unique_xyz':int(uniq),'unique_fraction':float(uniq/n) if n else 0.,'x_std':float(sd[0]),'y_std':float(sd[1]),'z_std':float(sd[2]),'sv_1':float(sigma[0]),'sv_2':float(sigma[1]),'sv_3':float(sigma[2]),'smallest_largest_sv_ratio':float(sigma[2]/sigma[0]) if sigma[0]>0 else None,'scope':'predicted mask; xyz in model coordinate units'})
  del net;torch.cuda.empty_cache()
 with (R/'stage3/pnp_correspondence_geometry.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
 summaries=[]
 for name in paths:
  rr=[x for x in records if x['model']==name];o={'model':name,'frames':len(rr)}
  for k in ['correspondences','unique_xyz','unique_fraction','x_std','y_std','z_std','smallest_largest_sv_ratio']:
   a=[x[k] for x in rr if x[k] is not None];o[k+'_mean']=float(np.mean(a)) if a else None;o[k+'_median']=float(np.median(a)) if a else None
  summaries.append(o)
 with (R/'stage3/pnp_correspondence_geometry.json').open('x') as f:json.dump(summaries,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  with (R/'checks/pnp_diagnostic_failure.json').open('x') as f:json.dump({'reason':str(e),'traceback':traceback.format_exc()},f,ensure_ascii=False,indent=2)
  raise
