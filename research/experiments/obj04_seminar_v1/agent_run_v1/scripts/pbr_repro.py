from pathlib import Path
import sys,json,os,time,subprocess,csv,hashlib,traceback
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TRAIN=BASE/'runs/baseline_seed42_20261103_214353/trained_only';sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D

def guard():
 if time.time()>=1793714609+75*60:raise RuntimeError('시간 예산: 보고서 우선')
 q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 other=[l for l in q.splitlines() if l.strip() and int(l.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('다른 GPU compute 작업 감지: '+str(other))
def main():
 assert json.loads((ROOT/'checks/stage1_gate.json').read_text())['status']=='PASS';assert json.loads((ROOT/'stage2/analysis_summary.json').read_text())['status']=='PASS'
 guard();torch.set_num_threads(2);samples,train,val=D.S.dataset();mod=D.S.explicit_module();net=D.S.build_network(mod,torch.tensor(D.LO,dtype=torch.float32,device='cuda'),torch.tensor(D.SIZE,dtype=torch.float32,device='cuda')).cuda();ck=torch.load(TRAIN/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True);assert ck['epoch']==155;net.load_state_dict(ck['model_state_dict']);net.eval();outrows=[]
 for at in range(0,len(val),4):
  guard();ids=val[at:at+4];x=torch.stack([samples[i]['tensors'][0] for i in ids]).cuda();box=torch.stack([samples[i]['bbox'] for i in ids]).cuda()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):pred=net.inference_batch(x,box)
  for j,i in enumerate(ids):
   s=samples[i];mask=pred['pred_mask'][j].cpu().numpy()>0;uv=pred['coord_2d_image'][j].float().cpu().numpy();xyz=pred['pred_front_code_obj'][j].float().cpu().numpy();r=D.pose('pbr_repro',i,mask,xyz,uv,s['K'],s['R'],s['t'].reshape(3));gt=s['tensors'][1].numpy()>.5;r['mask_iou']=float(np.count_nonzero(mask&gt)/max(np.count_nonzero(mask|gt),1));outrows.append(r)
 ref=json.loads((TRAIN/'checks/best_loss_pose.json').read_text());refmap={r['frame']:r for r in ref['rows']};assert len(outrows)==len(refmap)==20
 diffs=[abs(r['add']-refmap[r['frame']]['add_error']) for r in outrows if r['valid_pose'] and refmap[r['frame']]['valid_pose']];delta=max(diffs) if diffs else None;iou=max(abs(r['mask_iou']-refmap[r['frame']]['mask_iou']) for r in outrows);valid=sum(r['valid_pose'] for r in outrows);passed=sum(r['add_pass'] for r in outrows)
 gate={'status':'PASS' if valid==20 and passed==1 and len(diffs)==20 and delta<=.001 and iou<=1e-5 else 'FAIL','compared':20,'valid':valid,'ADD_pass':passed,'max_ADD_frame_difference':delta,'max_mask_IoU_frame_difference':iou,'thresholds':'all20 valid, 1/20 ADD, ADD max diff <=0.001 native unit, IoU max diff <=1e-5','mean_ADD':float(np.mean([r['add'] for r in outrows if r['valid_pose']]))}
 with (ROOT/'checks/pbr_repro_gate.json').open('x') as f:json.dump(gate,f,ensure_ascii=False,indent=2)
 with (ROOT/'stage2/pbr_baseline_reproduction.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in outrows for k in r)));w.writeheader();w.writerows(outrows)
 print(json.dumps(gate,ensure_ascii=False,indent=2));assert gate['status']=='PASS'
if __name__=='__main__':
 try:main()
 except Exception as e:
  with (ROOT/'checks/pbr_repro_failure.json').open('x') as f:json.dump({'reason':str(e),'traceback':traceback.format_exc()},f,ensure_ascii=False,indent=2)
  raise
