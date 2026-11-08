"""Reproduce previous common oracle evaluator, without modifying its files."""
import sys,json,csv,torch,numpy as np
from pathlib import Path
OLD=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_symmetry_eval_20261108')
sys.path.insert(0,str(OLD))
import evaluate as e
import symmetry
HERE=Path(__file__).resolve().parent

def main():
 cv=e.cv2;torch.set_num_threads(2);cv.setNumThreads(1);assert symmetry.validate()['status']=='PASS'
 apps=e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not apps,apps
 v,faces,syms,info=symmetry.load();data,_=e.get_data(v);mod=e.explicit_module();local=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([local['min_'+a] for a in 'xyz']);sz=np.array([local['size_'+a] for a in 'xyz']);a=torch.tensor(lo,dtype=torch.float32,device='cuda');b=torch.tensor(sz,dtype=torch.float32,device='cuda')
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):off=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 base=torch.load(e.RUN/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=True)
 out=HERE/'reproduction';out.mkdir(exist_ok=False);first={26,51,78,99,126};net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=a,size_xyz=b)
 rows,po=e.run_one('official',net,off,[('real_first5',[x for x in data['real'] if x['id'] in first]),('real_remaining',[x for x in data['real'] if x['id'] not in first])],syms,v,faces,info,lo,sz,out,True);del net;torch.cuda.empty_cache()
 net=e.build_network(mod,a,b);rr,pb=e.run_one('baseline',net,base,[('real',data['real'])],syms,v,faces,info,lo,sz,out,True);rows+=rr
 old=list(csv.DictReader((OLD/'run_20261108T143139/per_frame.csv').open()));lookup={(r['model'],int(r['frame'])):r for r in old if r['domain'].startswith('real')}
 diffs=[]
 for r in rows:
  prev=lookup[(r['model'],r['frame'])];assert r['pnp_valid']==(prev['pnp_valid']=='True')
  for k in ('ADD','ADD_S','symmetry_ADD'):
   if r[k] is not None:diffs.append(abs(float(r[k])-float(prev[k])))
 count=sum(r['symmetry_ADD_pass'] for r in rows if r['model']=='official');base_count=sum(r['symmetry_ADD_pass'] for r in rows if r['model']=='baseline');mx=max(diffs,default=0)
 result={'status':'PASS' if count==21 and base_count==0 and mx<.001 else 'FAIL','official_symmetry_pass':count,'baseline_symmetry_pass':base_count,'total_per_model':27,'max_abs_metric_difference':mx,'threshold':.001,'comparison':'per frame ADD, ADD-S, explicit symmetry ADD and PnP validity; prior fixed seed','official_peak':po,'baseline_peak':pb}
 with (HERE/'reproduction.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 fields=list(rows[0]);
 with (HERE/'per_frame.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows])
 print('REPRODUCTION',json.dumps(result),flush=True)
 if result['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
