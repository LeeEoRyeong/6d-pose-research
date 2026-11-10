import csv,collections,numpy as np,json
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'full'
with (O/'per_trial.csv').open(newline='') as f:rows=list(csv.DictReader(f))
a={(r['frame'],r['condition'],r['method']):r for r in rows if r['mode']=='original_order'};out=[]
for scope in ['all','baseline','changed_roi']:
 for method in sorted({r['method'] for r in rows if r['mode']=='original_order'}):
  items=[r for r in rows if r['mode']=='original_order' and r['method']==method and (scope=='all' or r['scope']==scope)];gain=loss=both=0;delta=[];rat=[]
  for r in items:
   x=a[(r['frame'],r['condition'],'A_all')];was=x['success']=='True';now=r['success']=='True';gain+=int(now and not was);loss+=int(was and not now)
   if was and now:
    both+=1;delta.append(float(r['full_median_error_px'])-float(x['full_median_error_px']));rat.append(float(r['full_final_ratio'])-float(x['full_final_ratio']))
  out.append({'scope':scope,'method':method,'cases':len(items),'pnp_gain_vs_A':gain,'pnp_loss_vs_A':loss,'both_pnp_returned':both,'median_full_error_delta_vs_A_px':float(np.median(delta)) if delta else '', 'median_full_ratio_delta_vs_A':float(np.median(rat)) if rat else '', 'fraction_full_error_lower':float(np.mean([x<0 for x in delta])) if delta else ''})
with (O/'paired_vs_all.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('PAIRED DONE')
