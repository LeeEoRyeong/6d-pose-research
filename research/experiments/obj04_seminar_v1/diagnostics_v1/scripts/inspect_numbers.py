import csv,json,numpy as np
from pathlib import Path
r=Path(__file__).resolve().parents[1]
def rows(p):return list(csv.DictReader((r/p).open()))
g=rows('supplement/tables/independent_CAD_ray_check.csv')
print('independent geometry',{k:{'mean':float(np.mean([float(x[k]) for x in g])),'max':max(float(x[k]) for x in g)} for k in ['front_cameraZ_error_mean','front_cameraZ_error_max','back_cameraZ_error_mean','back_cameraZ_error_max','A0_RE_with_SO3_GT_deg']})
b=rows('supplement/tables/HCCE_bit_channel_errors.csv')
for branch in ['front','back']:
    for axis in 'xyz':print('bit errors',branch,axis,[round(float(np.mean([float(x['hard_bit_error']) for x in b if x['branch']==branch and x['axis']==axis and int(x['bit_index'])==i])),4) for i in range(8)])
e=rows('supplement/tables/Predicted_XYZ_errors_raw_target.csv');p=json.loads((r/'matched_batch4/checks/prediction_results.json').read_text())['candidates']
for frame in [40,90]:
    print('example',frame,[x for x in e if int(x['frame'])==frame and x['scope']=='GTmask']);print([x for x in p if x['frame']==frame and x['stage']=='front'])
print('geo missing',sum(x['missing_label_GTmask_pixels'] for x in json.loads((r/'checks/geometry.json').read_text())))
