import tempfile,subprocess,csv,json
from pathlib import Path
from app import Store,png,mask_decode
with tempfile.TemporaryDirectory() as t:
 t=Path(t);s=Store(t/'ann');fr='000075';a=mask_decode(s.entries[fr]['candidates']['C_CAD_SAM'].read_bytes());a[0,0]=128;v=s.save(fr,png(a),'REVIEWED','C_CAD_SAM',None,False,'test')
 cmd=['/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python','-B','evaluate.py','--annotations',str(t/'ann'),'--output',str(t/'eval1')];subprocess.run(cmd,check=True,capture_output=True);x=json.loads((t/'eval1/evaluation_info.json').read_text());assert x['accepted_count']==0 and x['result_rows']==0
 s.save(fr,png(a),'ACCEPTED','C_CAD_SAM',v['revision'],True,'test temporary');cmd[-1]=str(t/'eval2');subprocess.run(cmd,check=True,capture_output=True);x=json.loads((t/'eval2/evaluation_info.json').read_text());assert x['accepted_count']==1 and x['result_rows']==5
 with (t/'eval2/per_frame.csv').open(encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
 c=next(r for r in r if r['method']=='C_CAD_SAM');assert float(c['iou'])==1 and int(c['ignore_pixels'])==1
 print('REVIEWED excluded; ACCEPTED included; ignore excluded; six-frame RobotSeg availability PASS (temporary test only)')
