"""선택규칙, 전체 평가 기록, 점수 집계 및 산출물을 독립 재확인한다."""
from pathlib import Path
import json,csv,math,hashlib,ast,re,subprocess
import numpy as np
R=Path(__file__).resolve().parents[1];B=R.parent

def rd(p):return json.loads(p.read_text())
def csvr(p):return list(csv.DictReader(p.open()))
summary=rd(R/'checks/training_summary.json');record=[json.loads(x) for x in (R/'logs/evaluation.jsonl').open()];trained=[r for r in record if r['epoch']>=1]
best=float('inf');official=None;wait=0
for row in trained:
 if row['val']['total_loss']<best-.0001:best=row['val']['total_loss'];official=row['epoch'];wait=0
 else:wait+=1
secondary=min(trained,key=lambda x:x['val_XYZ_score'])['epoch'];assert official==summary['official']['epoch']==170 and secondary==summary['secondary']['epoch']==195
must=['total_loss','mask_loss','front_HCCE_loss_3x','back_HCCE_loss_3x','GT_missed_fraction','front_XYZ_L2','back_XYZ_L2','front_Z0_error','back_Z0_error','front_x_MAE','front_y_MAE','front_z_MAE','back_x_MAE','back_y_MAE','back_z_MAE','mask_iou','front_PnP_valid','front_ADD_pass','back_PnP_valid','back_ADD_pass']
for row in record:
 for split in ['train','val']:
  for key in must:assert key in row[split] and row[split][key] is not None and math.isfinite(float(row[split][key])),(row['epoch'],split,key)
steps=[json.loads(x) for x in (R/'logs/train_steps.jsonl').open()];assert len(steps)==summary['optimizer_attempts']==3900;assert sum(bool(x['optimizer_step_applied']) for x in steps)==summary['optimizer_steps']==3899
first=json.loads((R/'checks/selection_frozen.json').read_text());officialfile=Path(first['official_checkpoint']);assert hashlib.sha256(officialfile.read_bytes()).hexdigest()==first['official_sha256'];assert not first['real27_used_for_selection']
real=csvr(R/'tables/real27_metrics.csv');pbr=csvr(R/'tables/pbr_validation_per_frame.csv');compare=csvr(R/'tables/pbr_validation_metrics.csv');assert len(real)==27 and len({x['frame'] for x in real})==27 and len(pbr)==20
real_sum=rd(R/'checks/real27_summary.json');assert sum(x['valid_pose']=='True' for x in real)==real_sum['PnP_valid'] and sum(x['add_pass']=='True' for x in real)==real_sum['ADD_pass'] and real_sum['real27_inference_runs']==1
new=next(x for x in compare if x['model']=='GT_mask_loss' and x['branch']=='front');assert abs(np.mean([float(x['new_XYZ_L2']) for x in pbr])-float(new['XYZ_L2']))<1e-5 and sum(x['new_ADD_pass']=='True' for x in pbr)==int(new['ADD_pass'])
required=['configs/experiment_config.json','checks/loss_equivalence.json','checks/mask_gate_validation.json','checks/initialization.json','checkpoints/best_val_loss.pt','checkpoints/best_val_xyz.pt','tables/train_history.csv','tables/pbr_validation_metrics.csv','tables/real27_metrics.csv','reports/GT_Mask_Loss_개선실험_최종보고서.md','reports/세미나_최종실험_요약.md','figures/xyz_L2_comparison.png','figures/ADD_comparison.png','figures/Z0_error_comparison.png','figures/train_val_loss.png','figures/validation_xyz_comparison.png','figures/real27_priority_overlay.png']
assert all((R/x).is_file() for x in required);assert len(list((R/'figures/real27_overlays').glob('*.png')))==27
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in (R/'reports').glob('*.md'):
 for q in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(q).is_file(),q
manifest=[]
for p in sorted(R.rglob('*')):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  manifest.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
with (R/'checks/final_artifact_manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
result={'status':'PASS','official_epoch_replayed':official,'secondary_epoch_replayed':secondary,'evaluation_records_including_initial':len(record),'each_train_val_metric_recorded':True,'optimizer_attempts':len(steps),'optimizer_steps':summary['optimizer_steps'],'real_frames':len(real),'real_inference_runs':1,'pbr_pair_frames':len(pbr),'required_artifacts':len(required),'artifacts_hashed':len(manifest),'GPU_compute_final':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),'retraining_runs':1}
with (R/'checks/final_audit.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
