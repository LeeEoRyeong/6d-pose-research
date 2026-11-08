"""기존 step 로그의 모든 epoch와 평가 주기·실제 저장 checkpoint를 결합한다."""
from pathlib import Path
import csv,json,collections,shutil
R=Path(__file__).resolve().parents[1];B=R.parent;O=B/'runs/baseline_seed42_20261103_214353/trained_only';A=B/'agent_run_v1'
inv=list(csv.DictReader((R/'checkpoint_inventory.csv').open()));comp=list(csv.DictReader((R/'checkpoint_comparison.csv').open()))
shutil.copyfile(R/'epoch_loss_history.csv',R/'tables/evaluation_epoch_history_initial.csv')
shutil.copyfile(R/'checkpoint_comparison.csv',R/'tables/checkpoint_comparison_initial.csv')
records=[];not_saved=[];maxdiff=0.
for model,epath,spath,cfgpath in [('baseline',O/'logs/epoch_metrics.jsonl',O/'logs/train_steps.jsonl',O/'config.json'),('rotation',A/'logs/rotation_epochs.jsonl',A/'logs/rotation_steps.jsonl',A/'stage3/config.json')]:
 evals={x['epoch']:x for x in map(json.loads,epath.open())};steps=collections.defaultdict(list)
 for x in map(json.loads,spath.open()):steps[x['epoch']].append(x)
 cfg=json.loads(cfgpath.read_text());last_eval=0
 for e in sorted({0}|set(steps)|set(evals)):
  sl=steps[e];v=evals.get(e);files=[x['filename'] for x in inv if x['model']==model and x['epoch'] and int(x['epoch'])==e];matches=[x for x in comp if x['model']==model and int(x['epoch'])==e]
  values={'model':model,'epoch':e,'optimizer_steps_cumulative':sl[-1]['optimizer_steps_total'] if sl else (v['optimizer_steps'] if v else ''),'train_optimization_loss':sum(x['loss'] for x in sl)/len(sl) if sl else '', 'train_evaluation_loss':v['train_eval']['loss'] if v else '', 'validation_loss':v['val']['loss'] if v else '', 'train_mask_iou':v['train_eval']['mask_iou_mean'] if v else '', 'val_mask_iou':v['val']['mask_iou_mean'] if v else '', 'mask_loss_historical':'N/A','front_HCCE_loss_historical':'N/A','back_HCCE_loss_historical':'N/A','learning_rate_config':cfg['lr'],'learning_rate_per_step_historical':'N/A','amp_skips_in_epoch':sum(not x['optimizer_step_applied'] for x in sl),'amp_skips_since_last_validation':sum(not x['optimizer_step_applied'] for i in range(last_eval+1,e+1) for x in steps[i]) if v and e>0 else '', 'validation_recorded':bool(v), 'saved_checkpoint_names':';'.join(files),'component_losses_are_recomputed_not_historical':bool(matches)}
  if matches:
   m=matches[0]
   for key in ['mask_loss_recomputed','front_HCCE_weighted_loss_recomputed','back_HCCE_weighted_loss_recomputed','val_loss_recomputed']:values[key]=m[key]
   if v:maxdiff=max(maxdiff,abs(float(m['val_loss_recomputed'])-v['val']['loss']))
  records.append(values)
  if v:
   if not files:not_saved.append({'model':model,'epoch':e,'validation_loss':v['val']['loss'],'checkpoint':'N/A — 해당 epoch 저장 checkpoint 미발견','status':'미평가; Loss 로그만 존재'})
   last_eval=e
for x in comp:
 x['gt_gate_counterfactual_total_fixedweights']=float(x['mask_loss_recomputed'])+float(x['front_HCCE_GTgate_fixedweights_counterfactual'])+float(x['back_HCCE_GTgate_fixedweights_counterfactual'])
 x['gt_gate_counterfactual_minus_actual']=x['gt_gate_counterfactual_total_fixedweights']-float(x['val_loss_recomputed'])
 assert abs(sum(float(x[k]) for k in ['mask_loss_recomputed','front_HCCE_weighted_loss_recomputed','back_HCCE_weighted_loss_recomputed'])-float(x['val_loss_recomputed']))<2e-6
keys=list(dict.fromkeys(k for x in records for k in x))
with (R/'epoch_loss_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
with (R/'checkpoint_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comp[0]));w.writeheader();w.writerows(comp)
with (R/'tables/unsaved_evaluation_epochs.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(not_saved[0]));w.writeheader();w.writerows(not_saved)
with (R/'checks/history_check.json').open('x') as f:json.dump({'status':'PASS','all_epoch_rows':len(records),'not_saved_evaluation_epochs':len(not_saved),'max_checkpoint_recomputed_vs_logged_loss_abs_difference':maxdiff,'historical_component_loss':'N/A; checkpoint-only recomputation clearly marked','learning_rate':'configured 0.0002; no historical per-step LR log','AMP_skips_rotation_epoch32':1},f,ensure_ascii=False,indent=2)
print(json.dumps({'epoch_rows':len(records),'not_saved':len(not_saved),'max_recomputed_loss_diff':maxdiff},ensure_ascii=False))
