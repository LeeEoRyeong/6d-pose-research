from pathlib import Path
import difflib,json
R=Path(__file__).resolve().parents[1];B=R.parent;source=B/'runs/baseline_seed42_20261103_214353/trained_only/code/network_model_explicit.py';s=source.read_text();t=s
old='def forward(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask,):';new='def forward(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask, valid_front, valid_back,):';assert t.count(old)==1;t=t.replace(old,new)
old='''        pred_mask = pred_mask_prob
          
        Front_L1Loss_v_l = []''';new='''        pred_mask = pred_mask_prob
        # 좌표 L1의 공간 게이트만 GT 객체와 각 XYZ support의 교집합으로 교체한다.
        # 위 pred_mask는 error ratio / dynamic weight history에 그대로 사용한다.
        assert gt_mask.shape == valid_front.shape == valid_back.shape
        front_gate = ((gt_mask > 0.5) & (valid_front > 0.5)).unsqueeze(1).to(dtype=pred_mask.dtype).detach()
        back_gate = ((gt_mask > 0.5) & (valid_back > 0.5)).unsqueeze(1).to(dtype=pred_mask.dtype).detach()
          
        Front_L1Loss_v_l = []''';assert t.count(old)==1;t=t.replace(old,new)
for kind in ['Front','Back']:
 for side in ['pred','gt']:
  axis='front' if kind=='Front' else 'back';base=f'{side}_{axis}[:, k*8:(k+1)*8]'
  if side=='pred':old=f'{base}*pred_mask.detach().clone()';new=f'{base}*{axis}_gate'
  else:old=f'({base} *2 -1)*pred_mask.detach().clone()';new=f'({base} *2 -1)*{axis}_gate'
  assert t.count(old)==1,(old,t.count(old));t=t.replace(old,new)
with (R/'code/network_gtmask.py').open('x') as f:f.write(t)
diff=''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(source),tofile=str(R/'code/network_gtmask.py')))
with (R/'checks/loss_change.diff').open('x') as f:f.write(diff)
assert 'cal_error_ratio(pred_front' not in diff and 'cal_error_ratio(pred_back' not in diff
with (R/'checks/implementation_scope.json').open('x') as f:json.dump({'source_copy':str(source),'modified_copy':str(R/'code/network_gtmask.py'),'changes':'signature adds branch supports; GT∩branch validity only replaces pred gate at Front/Back L1; predicted Mask remains for both error ratios and 100-step dynamic history','Mask_loss_and_3x_coefficients_and_mean':'unchanged'},f,ensure_ascii=False,indent=2)
print(diff)
