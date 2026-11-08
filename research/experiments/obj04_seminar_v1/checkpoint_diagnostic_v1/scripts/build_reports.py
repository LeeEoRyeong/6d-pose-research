"""실제 저장 checkpoint 및 로그만으로 한국어 진단 보고서와 그림을 생성."""
from pathlib import Path
import csv,json,hashlib,re,datetime,subprocess,math,ast
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];B=R.parent;O=B/'runs/baseline_seed42_20261103_214353/trained_only';A=B/'agent_run_v1';P=B/'diagnostics_v1';T=B/'diagnostics_train_val_v1'
def rd(p):return json.loads(p.read_text())
def csvr(p):return list(csv.DictReader(p.open()))
def save(p,s):
 with p.open('x') as f:f.write(s)
def js(p,v):save(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def link(p):return f'[{p.name}]({p})'
def num(v):return float(v)
def pretty(v,n=4):return 'N/A' if v is None or v in ('','N/A') else (f'{float(v):.{n}f}' if abs(float(v))<1e6 else f'{float(v):.3e}')
C=csvr(R/'checkpoint_comparison.csv');G=csvr(R/'correspondence_geometry.csv');H=csvr(R/'epoch_loss_history.csv');I=csvr(R/'checkpoint_inventory.csv');assert len(C)==len(I)==10 and len(G)==400
get=lambda model,epoch:next(x for x in C if x['model']==model and int(x['epoch'])==epoch)
b155=get('baseline',155);b195=get('baseline',195);r10=get('rotation',10);r40=get('rotation',40);r50=get('rotation',50);initial=get('diagnostic_initial_reference',0)
# Recheck all originals and all prior run manifests, including the immediate prior study.
integrity={}
for name,p,root in [('원본',B/'checks/original_manifest.json',None),('기존 세미나',B/'checks/artifact_manifest.json',B),('기존 Baseline',O.parent/'checks/final_artifact_manifest.json',O.parent),('이전 Pose 진단',P/'checks/final_artifact_manifest.json',P),('이전 Train/Val 진단',T/'checks/final_artifact_manifest.json',T),('자율 실험',A/'checks/final_artifact_manifest.json',A)]:
 data=rd(p);bad=[]
 for x in data:
  path=Path(x['path']) if root is None else root/x['path']
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=x['sha256']:bad.append(str(path))
 integrity[name]={'count':len(data),'mismatches':bad}
assert all(not x['mismatches'] for x in integrity.values()),integrity
for x in I:
 p=Path(x['file']);assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
integrity['GPU_compute_final']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();js(R/'checks/final_integrity.json',integrity)
# Plots: one point per actually saved state; do not invent intermediate XYZ values.
def font(n):
 try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
 except OSError:return ImageFont.load_default()
F=font(16);S=font(13);TITLE=font(23);BLUE='#216ba5';RED='#c34c3d';GRAY='#777777'
im=Image.new('RGB',(1250,760),'white');d=ImageDraw.Draw(im);d.text((45,20),'Saved checkpoints: Val loss vs Front XYZ error',font=TITLE,fill='black');d.text((45,55),'Val20, GT BBox ROI; each point is an actually saved checkpoint, epoch0 is untrained reference',font=F,fill='black');l,t,r,b=130,120,1190,610;xmin=.65;xmax=2.15;ymax=100
for i in range(6):
 x=l+(r-l)*i/5;y=b-(b-t)*i/5;d.line((l,y,r,y),fill='#ddd');d.line((x,t,x,b),fill='#eee');d.text((l-55,y-8),str(i*20),font=S,fill='black');d.text((x-12,b+15),f'{xmin+(xmax-xmin)*i/5:.2f}',font=S,fill='black')
for c in C:
 x=l+(num(c['val_loss_recomputed'])-xmin)/(xmax-xmin)*(r-l);y=b-num(c['front_xyz_l2'])/ymax*(b-t);model=c['model'];color=BLUE if model=='baseline' else RED if model=='rotation' else GRAY;label=('B' if model=='baseline' else 'R' if model=='rotation' else 'Init')+c['epoch'];d.ellipse((x-6,y-6,x+6,y+6),fill=color);d.text((x+8,y-16),label,font=S,fill=color)
d.text((525,670),'Recomputed Val loss',font=F,fill='black');d.text((130,85),'Front XYZ L2 (native units)',font=F,fill='black');d.text((120,715),'B = Baseline',font=F,fill=BLUE);d.text((470,715),'R = Rotation',font=F,fill=RED);d.text((820,715),'Init = separate epoch0',font=F,fill=GRAY);im.save(R/'figures/loss_vs_xyz.png')
# Valid PnP and ADD pass counts, same denominator 20.
select=sorted([x for x in C if x['model'] in ('baseline','rotation')],key=lambda x:(x['model']!='baseline',int(x['epoch'])))
im=Image.new('RGB',(1500,850),'white');d=ImageDraw.Draw(im);d.text((45,20),'PBR Val20: actual saved checkpoints',font=TITLE,fill='black');d.text((45,55),'Valid PnP is not ADD success; failures remain in denominator',font=F,fill='black');l,t,r,b=110,150,1450,650
for j in range(5):
 y=b-j/4*(b-t);d.line((l,y,r,y),fill='#ddd');d.text((50,y-8),str(j*5),font=F,fill='black')
for i,c in enumerate(select):
 x=l+20+i*145;v=int(c['front_PnP_valid']);p=int(c['front_ADD_pass'])
 for delta,val,color in [(0,v,BLUE),(55,p,RED)]:
  y=b-val/20*(b-t);d.rectangle((x+delta,y,x+delta+48,b),fill=color);d.text((x+delta,y-22),str(val),font=S,fill=color)
 d.text((x+4,b+16),('B' if c['model']=='baseline' else 'R')+c['epoch'],font=F,fill='black')
d.text((150,760),'Blue = valid PnP',font=F,fill=BLUE);d.text((650,760),'Red = ADD@0.1d passes',font=F,fill=RED);im.save(R/'figures/checkpoint_pose_comparison.png')
# Aggregate geometric eigenvalues are means of per-frame population covariances, not covariance after pooling poses.
def geo(model,epoch):
 rr=[x for x in G if x['model']==model and int(x['epoch'])==epoch and x['branch']=='front'];return {'n':len(rr),**{k:float(np.mean([num(v[k]) for v in rr])) for k in ('x_std','y_std','z_std','unique_xyz','cov_eigenvalue_1','cov_eigenvalue_2','cov_eigenvalue_3','effective_rank')}}
geom={(model,ep):geo(model,ep) for model,ep in [('baseline',155),('baseline',195),('rotation',10),('rotation',40),('rotation',50)]}
# Compute medians without replacing huge but finite poses with artificial cutoffs.
median={}
for ep in (40,50):
 vals=[num(x['PnP_add']) for x in G if x['model']=='rotation' and int(x['epoch'])==ep and x['branch']=='front' and x['PnP_add'] not in ('','N/A')];median[ep]={'n':len(vals),'median':float(np.median(vals)) if vals else None,'max':max(vals) if vals else None}
js(R/'checks/geometry_summary.json',{'groups':[{'model':m,'epoch':e,**v} for (m,e),v in geom.items()],'rotation_ADD_robust_summary':median,'rank_definition':'exp(-sum(p*log(p))), p=eigenvalues/sum; population covariance, per-frame then arithmetic mean'})
# True logged epoch range and AMP summary.
runstats={}
for model in ('baseline','rotation'):
 hh=[x for x in H if x['model']==model];ee=[x for x in hh if x['validation_loss'] not in ('','N/A') and int(x['epoch'])>=1];best=min(ee,key=lambda x:num(x['validation_loss']));runstats[model]={'best_epoch':int(best['epoch']),'best_loss':num(best['validation_loss']),'last_epoch':max(int(x['epoch']) for x in hh),'amp_skips':sum(int(x['amp_skips_in_epoch']) for x in hh)}
assert runstats['baseline']['best_epoch']==155 and runstats['rotation']['best_epoch']==10
invtable='\n'.join(f"| {x['model']} | {x['epoch']} | `{x['filename']}` | {x['status']} |" for x in I)
# Summary per checkpoint, front/back; all20 denominator except error means over valid.
ct=[]
for x in C:
 ct.append(f"| {x['model']} | {x['epoch']} | {pretty(x['val_loss_recomputed'])} | {pretty(x['mask_iou'])} | {pretty(x['front_xyz_l2'],2)}/{pretty(x['back_xyz_l2'],2)} | {num(x['front_z0_hard_error'])*100:.2f}%/{num(x['back_z0_hard_error'])*100:.2f}% | {x['front_PnP_valid']}/20 | {x['front_ADD_pass']}/20 |")
pnt=[]
for x in C:
 pnt.append(f"| {x['model']} | {x['epoch']} | {x['front_PnP_valid']} | {pretty(x['front_add_mean_valid'],2)} | {pretty(x['front_rotation_deg_mean_valid'],2)} | {pretty(x['front_translation_mean_valid'],2)} | {x['back_PnP_valid']} | {pretty(x['back_add_mean_valid'],2)} | {pretty(x['back_rotation_deg_mean_valid'],2)} | {pretty(x['back_translation_mean_valid'],2)} |")
gt=[]
for (model,e),g in geom.items():gt.append(f"| {model} | {e} | {g['x_std']:.4f}/{g['y_std']:.4f}/{g['z_std']:.4f} | {g['unique_xyz']:.2f} | {g['cov_eigenvalue_1']:.3f}/{g['cov_eigenvalue_2']:.3f}/{g['cov_eigenvalue_3']:.5f} | {g['effective_rank']:.4f} |")
losst=[]
for x in [initial,b155,r10,r40,r50]:losst.append(f"| {x['model']} | {x['epoch']} | {num(x['val_loss_recomputed']):.6f} | {num(x['mask_loss_recomputed']):.6f} | {num(x['front_HCCE_weighted_loss_recomputed']):.6f} | {num(x['back_HCCE_weighted_loss_recomputed']):.6f} | {num(x['GT_foreground_missed_fraction_batchmean'])*100:.2f}% | {num(x['gt_gate_counterfactual_total_fixedweights']):.6f} |")
now=datetime.datetime.now(datetime.timezone.utc);at=rd(R/'checks/evaluation_status.json');
report=f'''# HCCEPose 체크포인트 선택 및 XYZ Collapse 원인 진단

작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}  
결과: `{R}`

**결론: 회전 epoch10의 X축 붕괴는 epoch40·50에서 크게 완화됐지만, 세 checkpoint 모두 PBR Val ADD@0.1d가 0/20이다. Val Loss 기준은 붕괴된 epoch10을 선택했다. GT Mask로 좌표 Loss의 게이트만 바꾼 사후 대조에서 Loss 순서가 뒤집혀, 예측 Mask 게이팅이 선택 결과를 왜곡할 수 있다는 실측 근거가 있다.** 이 대조는 재학습이나 실제 개선 성능이 아니며 모든 학습 실패의 인과 원인을 확정하지 않는다.

- 지정된 두 run의 실제 저장 checkpoint 9개를 모두 평가했다. 손상·미평가 checkpoint는 없었다. 별도 진단 run에 실제 존재하는 untrained epoch0 checkpoint를 **초기 참고** 1개로 추가했다. 두 학습 run의 추가 epoch checkpoint를 복원하거나 새로 만들지 않았다.
- 같은 PBR Val20, epoch별 고정 checkpoint, GT visible BBox oracle crop, batch4, 기존 BGR 정규화, GPU AMP float16, 원본 HCCE 복호화·RANSAC EPNP 설정을 사용했다.
- Train optimizer·재학습·실사27 평가나 선택·코드/모델 수정은 수행하지 않았다. 회전 epoch40/50 평가는 사용자가 요청한 사후 진단이며 독립 테스트가 아니다.
- 전체 길이 단위는 PBR/CAD 원본 수치 단위다. mm를 확정하지 않는다. PnP 유효는 solver 성공·유한 오차·CAD 양의 깊이 기준이며 ADD 통과와 다르다.

## 1. 직접 답변

1. **epoch10과 last 중 XYZ가 좋은 쪽:** 회전 last epoch50이다. Front 평균 L2 {num(r10['front_xyz_l2']):.4f}→{num(r50['front_xyz_l2']):.4f}, Back {num(r10['back_xyz_l2']):.4f}→{num(r50['back_xyz_l2']):.4f}. Front PnP 유효0→20이지만 ADD는 **둘 다0/20**이다. epoch40의 Front L2 {num(r40['front_xyz_l2']):.4f}는 세 저장 시점 중 더 낮지만 ADD도0/20이다.
2. **Loss 선택이 붕괴를 가렸는가:** 저장된 범위에서 그렇다. 선택된 epoch10의 Front X 표준편차 {num(r10['front_x_std_mean']):.4f}와 unique XYZ {num(r10['front_unique_xyz_mean']):.1f}는 epoch50의 {num(r50['front_x_std_mean']):.4f}, {num(r50['front_unique_xyz_mean']):.1f}보다 매우 작다. 이는 선택된 상태의 붕괴와 후속 상태의 복구를 보여주지만, 낮은 Loss가 모든 붕괴를 잡지 못한다는 범위의 결론이다.
3. **학습 부족·선택 오류·다른 문제:** epoch10 이후 X 분산과 XYZ가 개선되어 초기 상태를 고른 영향이 확인된다. 하지만 epoch40/50의 ADD도0이어서 선택 오류만으로 최종 정확도 실패가 설명되지는 않는다. 회전 증강·시점 다양성·손실 설계·추가 학습량의 개별 인과는 미확정이다.
4. **재학습 없이 해결 가능한가:** 실제 저장된 epoch40/50으로 PnP 반환과 XYZ의 일부를 회복할 수 있다. 이 Val에서는 ADD가 여전히0/20이므로 **정확도 문제가 해결됐다는 증거는 없다.** 본 작업에서 실사 checkpoint 선택·추론은 하지 않았다.
5. **다음에 바꿀 한 가지:** PBR 검증의 checkpoint 선택 점수만 기존 총 Val Loss에서 **GT 유효 Mask 내부 Front/Back XYZ L2의 평균**으로 바꾸는 사전 고정 대조를 제안한다. 다른 학습·증강·PnP 조건은 유지한다. 이 저장 checkpoint 집합에서는 epoch40이 선택되지만 ADD0/20이므로 Pose 개선 보장은 없고, 같은 Val을 보고 고른 사후 관찰은 독립 검증 결과가 아니다.

## 2. 실제 파일과 학습 기록

| run | 저장 epoch | 실제 checkpoint | 상태 |
|---|---:|---|---|
{invtable}

초기 참고 checkpoint는 `runs/baseline_seed42_20261103_214353/checkpoints/best_val_loss.pt`의 **별도 초기 진단 run epoch0**이다. 실제 저장된 파일을 읽었으며 학습 checkpoint로 포장하지 않았다. 기존 회전 run의 초기화 state가 이 파일과 같다는 기록은 `{A/'checks/rotation_initialization.json'}`이다.

| 구분 | 학습 후 평가된 최저 Val Loss | 학습 종료 | optimizer 실제 적용 | AMP skip | 정지 사유 |
|---|---:|---:|---:|---:|---|
| Baseline | epoch155, {runstats['baseline']['best_loss']:.6f} | epoch195 | 3900 | {runstats['baseline']['amp_skips']} | Val Loss early stopping |
| 회전 증강 | epoch10, {runstats['rotation']['best_loss']:.6f} | epoch50 | 999 | {runstats['rotation']['amp_skips']} | Val Loss early stopping |

무학습 epoch0의 logged Val Loss 0.781321은 두 run의 trained checkpoint 선택 후보에서 의도적으로 제외됐다. 회전 run의 AMP skip은 epoch32에서1회다. Train step 로그는 모든 epoch의 총 optimizer Loss를 포함하고, Train 평가·Val Loss·Mask IoU는 epoch1과 매5epoch에 기록됐다. 각 epoch의 값과 기록되지 않은 평가는 {link(R/'epoch_loss_history.csv')}에 있다.

기존 로그에는 epoch별 Mask·Front·Back 개별 Loss와 step별 learning rate가 없다. 해당 필드는 **N/A**이며 checkpoint에서 이번에 재계산한 성분과 구분한다. config와 코드에는 Adam lr=0.0002가 고정돼 있고 scheduler는 확인되지 않았지만, 누락된 과거 step LR 로그를 새로 측정했다고 주장하지 않는다. 저장되지 않은 평가 epoch {len(csvr(R/'tables/unsaved_evaluation_epochs.csv'))}개의 목록: {link(R/'tables/unsaved_evaluation_epochs.csv')}.

## 3. 원본 Loss와 실제 게이트 반사실

원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`의 `HccePose_Loss.forward`는 다음을 계산한다.

- Mask Loss: 예측 sigmoid와 GT Mask의 L1을 crop 전체 픽셀에서 평균.
- Front/Back HCCE: **예측 sigmoid Mask를 round·detach해 각 코드와 GT 코드 양쪽에 곱한 뒤**, 축별 L1을 batch·height·width 전체에서 평균. GT Mask 또는 유효 라벨 Mask를 직접 좌표 Loss의 게이트로 쓰지 않는다. 예측 전경 밖 GT 전경의 좌표 오차는 이 항에서 0이 된다. 반대로 예측된 배경의 코드 오차는 포함될 수 있다.
- 축별8 plane의 동적 가중치는 예측 Mask 내부의 hard error를 `pred_mask.sum()+1`로 정규화하고 길이100 history를 갱신해 계산한다. Soft HCCE target은 `2*GT_code-1`이다. 총 Loss는 Mask + 3×Front 축별 합 + 3×Back 축별 합이다. 근거: `{O/'scripts/support.py'}`의 `loss_value`.

각 checkpoint를 같은 순서의 Val20에서 **새 Loss history**로 평가한 총 Loss는 원래 기록과 최대 원소 차이 {rd(R/'checks/history_check.json')['max_checkpoint_recomputed_vs_logged_loss_abs_difference']:.1f}였다. 개별 성분은 이때의 사후 재계산이며 과거 epoch별 train 로그가 아니다.

반사실에서는 원래 계산에서 얻은 **각 checkpoint의 예측 Mask history·동적 가중치와 원래 Mask Loss를 그대로 고정**하고, Front/Back 코드 항의 gate만 GT Mask로 바꿨다. 다시 학습하지 않았으며 가중치를 GT 기준으로 갱신하지도 않았다. 따라서 실제 GT-gated 재학습 Loss와는 다르다. 독립 예측-gate 수식을 두 checkpoint 첫 배치의 원본 값과 비교한 최대 차이는1.5e-8 이하다. {link(R/'checks/counterfactual_formula_check.json')}.

| 모델 | epoch | 실제 총 Val Loss | Mask Loss | 3×Front | 3×Back | GT 전경 누락률 | GT gate 반사실 총합 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(losst)}

**실제 기준:** 회전 epoch10 {num(r10['val_loss_recomputed']):.6f} < epoch50 {num(r50['val_loss_recomputed']):.6f}. **고정 가중치 GT gate 반사실:** epoch10 {num(r10['gt_gate_counterfactual_total_fixedweights']):.6f} > epoch50 {num(r50['gt_gate_counterfactual_total_fixedweights']):.6f}. epoch10은 GT 전경의 평균 {num(r10['GT_foreground_missed_fraction_batchmean'])*100:.2f}%를 예측 gate에서 누락했고 epoch50은 {num(r50['GT_foreground_missed_fraction_batchmean'])*100:.2f}%였다. 같은 라벨 배열·가중치를 고정한 게이트 교체가 저장된 두 시점의 Loss 순서를 뒤집는다는 직접 증거다. 두 모델 각각의 history 가중치는 기존 예측 Mask로 형성됐다. 원본 Loss와 동일하게 GT Mask 안의 별도 무효 라벨도 추가 필터링하지 않았으므로 이 반사실은 GT 유효 Mask 기반 선택 지표와 구분한다.

별도 untrained epoch0 참고에서는 pred gate가 전체 crop의 {num(initial['pred_mask_fraction'])*100:.2f}%, GT Mask가 {num(initial['GT_mask_fraction'])*100:.2f}%였고 GT 전경의 약 {num(initial['GT_foreground_missed_fraction_batchmean'])*100:.2f}%가 누락됐다. 총 Loss가 {num(initial['val_loss_recomputed']):.6f}로 낮지만 Front XYZ는 {num(initial['front_xyz_l2']):.4f}였다. 초기 reference Mask IoU는 이번 원본 `inference_batch` FP16 sigmoid threshold에서 {num(initial['mask_iou']):.6f}이며 과거 로그의 `raw_mask>0` IoU와 임계값 반올림 때문에 소폭 다를 수 있다. 참고 상태는 선택 후보에서 제외됐다.

## 4. 실제 checkpoint별 Val20 평가

| run | epoch | 재계산 Val Loss | Mask IoU | Front/Back XYZ L2 | Front/Back Z0 오류 | Front 유효 PnP | Front ADD 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ct)}

XYZ는 GT Mask와 유효 Front/Back label이 공통으로 있는 crop 픽셀에서 측정했다. Mask IoU는 전체 crop 기준이다. HCCE hard bit는 원본의 FP16 `sigmoid(raw)>0.5` 기준으로 평가했다. PBR `min_xyz/size_xyz`, channel0=X,1=Y,2=Z, 원본 encoder/decoder·crop/정규화는 두 run에서 동일하며 checkpoint는 strict loading을 통과했다. 이전 256값 HCCE round-trip과 GT→PnP Oracle 결과도 이 경로에 대한 근거지만, 이것만으로 회전 데이터의 통계적 적합성을 증명하지는 않는다.

| run | epoch | Front 유효 n | Front 평균 ADD | Front 평균 RE(°) | Front 평균 TE | Back 유효 n | Back 평균 ADD | Back 평균 RE(°) | Back 평균 TE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pnt)}

평균 오차는 **유효 Pose만** 대상으로 하며, 유효 반환이0이면 N/A다. ADD 통과율은 실패를 포함해 항상20으로 나눈다. 회전 epoch50 Front 평균 ADD {num(r50['front_add_mean_valid']):.4f}는 최대 {median[50]['max']:.4f}인 극단적 유한 값에 민감하며 중앙값은 {median[50]['median']:.4f}다. 유효 반환만을 정확한 Pose로 해석하지 않는다. ADD·RE·TE 평균은 유효 n이 다른 checkpoint 사이에서 직접적인 전체 샘플 평균 비교가 아니다.

Baseline의 epoch155는 ADD1/20, epoch195는3/20이지만 Val Loss는 epoch155가 더 낮았다. 이 작은 차이는 같은 Val20에 대한 사후 관찰로, 독립 Pose 성능이나 우월한 checkpoint 정책을 입증하지 않는다.

## 5. 3D Correspondence 붕괴

예측 Mask 내부 Front 3D 점의 축별 표준편차와 unique XYZ 수를 프레임마다 구했다. 공분산은 각 프레임의 `(X-mean)ᵀ(X-mean)/N`이고, 고유값을큰 순서로 표기했다. effective rank는 `exp(-sum(p*log(p)))`, `p=λ/sum(λ)`이다. 아래 값은 프레임별 통계의20장 평균이며 전체 점을 한데 합친 공분산이 아니다.

| run | epoch | X/Y/Z std 평균 | unique XYZ 평균 | covariance 고유값 평균 λ1/λ2/λ3 | effective rank 평균 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(gt)}

회전 epoch10의 X std {geom[('rotation',10)]['x_std']:.4f}, λ3 {geom[('rotation',10)]['cov_eigenvalue_3']:.5f}와 상대적으로 낮은 rank는 예측 3D 점이 한 축에 거의 몰렸음을 보여준다. 같은 epoch의 Y/Z 표준편차는 24.36/19.19이므로 모든 좌표가 같은 상수라는 뜻은 아니다. epoch40과50에서는 X std가 22.25/23.95로 회복되고 unique 점 수도 증가했다. 그러나 40/50 모두 ADD0/20이라 분산 회복만으로 Pose가 정확하다고 판단하면 안 된다.

전체 Front/Back 및 프레임별 고유값, rank, PnP 반환·실패: {link(R/'correspondence_geometry.csv')}. 시각화: {link(R/'figures/loss_vs_xyz.png')}, {link(R/'figures/checkpoint_pose_comparison.png')}.

## 6. 가설별 판정

| 가설 | 지지하는 관측 | 반박·제한 | 판정 |
|---|---|---|---|
| A. 회전 학습 자체가 충분히 진행되지 않음 | epoch10→40/50에서 X 다양성, XYZ L2와 PnP 반환이 개선 | epoch40/50도 ADD0/20; 더 긴 학습의 효과는 미실행 | 초기 단계 부족은 가능. 추가 학습만으로 해결되는지는 미확정 |
| B. 더 나은 좌표를 학습했지만 Val Loss가 잘못 선택 | 낮은 Val Loss의 epoch10은 X 붕괴·PnP0; epoch40/50은 더 낮은 XYZ와 PnP20 | 나중 모델 ADD도0/20, 선택 epoch가 다르면 학습 단계도 달라짐 | checkpoint 선택이 붕괴를 가린 증거는 강함. 선택 변경만의 Pose 개선은 미입증 |
| C. Predicted Mask Loss 게이팅으로 품질과 Loss가 분리 | epoch10 GT 전경 누락15.27%; 동일 가중치 GT gate 대조에서 epoch10/50 Loss 순서가 역전 | GT gate 대조는 학습 없이 고정된 history에서 계산. Mask 이외 코드 값·가중치도 달라짐 | 현재 checkpoint 선택 왜곡의 직접적인 실측 근거. 전체 학습 실패의 단독 인과는 미확정 |
| D. 회전 구현 또는 데이터 분포 문제 | 같은 Val에서 회전 epoch50 XYZ {num(r50['front_xyz_l2']):.2f}, Baseline epoch50 {num(get('baseline',50)['front_xyz_l2']):.2f}; augmented 학습 분포가 변경됨 | 이전 0° 동일성, 382,368개 독립 최근접 표본의 불일치0, affine/HCCE CPU 검증 통과. 한 seed, optimizer skip도 1회 차이 | 단순 좌표 불일치 근거는 약함. 증강 분포의 실제 영향은 별도 통제 실험 필요 |

기존 Val20을 이미 학습 checkpoint 선택과 이번 진단에 사용했다. 새로운 선택 규칙을 이 결과에 맞춰 반복 설계하거나 이 수치를 독립 최종 테스트로 보고하지 않는다.

## 7. 다음 한 가지 제안과 범위

다음 대조에서 **checkpoint 선택 점수만 `0.5 × (Front XYZ L2 + Back XYZ L2)`로 변경**하고, 각 L2는 PBR Val의 GT Mask∩유효 label에서 계산한다. 기존 seed·데이터 분할·모델·optimizer·증강·학습 budget·PnP를 유지하고 변경 전 점수·선택 epoch를 사전 기록한다. GT는 PBR validation의 모델 선택에만 사용하며 실사27을 선택에 쓰지 않는다. 이 분석의 세 저장 회전 checkpoint에서 사후 최소 XYZ는 epoch40이지만 그 ADD가0/20이므로, 이 제안은 **개선 보장이 없는 가설**이다. 새 실험이나 코드 수정은 실행하지 않았다.

## 8. 산출물과 보존

| 산출물 | 내용 |
|---|---|
| {link(R/'checkpoint_inventory.csv')} | 실존9개와 별도 epoch0 참고1개, 파일 크기·해시·strict loading 상태 |
| {link(R/'epoch_loss_history.csv')} | Train 전 epoch, 검증 기록 시점, AMP skip, 미기록 성분 N/A |
| {link(R/'checkpoint_comparison.csv')} | Val Loss 재계산·Mask·XYZ·Z0·PnP·ADD/RE/TE·GT gate 반사실 |
| {link(R/'correspondence_geometry.csv')} | Front/Back 프레임별 std·unique·고유값·rank·PnP |
| {link(R/'tables/checkpoint_frame_metrics.csv')} | 주9개 checkpoint의 프레임별 정확도 |
| {link(R/'figures/loss_vs_xyz.png')} / {link(R/'figures/checkpoint_pose_comparison.png')} | 실제 저장 시점에 한정한 발표용 그림 |
| {link(R/'checks/final_integrity.json')} | 기존 자산·checkpoint 보존 SHA-256 검사 |

기존 원본740파일, 세미나366파일, Baseline61파일, Pose진단102파일, Train/Val진단63파일, 자율실험146파일 manifest를 모두 재검증해 일치했다. 10개 checkpoint의 평가 전후 해시도 일치했다. 기존 전용 venv를 재사용했고 패키지 설치·외부 다운로드·실사27 사용은 없었다. 종료 GPU compute 조회: `{integrity['GPU_compute_final'] or '실행 중인 compute process 없음'}`.

## 9. 5줄 핵심 결론

1. 회전 last epoch50의 XYZ는 best epoch10보다 낮은 오차를 보이지만 ADD는 둘 다0/20이다.
2. 선택된 epoch10의 X축 붕괴는 epoch40·50에서 완화됐으므로 Val Loss는 좌표 붕괴를 가렸다.
3. GT Mask 게이트를 적용한 고정 가중치 반사실에서 epoch10·50의 Loss 순서가 뒤집혔다.
4. 재학습 없이 저장 checkpoint로 PnP 반환은 회복할 수 있지만 Pose 정확도 해결은 확인되지 않았다.
5. 다음에는 PBR GT 유효 Mask의 Front/Back XYZ L2만 checkpoint 선택 점수로 바꾸는 대조를 사전 고정한다.
'''
save(R/'reports/Checkpoint_선택_원인분석.md',report)
summary=f'''# 세미나 추가 진단 요약

**핵심:** 회전 epoch10은 예측 3D X축이 붕괴했지만 Val Loss 기준으로 선택됐다. epoch40·50에서는 X 분산과 PnP 반환이 회복됐고, ADD는 여전히0/20이다.

| 지표 — PBR Val20 | 회전 epoch10 | 회전 epoch40 | 회전 epoch50 |
|---|---:|---:|---:|
| 실제 Val Loss | {num(r10['val_loss_recomputed']):.4f} | {num(r40['val_loss_recomputed']):.4f} | {num(r50['val_loss_recomputed']):.4f} |
| GT gate 반사실 합 | {num(r10['gt_gate_counterfactual_total_fixedweights']):.4f} | {num(r40['gt_gate_counterfactual_total_fixedweights']):.4f} | {num(r50['gt_gate_counterfactual_total_fixedweights']):.4f} |
| Front XYZ L2 | {num(r10['front_xyz_l2']):.2f} | {num(r40['front_xyz_l2']):.2f} | {num(r50['front_xyz_l2']):.2f} |
| Front X std | {num(r10['front_x_std_mean']):.4f} | {num(r40['front_x_std_mean']):.4f} | {num(r50['front_x_std_mean']):.4f} |
| Front 유효 PnP | {r10['front_PnP_valid']}/20 | {r40['front_PnP_valid']}/20 | {r50['front_PnP_valid']}/20 |
| Front ADD 통과 | {r10['front_ADD_pass']}/20 | {r40['front_ADD_pass']}/20 | {r50['front_ADD_pass']}/20 |

GT gate는 동적 가중치와 Mask Loss를 원래 값으로 고정한 **사후 반사실**이다. GT를 사용하는 실제 추론 결과가 아니다. epoch50 평균 ADD는 극단값에 민감하며 중앙값은 {median[50]['median']:.2f}다.

- 저장된 두 run의 checkpoint9개와 별도 실제 epoch0 초기 참고1개를 평가했다. 미저장 epoch는 복원하지 않았다.
- 이전 학습 로그에서 개별 Mask·HCCE Loss는 기록되지 않아 N/A이며, 위 분해 값은 checkpoint 사후 재계산이다.
- 다음 한 변수 제안: PBR GT 유효 Mask 내부 Front/Back XYZ L2 평균을 checkpoint 선택 점수로 사전 지정한다. 같은 Val에 대한 사후 epoch40은 ADD0/20이므로 개선 보장은 없다.
- 실사27·재학습·추가 패키지는 사용하지 않았다.

그림: {link(R/'figures/loss_vs_xyz.png')}, {link(R/'figures/checkpoint_pose_comparison.png')}.  
상세: {link(R/'reports/Checkpoint_선택_원인분석.md')}.
'''
save(R/'reports/세미나_추가진단_요약.md',summary)
for p in (R/'reports').glob('*.md'):
 for q in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(q).is_file(),q
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
js(R/'checks/report_check.json',{'status':'PASS','reports':2,'actual_checkpoints':10,'main_run_checkpoints':9,'epoch0_reference_separate':True,'links_resolved':True,'unrecorded_history_components':'N/A','training_or_real27':False})
print(json.dumps({'status':'PASS','baseline155':{'loss':b155['val_loss_recomputed'],'XYZ':b155['front_xyz_l2']},'rotation10':{'loss':r10['val_loss_recomputed'],'XYZ':r10['front_xyz_l2']},'rotation50':{'loss':r50['val_loss_recomputed'],'XYZ':r50['front_xyz_l2']},'integrity':integrity},ensure_ascii=False,indent=2))
