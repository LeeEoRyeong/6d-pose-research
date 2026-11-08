"""실측 결과로 한국어 최종보고서와 발표용 도표를 생성한다."""
from pathlib import Path
import json,csv,hashlib,re,ast,datetime,subprocess,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];B=R.parent;OLD=B/'runs/baseline_seed42_20261103_214353/trained_only';AG=B/'agent_run_v1';DIAG=B/'checkpoint_diagnostic_v1'
def rd(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def save(p,s):
 with p.open('x') as f:f.write(s)
def js(p,v):save(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def link(p):return f'[{p.name}]({p})'
def n(x):return float(x)
def fmt(x,k=4):return 'N/A' if x in (None,'') else (f'{float(x):.{k}f}' if abs(float(x))<1e6 else f'{float(x):.3e}')
def diff(new,old):
 if new<old:return f'{old-new:.4f} 감소'
 if new>old:return f'{new-old:.4f} 증가'
 return '변화 없음'
def rate(new,old):
 if new>old:return f'{new-old:+d}개 증가'
 if new<old:return f'{new-old:+d}개 감소'
 return '변화 없음'
train=rd(R/'checks/training_summary.json');locked=rd(R/'checks/selection_frozen.json');real=rd(R/'checks/real27_summary.json');validation=rd(R/'checks/pbr_final_reproduction.json');support=rd(R/'checks/support_validation.json');equiv=rd(R/'checks/loss_equivalence.json');maskcheck=rd(R/'checks/mask_gate_validation.json');init=rd(R/'checks/initialization.json');smoke=rd(R/'checks/gpu_smoke.json');config=rd(R/'configs/experiment_config.json');prior=rd(AG/'checks/stage1_gate.json');tab=rows(R/'tables/pbr_validation_metrics.csv');history=rows(R/'tables/train_history.csv');realrows=rows(R/'tables/real27_metrics.csv');pair_analysis=rd(R/'checks/pbr_frame_analysis.json');assert pair_analysis['status']=='PASS';recs=[json.loads(x) for x in (R/'logs/evaluation.jsonl').open()]
assert all(x['status']=='PASS' for x in [train,real,validation,support,equiv,maskcheck,init,smoke]);assert locked['status']=='FROZEN' and locked['real27_used_for_selection'] is False and real['real27_inference_runs']==1
assert len(realrows)==27 and len({x['frame'] for x in realrows})==27 and len(tab)==4
get=lambda m,b:next(x for x in tab if x['model']==m and x['branch']==b)
bf=get('baseline','front');bb=get('baseline','back');nf=get('GT_mask_loss','front');nb=get('GT_mask_loss','back');bestval=train['official']['metrics']['val'];bestxyz=train['secondary']['metrics']['val'];
# Verify seeded batch sequence over all epochs actually completed.
def all_orders(p):
 out={}
 for line in p.open():
  x=json.loads(line);out.setdefault(x['epoch'],[]).extend(x['batch_ids'])
 return out
baseline_orders=all_orders(OLD/'logs/train_steps.jsonl');current_orders=all_orders(R/'logs/train_steps.jsonl');mismatch=[e for e,v in current_orders.items() if v!=baseline_orders.get(e,[])[:len(v)]]
assert not mismatch,mismatch
js(R/'checks/batch_order_verification.json',{'status':'PASS','epochs_compared':len(current_orders),'mismatched_epochs':mismatch,'baseline_reference':str(OLD/'logs/train_steps.jsonl'),'all_compared_batches_equal':True})
# Read-only preservation comparison, including all earlier diagnosis and experimental artifacts.
integrity={}
for name,p,root in [('원본 코드·데이터',B/'checks/original_manifest.json',None),('기존 세미나',B/'checks/artifact_manifest.json',B),('기존 Baseline',OLD.parent/'checks/final_artifact_manifest.json',OLD.parent),('Pose 진단',B/'diagnostics_v1/checks/final_artifact_manifest.json',B/'diagnostics_v1'),('Train/Val 진단',B/'diagnostics_train_val_v1/checks/final_artifact_manifest.json',B/'diagnostics_train_val_v1'),('자율 실험',AG/'checks/final_artifact_manifest.json',AG),('체크포인트 진단',DIAG/'checks/final_artifact_manifest.json',DIAG)]:
 records=rd(p);bad=[]
 for x in records:
  path=Path(x['path']) if root is None else root/x['path']
  if not path.is_file():bad.append(str(path));continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   for c in iter(lambda:f.read(1048576),b''):h.update(c)
  if h.hexdigest()!=x['sha256']:bad.append(str(path))
 integrity[name]={'count':len(records),'mismatches':bad}
assert all(not x['mismatches'] for x in integrity.values()),integrity
integrity['GPU_완료시_compute_조회']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();js(R/'checks/final_integrity.json',integrity)
# Existing Noto CJK fonts make Korean chart text legible in a presentation without an installation.
def font(n,bold=False):
 p=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 try:return ImageFont.truetype(str(p),n,index=1)
 except OSError:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
TITLE=font(42,True);TEXT=font(30);SMALL=font(24);BLUE='#246b9e';RED='#bc4f3b';GREEN='#25805f'
def heading(im,title,sub):
 d=ImageDraw.Draw(im);d.text((60,28),title,font=TITLE,fill='#17202b');d.text((60,92),sub,font=SMALL,fill='#444');return d
def bars(path,title,sub,groups,ytitle,percent=False):
 im=Image.new('RGB',(1800,1050),'white');d=heading(im,title,sub);l,t,r,b=160,180,1730,800;top=max([float(v) for g in groups for v in g['values']]+[1e-4])*1.25
 if percent:top=max(100.,top)
 for k in range(6):
  y=b-(b-t)*k/5;d.line((l,y,r,y),fill='#dddddd',width=2);d.text((30,y-15),f'{top*k/5:.1f}',font=SMALL,fill='black')
 width=(r-l)/len(groups)
 for i,g in enumerate(groups):
  center=l+(i+.5)*width
  for j,(v,col) in enumerate(zip(g['values'],[BLUE,RED])):
   x=center+(j-.5)*150-60;y=b-float(v)/top*(b-t);d.rectangle((x,y,x+120,b),fill=col);label=g.get('labels',[None,None])[j] or f'{float(v):.2f}';d.text((x-5,y-42),label,font=SMALL,fill=col)
  d.text((center-125,b+28),g['name'],font=TEXT,fill='black')
 d.text((l,t-55),ytitle,font=SMALL,fill='black');d.text((300,950),'파랑: 기존 Baseline',font=TEXT,fill=BLUE);d.text((1000,950),'빨강: GT Mask Loss',font=TEXT,fill=RED);im.save(path)
bars(R/'figures/xyz_L2_comparison.png','좌표 오차 비교','PBR Val20 · GT visible BBox Oracle · 프레임 평균',[{'name':'Front XYZ L2','values':[n(bf['XYZ_L2']),n(nf['XYZ_L2'])]},{'name':'Back XYZ L2','values':[n(bb['XYZ_L2']),n(nb['XYZ_L2'])]}],'CAD 원본 수치 단위')
bars(R/'figures/ADD_comparison.png','ADD@0.1d 비교','PnP 실패도 분모에 포함 · 실사27은 개발 세트',[{'name':'PBR Val20','values':[n(bf['ADD_rate'])*100,n(nf['ADD_rate'])*100],'labels':[f"{bf['ADD_pass']}/20",f"{nf['ADD_pass']}/20"]},{'name':'실사27','values':[prior['ADD_rate']*100,real['ADD_rate']*100],'labels':[f"{prior['ADD_pass']}/27",f"{real['ADD_pass']}/27"]}],'ADD 통과율 (%)',True)
bars(R/'figures/Z0_error_comparison.png','Z0 hard-bit 오류 비교','PBR Val20 · GT 유효영역 · 원본 FP16 임계값',[{'name':'Front Z0','values':[n(bf['Z0_error'])*100,n(nf['Z0_error'])*100],'labels':[f"{n(bf['Z0_error'])*100:.2f}%",f"{n(nf['Z0_error'])*100:.2f}%"]},{'name':'Back Z0','values':[n(bb['Z0_error'])*100,n(nb['Z0_error'])*100],'labels':[f"{n(bb['Z0_error'])*100:.2f}%",f"{n(nb['Z0_error'])*100:.2f}%"]}],'오류율 (%)',True)
def lineplot(path,title,sub,series,ylabel):
 im=Image.new('RGB',(1800,1050),'white');d=heading(im,title,sub);l,t,r,b=170,190,1730,820;values=[y for _,_,pts in series for x,y in pts];epochs=[x for _,_,pts in series for x,y in pts];top=max(values)*1.12;xmax=max(epochs) or 1
 for k in range(6):
  y=b-(b-t)*k/5;d.line((l,y,r,y),fill='#ddd',width=2);d.text((20,y-14),f'{top*k/5:.2f}',font=SMALL,fill='black');x=l+(r-l)*k/5;d.text((x-20,b+22),f'{round(xmax*k/5)}',font=SMALL,fill='black')
 for j,(label,col,pts) in enumerate(series):
  pp=[(l+x/xmax*(r-l),b-y/top*(b-t)) for x,y in pts]
  if len(pp)>1:d.line(pp,fill=col,width=5)
  for x,y in pp:d.ellipse((x-4,y-4,x+4,y+4),fill=col)
  d.text((160+j*770,955),label,font=TEXT,fill=col)
 d.text((l,t-52),ylabel,font=SMALL,fill='black');d.text((880,880),'Epoch',font=TEXT,fill='black');im.save(path)
lineplot(R/'figures/train_val_loss.png','GT Mask Loss 학습 곡선','동일한 변경 Loss 정의 내 Train 평가 / Val 평가 비교', [('Train 평가',BLUE,[(x['epoch'],x['train']['total_loss']) for x in recs]),('Val 평가',RED,[(x['epoch'],x['val']['total_loss']) for x in recs])],'Total Loss')
xs=[(int(x['epoch']),float(x['val_front_XYZ_L2'])) for x in history];basevalue=n(bf['XYZ_L2']);lineplot(R/'figures/validation_xyz_comparison.png','검증 Front XYZ L2 변화','기존 Baseline은 epoch155 고정 참고선 · 검증 시점만 연결',[('GT Mask Loss 검증',RED,xs),('Baseline epoch155 고정',BLUE,[(0,basevalue),(max(x[0] for x in xs),basevalue)])],'CAD 원본 수치 단위')
# Fixed frames: input and corresponding GT/predicted overlay, including failures.
priority=[26,51,56,92,126];canvas=Image.new('RGB',(1800,1900),'white');d=ImageDraw.Draw(canvas);d.text((45,20),'실사 개발 세트: GT와 예측 Pose',font=TITLE,fill='black');d.text((45,80),'GT 녹색 / 예측 자홍 · 실패 시 GT만 표시 · 공통 Oracle ROI',font=SMALL,fill='black');byid={int(x['frame']):x for x in realrows}
for j,i in enumerate(priority):
 row=byid[i];h=300;y=140+j*345
 refrows=rd(AG/'stage1/pose_results.json')['rows'];ref=next(v for v in refrows if v['frame']==i);bx,by,bw,bh=json.loads(ref['bbox_padded_xywh']);region=(max(0,int(bx-bw*.15)),max(0,int(by-bh*.15)),min(1280,int(bx+bw*1.15)),min(720,int(by+bh*1.15)))
 for k,src in enumerate([AG/'stage1/controlled_rgb'/f'{i:06d}.png',R/'figures/real27_overlays'/f'{i:06d}.png']):
  im=Image.open(src).convert('RGB').crop(region);im.thumbnail((850,h));canvas.paste(im,(k*900+(900-im.width)//2,y+34+(h-im.height)//2))
 d.text((35,y),f'{i:06d}  원본 보정 영상',font=SMALL,fill='black');d.text((935,y),f"PnP: {'유효' if row['valid_pose']=='True' else '실패'} / ADD: {fmt(row['add'],2)}",font=SMALL,fill='black')
canvas.save(R/'figures/real27_priority_overlay.png')
# Paired statistics apply only when both returned a pose, never fill failure with zero error.
paired=[x for x in realrows if x['valid_pose']=='True' and x['baseline_valid']=='True'];paired_base=float(np.mean([float(x['baseline_add']) for x in paired])) if paired else None;paired_new=float(np.mean([float(x['add']) for x in paired])) if paired else None
mean_field=lambda k:fmt(real.get(k));val_table=[]
for label,a,b,style in [('Mask IoU',bf['mask_iou'],nf['mask_iou'],'number'),('Front XYZ L2',bf['XYZ_L2'],nf['XYZ_L2'],'number'),('Back XYZ L2',bb['XYZ_L2'],nb['XYZ_L2'],'number'),('Front Z0 오류율',bf['Z0_error'],nf['Z0_error'],'percent'),('Back Z0 오류율',bb['Z0_error'],nb['Z0_error'],'percent'),('Front PnP 유효',bf['PnP_valid'],nf['PnP_valid'],'count'),('Front ADD@0.1d',bf['ADD_pass'],nf['ADD_pass'],'count'),('Front 평균 ADD (유효 반환)',bf['mean_ADD_valid'],nf['mean_ADD_valid'],'number'),('Front 평균 RE ° (유효 반환)',bf['mean_RE_valid'],nf['mean_RE_valid'],'number'),('Front 평균 TE (유효 반환)',bf['mean_TE_valid'],nf['mean_TE_valid'],'number')]:
 if style=='percent':sa=f'{n(a)*100:.2f}%';sb=f'{n(b)*100:.2f}%'
 elif style=='count':sa=f'{a}/20';sb=f'{b}/20'
 else:sa=fmt(a);sb=fmt(b)
 val_table.append(f'| {label} | {sa} | {sb} |')
base_sel=155;new_sel=train['official']['epoch'];secondary_ep=train['secondary']['epoch'];
xyz_score0=.5*(n(bf['XYZ_L2'])+n(bb['XYZ_L2']));xyz_score1=.5*(n(nf['XYZ_L2'])+n(nb['XYZ_L2']));
resultXYZ='감소' if xyz_score1<xyz_score0 else '증가' if xyz_score1>xyz_score0 else '변화 없음';resultZ='감소' if n(nf['Z0_error'])<n(bf['Z0_error']) else '증가' if n(nf['Z0_error'])>n(bf['Z0_error']) else '변화 없음';resultADD=rate(int(nf['ADD_pass']),int(bf['ADD_pass']));resultReal=rate(real['ADD_pass'],prior['ADD_pass']);
now=datetime.datetime.now(datetime.timezone.utc);elapsed=(now-datetime.datetime.fromtimestamp(1793717375,datetime.timezone.utc)).total_seconds()/60
identity=equiv['actual_batch_comparison'];
text=f'''# HCCEPose SMT obj04 — GT Mask 좌표 Loss 개선 실험 최종보고서

작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}  
실험 루트: `{R}`  
보고서 작성 시 전체 약 {elapsed:.2f}분 / 최대90분.

**결과: PBR Val에서 Front/Back 평균 XYZ 오차는 {xyz_score0:.4f}→{xyz_score1:.4f}로 {resultXYZ}했다. Front ADD 통과는 {bf['ADD_pass']}/20→{nf['ADD_pass']}/20({resultADD}), 실사 개발27 ADD 통과는 {prior['ADD_pass']}/27→{real['ADD_pass']}/27({resultReal})이다.** 원본 코드는 유지하고 GT 유효 Mask의 좌표 Loss 게이트만 바꾼 단일 학습 결과다.

## 1. 연구 목적과 기존 문제

기존 Baseline은 PBR Train80/Val20의 obj04, random ResNet34, GT BBox Oracle crop으로 학습했다. Val Front/Back XYZ L2는 {n(bf['XYZ_L2']):.4f}/{n(bb['XYZ_L2']):.4f}, Front Z0 오류는 {n(bf['Z0_error'])*100:.2f}%, Front ADD는 {bf['ADD_pass']}/20이었다. 기존 실사 개발27 ADD는 {prior['ADD_pass']}/27이다.

이전 분석에서 예측 Mask를 좌표 Loss의 게이트로 쓰면 놓친 GT 객체 픽셀의 코드 오차가 빠지는 사실을 확인했다. GT Mask를 이용한 고정 가중치 반사실에서 일부 checkpoint의 Loss 순서가 달랐지만, 이는 재학습 효과의 증거가 아니었다. 이번에는 **좌표 L1의 공간 게이트만** 변경한 실제 단일 학습으로 가설을 시험했다.

## 2. 기존 Loss와 변경 수식

기존 좌표 L1은 `round(sigmoid(pred_mask)).detach()`를 Front/Back 코드와 target에 곱한 후 batch·height·width 전체 평균을 취한다. GT 객체이지만 예측 Mask가 놓친 픽셀은 좌표 항에 기여하지 않는다. Mask Loss 자체는 `L1(sigmoid(pred_mask), gt_mask)`이다.

이번 변경은 branch별 `G_front = GT_visible_mask ∩ valid_front`, `G_back = GT_visible_mask ∩ valid_back`을 Front/Back L1의 공간 게이트로 사용한다. Label validity는 실제 렌더의 유한 near/far depth support를 쓰며 단순히 XYZ 값이 0이 아니라고 가정하지 않는다.

```text
L_front_axis = weighted_mean_planes(mean_BHW(abs(pred_code * G_front
                                                  - (2 * GT_code - 1) * G_front)))
L_back_axis  = 같은 식에서 G_back 사용
L_total = L_mask + 3 * sum(L_front_axis) + 3 * sum(L_back_axis)
```

plane별 동적 weight와 길이100 history는 **이전과 동일하게 predicted Mask** 기반이다. Mask Loss·HCCE encoding/복호화·channel order·target 변환·3배 계수·batch/crop 전체 평균·AMP는 유지했다. 유효 픽셀 수로 나누는 새 정규화는 추가하지 않았다. 유효 픽셀 수 변화는독립변수의 일부다.

근거: 원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`, 기존 호환 복사본 `{OLD/'code/network_model_explicit.py'}`, 새 복사본 {link(R/'code/network_gtmask.py')}, 정확한 차이 {link(R/'checks/loss_change.diff')}.

## 3. 유효 support·학습 전 검증

기존 CPU near/far CAD renderer에서 100프레임의 finite-depth support를 재구성하고, 원본 Front/Back 양자화 PNG 전체와 각 픽셀을 다시 계산해 **100/100 정확히 일치**시켰다. 이 데이터에서는 유효한 양자화 전체0 픽셀 Front/Back 각각 {support['full_valid_allzero_quant_front']}/{support['full_valid_allzero_quant_back']}개, 두 branch support 차이 {support['front_back_support_difference_total']}픽셀이었다. 별도 branch support 이미지를 저장하고 GT visible Mask와 각각 교차했다. 원본 VisPy 구현과의 bitwise 비교는 이전과 마찬가지로 미완료다.

| 확인 | 실제 결과 |
|---|---|
| 기존 epoch155 Val Total Loss | 기록 {equiv['baseline_checkpoint_val_reproduction']['logged_val_loss']:.9f}, 재계산 {equiv['baseline_checkpoint_val_reproduction']['recomputed_val_loss']:.9f}, 차이 {equiv['baseline_checkpoint_val_reproduction']['abs_diff']} |
| 동일 Mask 인공 배치 | CPU/GPU 기존·새 Loss와 gradient 일치; 상세 {link(R/'checks/loss_equivalence.json')} |
| 예측 Mask가 GT 전경 전체 누락 | 기존 좌표 항0, 변경 항은 Front/Back 양수; GT 유효영역에 gradient 전달 |
| 배경·무효 support | 각 branch의 게이트 밖 좌표 gradient 0; branch별로 다른 support 인공 사례 통과 |
| Mask Loss·dynamic weight/history | Mask Loss 동일; 동일 predicted Mask에서 gate가 달라도 dynamic weight와 history 일치 |
| GPU smoke | optimizer 적용0회, Front/Back gradient 유한·0 아님, 독립 모델 폐기 |
| 초기화·배치 순서 | random seed42; 모델 state 전부 epoch0 참조와 동일. 실제 학습 {len(current_orders)}개 epoch의 기록된 batch 순서도 Baseline과 일치 |

실제 Val 첫4장 동일 출력에서 기존/변경 total은 {identity['old_total']:.6f}/{identity['new_total']:.6f}, Mask 항은 둘 다 {identity['old_mask']:.6f}였다. 당시 predicted gate {identity['pred_gate_pixels']}픽셀, GT 유효 Front/Back gate {identity['GT_front_gate_pixels']}/{identity['GT_back_gate_pixels']}픽셀이었다. 두 Loss는 정의가 다르므로 이 숫자의 크기 차이를 성능 개선으로 해석하지 않는다.

상세: {link(R/'checks/support_validation.json')}, {link(R/'checks/mask_gate_validation.json')}, {link(R/'checks/initialization.json')}, {link(R/'checks/gpu_smoke.json')}, {link(R/'checks/batch_order_verification.json')}.

## 4. 학습 조건과 checkpoint 선택

| 항목 | 실제 설정·결과 |
|---|---|
| 데이터 | 기존 Train80 / Val20, 생성 block {config['train_blocks']} / {config['val_blocks']}, 실사0 |
| 모델·초기화 | ResNet34 random, seed42, pretrained 없음, 초기 state 동일 |
| 입력·optimizer | BGR 정규화, crop256/label128, batch4, Adam lr0.0002, AMP float16 |
| 증강·PnP | 증강 없음, 원본 Front/Back RANSAC EPNP 1.5px·150회·seed42+frame |
| 상한·중단 | 최대195epoch/3900 optimizer step, 동일 ValLoss patience8·min_delta0.0001 |
| 실제 학습 | **1회**, {train['epoch_completed']}epoch, optimizer {train['optimizer_steps']}회 / 시도{train['optimizer_attempts']}회, AMP skip {train['amp_skipped_steps']}회, 종료 `{train['stop_reason']}` |
| 자원·시간 | 학습 wall {train['wall_seconds']:.2f}초, peak allocated {train['peak_allocated_bytes']/2**20:.2f}MiB. 평가·저장 포함 wall 시간 |
| 공식 checkpoint | 변경 Loss의 PBR Val total Loss 기준 epoch **{new_sel}**; {link(R/'checkpoints/best_val_loss.pt')} |
| 보조 checkpoint | PBR Val GT 유효영역 `0.5*(Front XYZ L2+Back XYZ L2)` 기준 epoch **{secondary_ep}**; {link(R/'checkpoints/best_val_xyz.pt')} |

두 기준은 {'같은 epoch를 선택했다.' if train['same_selected_epoch'] else '서로 다른 epoch를 선택했다. 공식 결과는 Val Loss checkpoint이며 XYZ checkpoint는 PBR 보조 분석으로만 보존했다.'} 공식 선택은 실사 데이터를 읽기 전에 {link(R/'checks/selection_frozen.json')}에 epoch·SHA-256을 고정했다. 실사 결과를 이용한 재선택·추가 학습은 없다.

각 평가 시점의 Train/Val total·Mask·Front/Back loss, GT 전경 누락률, Front/Back XYZ·축별 MAE·Z0, Mask IoU, PnP·ADD는 {link(R/'tables/train_history.csv')}와 `logs/evaluation.jsonl`에 있다. Loss history는 평가마다 새로 만들며, 교육용과 validation history를 섞지 않았다. {link(R/'figures/train_val_loss.png')}.

## 5. PBR Validation — 실제 예측 비교

| 지표 | 기존 Baseline epoch155 | GT Mask Loss 공식 epoch{new_sel} |
|---|---:|---:|
{chr(10).join(val_table)}

XYZ는 GT Mask∩실제 branch valid support 내부의 동일 128×128 픽셀에서 계산한 모델 좌표 L2다. Mask IoU는 crop 전체에서 측정했다. 정확도는 실제 네트워크가 예측한 HCCE·Mask와 원본 PnP 반환으로 계산했으며, 이전 GT XYZ Oracle 결과를 모델 성능에 섞지 않았다. 실패한 PnP는 ADD 실패로 포함해 분모를20으로 유지했다.

Front Z0 오류율은 {n(bf['Z0_error'])*100:.2f}%→{n(nf['Z0_error'])*100:.2f}%로 {resultZ}했다. 같은20프레임에서 Front XYZ 오차가 줄어든 프레임은 {pair_analysis['XYZ_better_frames']}개, 늘어난 프레임은 {pair_analysis['XYZ_worse_frames']}개였다. ADD 통과 프레임은 기존 {pair_analysis['baseline_ADD_pass_frames']}에서 새 모델 {pair_analysis['new_ADD_pass_frames']}로 바뀌었고, 평균 Front PnP inlier 비율은 {pair_analysis['baseline_mean_inlier_ratio']*100:.3f}%→{pair_analysis['new_mean_inlier_ratio']*100:.3f}%였다. 평균 XYZ와 threshold 통과 수가 서로 다른 방향으로 바뀐 관찰이며, inlier 변화만이 원인이라는 뜻은 아니다. Front와 Back의 x/y/z MAE 및 유효 PnP만의 ADD·RE·TE는 {link(R/'tables/pbr_validation_metrics.csv')}에 있고 동일 프레임별 비교는 {link(R/'tables/pbr_validation_per_frame.csv')}에 있다. 유효 PnP 수가 다르면 조건부 평균의 분모가 다르며, 유효 Pose가 없을 때 평균은 N/A다. 위 길이는 CAD 원본 수치 단위이고 실제 mm는 독립 계측되지 않았다.

보조 XYZ checkpoint epoch{secondary_ep}의 PBR Val Front/Back XYZ는 {bestxyz['front_XYZ_L2']:.4f}/{bestxyz['back_XYZ_L2']:.4f}, Front PnP {bestxyz['front_PnP_valid']}/20, ADD {bestxyz['front_ADD_pass']}/20으로 학습 중 보조 기록됐다. {'공식 checkpoint와 동일하다.' if train['same_selected_epoch'] else '공식 Loss checkpoint와 다르며, 보조 checkpoint를 실사에 적용하거나 사후 교체하지 않았다.'}

그림: {link(R/'figures/xyz_L2_comparison.png')}, {link(R/'figures/Z0_error_comparison.png')}, {link(R/'figures/ADD_comparison.png')}, {link(R/'figures/validation_xyz_comparison.png')}.

## 6. 실사 SMT 27프레임 — 공식 checkpoint 1회

기존 `agent_run_v1/stage1`의 동일한 K 명시 controlled-undistorted RGB, GT CAD 투영 Oracle ROI, GT Pose와 ADD threshold를 그대로 재사용했다. 전체27프레임에 공식 checkpoint를 **한 번만** 적용했고 누락·PnP 실패도 ADD 실패로 분모에 포함했다.

| 지표 | 기존 Baseline | GT Mask Loss |
|---|---:|---:|
| 유효 PnP | {prior['PnP_valid']}/27 | {real['PnP_valid']}/27 |
| ADD@0.1d | {prior['ADD_pass']}/27 | {real['ADD_pass']}/27 |
| 평균 inlier 비율, 전체27 | {prior['inlier_ratio_mean_all_frames']*100:.4f}% | {real['mean_inlier_ratio_all27']*100:.4f}% |
| 평균 ADD, 유효 Pose만 | {prior['valid_pose_error_stats']['add']['mean']:.4f} (n={prior['PnP_valid']}) | {mean_field('mean_ADD_valid')} (n={real['PnP_valid']}) |
| 중앙 ADD, 유효 Pose만 | {prior['valid_pose_error_stats']['add']['median']:.4f} | {mean_field('median_ADD_valid')} |
| 평균 RE, 유효 Pose만 | {prior['valid_pose_error_stats']['rotation_deg']['mean']:.4f}° | {mean_field('mean_RE_valid')}° |
| 중앙 RE, 유효 Pose만 | {prior['valid_pose_error_stats']['rotation_deg']['median']:.4f}° | {mean_field('median_RE_valid')}° |
| 평균 TE, 유효 Pose만 | {prior['valid_pose_error_stats']['translation']['mean']:.4f} | {mean_field('mean_TE_valid')} |
| 중앙 TE, 유효 Pose만 | {prior['valid_pose_error_stats']['translation']['median']:.4f} | {mean_field('median_TE_valid')} |

조건부 평균은 유효 수가 다르면 직접 비교할 수 없다. 양 모델 모두 유효했던 동일 {len(paired)}개 프레임의 평균 ADD는 기존 {fmt(paired_base)} / 새 모델 {fmt(paired_new)}이다. {'전체27프레임 모두 유효했다.' if len(paired)==27 else '유효 반환에 한정한 선택된 하위집합의 기술값이다.'} 실사 Pose GT와 카메라의 물리 단위·정밀도는 완전히 검증되지 않았고, 5개 프레임은 이전 거시 검토에서 판단 불가였다. 제외 없이 보고했으며 이27장은 독립 최종 테스트가 아니다. 실제 visible Mask가 없어 실사 Mask IoU는 계산하지 않았다.

전체 결과와 실패 사유: {link(R/'tables/real27_metrics.csv')}; 우선5장 비교: {link(R/'figures/real27_priority_overlay.png')}; 개별 overlay 27장 `{R/'figures/real27_overlays'}`. GT는 녹색, 유효 PnP 예측은 자홍색이며 실패 시 GT만 보인다. CAD wireframe은 실제 도구 가림을 복원하지 않는다.

## 7. 질문별 해석과 한계

1. **PBR XYZ:** Front/Back 평균 L2 {xyz_score0:.4f}→{xyz_score1:.4f}, {resultXYZ}. Front는 {n(bf['XYZ_L2']):.4f}→{n(nf['XYZ_L2']):.4f}, Back은 {n(bb['XYZ_L2']):.4f}→{n(nb['XYZ_L2']):.4f}다.
2. **Z0:** Front 오류는 {n(bf['Z0_error'])*100:.2f}%→{n(nf['Z0_error'])*100:.2f}%로 {resultZ}; Back은 {n(bb['Z0_error'])*100:.2f}%→{n(nb['Z0_error'])*100:.2f}%다. 한 plane의 개선을 전체 Pose 개선으로 간주하지 않는다.
3. **XYZ와 ADD:** Front ADD {bf['ADD_pass']}/20→{nf['ADD_pass']}/20({resultADD}). XYZ와 PnP·ADD가 다른 방향이면 correspondence 분포 및 PnP 민감도의 가능성을 따로 검증해야 하며, 이번 데이터만으로 원인을 확정하지 않는다.
4. **게이트의 기여:** 초기 상태·순서·주요 설정을 고정하고 공간 gate만 변경한 단일 Run의 관찰이다. 공식 checkpoint epoch는 Baseline155 대 새 모델{new_sel}, AMP skip·early stopping 경로도 실측대로 달라 단일 seed의 최종 차이를 특정 원인에 대한 보편적인 인과 효과로 확정할 수 없다. 기존/새 total Loss의 절댓값은 정의가 달라 직접 비교하지 않는다.
5. **실사:** 전체27 ADD {prior['ADD_pass']}→{real['ADD_pass']}({resultReal}), 평균 ADD {prior['valid_pose_error_stats']['add']['mean']:.4f}→{mean_field('mean_ADD_valid')}(유효 n={prior['PnP_valid']}/{real['PnP_valid']})이며 동일 Oracle ROI 조건이다. 작은 개발 세트에서 평균의 수치 차이를 독립적인 정확도 개선으로 확정하지 않는다. 실사는 checkpoint 선택 전에 보지 않았으며 이후 재학습·재선택·조정하지 않았다.

PBR Val 2개 생성 블록과 연속적인 실사27은 작은 개발 자료다. GT 기반 ROI, 제한적인 가림 다양성, 정밀 GT·물리 단위·대칭·실사 visible Mask·검출기 미검증, 단일 seed가 한계다. 이 실험의 성능을 독립 최종 테스트나 공정 허용오차 수준으로 주장할 수 없다.

## 8. 후속 연구 — 제안만, 추가 학습 없음

이번 작업의 추가 개선 학습은 종료한다. 이후 독립 촬영 세션의 정확도 검증, 실측 CAD 단위·실사 카메라/GT 정밀도 확인, 실사 visible Mask 및 detector 검증, 사전에 고정된 독립 테스트와 반복 seed를 권장한다. source PnP Back/Mixed inlier 집계 문제는 별도 회귀검증 대상으로 남는다. 어떤 개선도 아직 확인된 사실로 가정하지 않는다.

## 9. 결과물과 보존

- 실험 설정: {link(R/'configs/experiment_config.json')}.
- 사전 검증: {link(R/'checks/loss_equivalence.json')}, {link(R/'checks/mask_gate_validation.json')}, {link(R/'checks/initialization.json')}.
- 학습: {link(R/'tables/train_history.csv')}, {link(R/'checks/training_summary.json')}, 공식/보조 checkpoint는 4절 참조.
- 평가: {link(R/'tables/pbr_validation_metrics.csv')}, {link(R/'tables/real27_metrics.csv')}, {link(R/'checks/real27_summary.json')}.
- 전체 그림: `{R/'figures'}`. 모든 숫자는 저장된 실제 Run의 결과다.
- 원본 및 이전 실험 manifest 재검증: {link(R/'checks/final_integrity.json')}. 모두 일치. 추가 설치·다운로드 없음. 종료 GPU compute 조회: `{integrity['GPU_완료시_compute_조회'] or '실행 중인 compute process 없음'}`.
'''
save(R/'reports/GT_Mask_Loss_개선실험_최종보고서.md',text)
summary=f'''# 세미나 최종 실험 — GT Mask 좌표 Loss

SMT obj04 RGB 6D Pose에서 기존 HCCEPose의 PBR Validation과 실사 일반화가 낮았다. GT XYZ Oracle은 PBR Val20에서20/20이지만 기존 모델은 Front ADD1/20, 실사 ADD0/27이었다. 예측 Mask의 전경 누락이 좌표 Loss를 낮출 수 있어 이번 실험에서는 좌표 L1 gate만 GT visible Mask∩실제 XYZ valid support로 바꿨다.

동적 plane weight·history는 predicted Mask 기반으로 유지했다. 같은 초기 state·Train80/Val20·seed42·optimizer·증강 없음 조건의 단일 학습이며, CPU/GPU 게이트·gradient 검증을 통과했다.

| 지표 | 기존 Baseline | GT Mask Loss 공식 checkpoint |
|---|---:|---:|
| PBR Mask IoU | {n(bf['mask_iou']):.4f} | {n(nf['mask_iou']):.4f} |
| PBR Front / Back XYZ L2 | {n(bf['XYZ_L2']):.2f} / {n(bb['XYZ_L2']):.2f} | {n(nf['XYZ_L2']):.2f} / {n(nb['XYZ_L2']):.2f} |
| PBR Front Z0 오류 | {n(bf['Z0_error'])*100:.2f}% | {n(nf['Z0_error'])*100:.2f}% |
| PBR Front PnP / ADD | {bf['PnP_valid']}/20 / {bf['ADD_pass']}/20 | {nf['PnP_valid']}/20 / {nf['ADD_pass']}/20 |
| 실사27 PnP / ADD | {prior['PnP_valid']}/27 / {prior['ADD_pass']}/27 | {real['PnP_valid']}/27 / {real['ADD_pass']}/27 |

**최종 결론**

1. 공식 Val Loss checkpoint에서 PBR Front/Back 평균 XYZ는 {xyz_score0:.2f}→{xyz_score1:.2f}로 {resultXYZ}했다.
2. Front PBR ADD는 {bf['ADD_pass']}/20→{nf['ADD_pass']}/20, 실사 개발27 ADD는 {prior['ADD_pass']}/27→{real['ADD_pass']}/27이었다.
3. 단일 seed와 작은 PBR Val/실사 개발 세트 결과이며 GT Oracle ROI·GT 정밀도 제한이 있다. 추가 개선 학습은 수행하지 않았다.

발표용 그림: {link(R/'figures/xyz_L2_comparison.png')}, {link(R/'figures/ADD_comparison.png')}, {link(R/'figures/Z0_error_comparison.png')}, {link(R/'figures/train_val_loss.png')}, {link(R/'figures/validation_xyz_comparison.png')}, {link(R/'figures/real27_priority_overlay.png')}.

상세: {link(R/'reports/GT_Mask_Loss_개선실험_최종보고서.md')}
'''
save(R/'reports/세미나_최종실험_요약.md',summary)
for p in (R/'reports').glob('*.md'):
 for q in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(q).is_file(),q
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
js(R/'checks/report_validation.json',{'status':'PASS','reports':2,'PBR_rows':4,'real_rows':27,'training_runs':1,'real_inference_runs':1,'official_epoch':new_sel,'secondary_epoch':secondary_ep,'links_resolve':True,'model_selection_frozen_before_real':True,'elapsed_minutes':elapsed})
print(json.dumps({'status':'PASS','official_epoch':new_sel,'secondary_epoch':secondary_ep,'PBR_comparison':tab,'real27':real,'elapsed_minutes':elapsed,'integrity':integrity},ensure_ascii=False,indent=2))
