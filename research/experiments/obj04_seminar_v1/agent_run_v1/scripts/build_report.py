"""단계별 실측과 실패를 바탕으로 한국어 연구 보고서와 비교 그림 생성."""
from pathlib import Path
import csv,json,datetime,hashlib,re,subprocess,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];B=R.parent;ORIG=Path('/RESEARCH_DATA/HCCEPose');OLD=B/'runs/baseline_seed42_20261103_214353/trained_only';PREV=B/'diagnostics_v1';TV=B/'diagnostics_train_val_v1'
def rd(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def save(p,s):
 with p.open('x') as f:f.write(s)
def jsonout(p,x):save(p,json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def link(p):return f'[{p.name}]({p})'
def fmt(x,precision=4):return '판단 불가' if x is None or x=='' else f'{float(x):.{precision}f}'
stage1=rd(R/'checks/stage1_gate.json');pre=rd(R/'checks/stage1_preflight.json');s2=rd(R/'stage2/analysis_summary.json');repro=rd(R/'checks/pbr_repro_gate.json');align=rd(R/'checks/augmentation_alignment.json');smoke=rd(R/'checks/augmentation_cpu_smoke.json');init=rd(R/'checks/rotation_initialization.json');train=rd(R/'checks/rotation_training.json');s3=rd(R/'checks/stage3_gate.json');real=rd(R/'stage3/real_comparison.json');pbr=rows(R/'stage3/pbr_comparison.csv');geometry=rd(R/'stage3/pnp_correspondence_geometry.json');oldp=rd(OLD/'checks/training_summary.json');checks=[stage1,s2,repro,align,smoke,init,train,s3];assert all(x['status']=='PASS' for x in checks)
def row(model,split,branch):return next(x for x in pbr if x['model']==model and x['split']==split and x['branch']==branch)
bf=row('baseline','val','front');af=row('rotation','val','front');bb=row('baseline','val','back');ab=row('rotation','val','back');rf=rows(R/'stage3/real_comparison_per_frame.csv');paired=[x for x in rf if x['valid_pose']=='True'];paired_base=float(np.mean([float(x['baseline_add']) for x in paired])) if paired else None;paired_new=float(np.mean([float(x['add']) for x in paired])) if paired else None
# Existing manifests verify original and previous research assets (all files are read only).
integrity={}
for name,p,root in [('원본 코드·PBR·실사',B/'checks/original_manifest.json',None),('이전 세미나',B/'checks/artifact_manifest.json',B),('기존 Baseline',OLD.parent/'checks/final_artifact_manifest.json',OLD.parent),('기존 Pose 진단',PREV/'checks/final_artifact_manifest.json',PREV),('기존 Train/Val 진단',TV/'checks/final_artifact_manifest.json',TV)]:
 data=rd(p);bad=[]
 for x in data:
  path=Path(x['path']) if root is None else root/x['path']
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=x['sha256']:bad.append(str(path))
 integrity[name]={'checked':len(data),'mismatches':bad}
assert all(not v['mismatches'] for v in integrity.values()),integrity
integrity['종료 시 GPU compute 조회']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
jsonout(R/'checks/final_integrity.json',integrity)
# PNG charts use actual values, distinct scales and truthful denominators.
def font(n):
 try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
 except OSError:return ImageFont.load_default()
FT=font(22);FM=font(16);FS=font(13);BLUE='#3075a7';RED='#c65242'
def barpair(draw,box,title,values,top,unit):
 l,t,r,b=box;draw.text((l,t-35),title,font=FM,fill='black')
 for j in range(5):
  y=b-(b-t)*j/4;draw.line((l,y,r,y),fill='#dddddd');draw.text((l-52,y-8),f'{top*j/4:.1f}',font=FS,fill='black')
 for i,(name,value,col) in enumerate(values):
  x=l+80+i*170;y=b-(b-t)*min(value/top,1);draw.rectangle((x,y,x+100,b),fill=col);draw.text((x,y-22),f'{value:.3f}{unit}',font=FS,fill=col);draw.text((x-5,b+12),name,font=FS,fill='black')
im=Image.new('RGB',(1400,860),'white');d=ImageDraw.Draw(im);d.text((40,20),'PBR Val20: baseline vs synchronized crop rotation',font=FT,fill='black');d.text((40,55),'Baseline epoch155 / rotation epoch10; same selection rule, different stop epochs',font=FM,fill='black')
barpair(d,(110,135,640,345),'Mask IoU', [('Baseline',float(bf['mask_iou']),BLUE),('Rotation',float(af['mask_iou']),RED)],1,'')
barpair(d,(820,135,1350,345),'Front XYZ mean L2 (native units)', [('Baseline',float(bf['xyz_l2']),BLUE),('Rotation',float(af['xyz_l2']),RED)],70,'')
barpair(d,(110,470,640,680),'Front Z0 hard bit error', [('Baseline',float(bf['Z0_hard_error'])*100,BLUE),('Rotation',float(af['Z0_hard_error'])*100,RED)],60,'%')
barpair(d,(820,470,1350,680),'Front ADD@0.1d (all 20)', [('Baseline',float(bf['ADD_rate'])*100,BLUE),('Rotation',float(af['ADD_rate'])*100,RED)],100,'%')
d.text((75,790),'Oracle ROI; Train/Val blocks correlated; not an independent test.',font=FM,fill='black');im.save(R/'stage3/pbr_comparison.png')
im=Image.new('RGB',(1200,620),'white');d=ImageDraw.Draw(im);d.text((40,20),'Real development set (27 frames): baseline vs rotation',font=FT,fill='black');d.text((40,58),'Same controlled undistortion and CAD-derived oracle ROI; checkpoints fixed using PBR Val loss',font=FM,fill='black')
barpair(d,(100,155,550,460),'Valid PnP returned (out of 27)', [('Baseline',stage1['PnP_valid'],BLUE),('Rotation',real['PnP_valid'],RED)],27,'')
barpair(d,(700,155,1150,460),'ADD@0.1d passes (out of 27)', [('Baseline',stage1['ADD_pass'],BLUE),('Rotation',real['ADD_pass'],RED)],27,'')
d.text((70,550),'GT quality: coarse visual verification; five frames inconclusive; absolute physical unit unverified.',font=FS,fill='black');im.save(R/'stage3/real_comparison.png')
# Validation-loss curves from saved logs, excluding untrained epoch0 for selection clarity.
e0=[json.loads(x) for x in (OLD/'logs/epoch_metrics.jsonl').read_text().splitlines()];e1=[json.loads(x) for x in (R/'logs/rotation_epochs.jsonl').read_text().splitlines()];im=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(im);d.text((45,18),'PBR Validation loss during training',font=FT,fill='black');d.text((45,50),'Fresh Loss history per evaluation, epoch0 excluded; lower loss may accompany poor coordinates',font=FM,fill='black');l,t,r,b=120,105,1230,580;xmax=195;ymax=max(e['val']['loss'] for e in e0[1:]+e1[1:])*1.1
for i in range(6):
 y=b-i/5*(b-t);d.line((l,y,r,y),fill='#ddd');d.text((45,y-8),f'{ymax*i/5:.2f}',font=FS,fill='black');x=l+i/5*(r-l);d.text((x-8,b+12),str(round(i*xmax/5)),font=FS,fill='black')
for data,name,col in [(e0,'Baseline',BLUE),(e1,'Rotation',RED)]:
 pts=[(l+e['epoch']/xmax*(r-l),b-e['val']['loss']/ymax*(b-t)) for e in data if e['epoch']>0]
 if len(pts)>1:d.line(pts,fill=col,width=3)
 for x,y in pts:d.ellipse((x-3,y-3,x+3,y+3),fill=col)
d.text((220,650),'Baseline (selected epoch155)',font=FM,fill=BLUE);d.text((700,650),'Rotation (selected epoch10)',font=FM,fill=RED);im.save(R/'stage3/validation_loss_curve.png')
# Frame-specific additive analysis, using independently recorded human occlusion annotations.
rr=rows(R/'stage2/real_frame_analysis.csv');rt=[]
for x in rr:
 rt.append(f"| {int(x['frame']):06d} | {x['occlusion_prior']} | {x['GT_quality_prior']} | {'성공' if x['valid_pose']=='True' else '실패'} | {fmt(x['add'],2)} | {fmt(float(x['inlier_ratio_actual'])*100,2)}% | {'통과' if x['add_pass']=='True' else '미통과'} |")
now=datetime.datetime.now(datetime.timezone.utc);elapsed=(now-datetime.datetime.fromtimestamp(1793714609,datetime.timezone.utc)).total_seconds()/60
stage1_text=f'''# Stage 1 — 실사27 Baseline 평가

- 결과: 27프레임 매핑·K/GT/CAD 선행조건 통과. 모델은 epoch155, CAD 기반 GT Oracle ROI.
- 데이터: 원본 RGB를 K,D로 보정해 출력 K=원본 K인 PNG를 신규 생성. 기존 UND 생성 이력은 확정하지 않았다. 두 보정 영상의 평균 픽셀 MAE {rd(R/'checks/camera_check.json')['remap_mae_mean']:.4f} (8-bit)이며, 이는 픽셀 수준 재현 비교이지 캘리브레이션 정밀도 자체가 아니다.
- 결과: 유효 PnP {stage1['PnP_valid']}/27, ADD@0.1d {stage1['ADD_pass']}/27, 평균 ADD {stage1['valid_pose_error_stats']['add']['mean']:.4f}, RE {stage1['valid_pose_error_stats']['rotation_deg']['mean']:.4f}°, TE {stage1['valid_pose_error_stats']['translation']['mean']:.4f}(PBR/CAD 수치 단위). 평균 inlier 비율 {stage1['inlier_ratio_mean_all_frames']*100:.4f}%.
- 전체27 포함, 066/068/070/072/073은 기존 시각 검토의 판단 불가 표식을 유지. 실사 GT는 조건부 개발 평가로서 정밀 mm·deg 정확도와 독립 최종 테스트 지위를 보장하지 않는다.

프레임별: {link(S/'real_pose_per_frame.csv') if (S:=R/'stage1') else ''}. 카메라·GT 체크: {link(R/'checks/stage1_preflight.json')}. 오버레이: `{R/'stage1/overlays'}`.
'''
save(R/'stage1/실사_Baseline_평가.md',stage1_text)
stage2_text=f'''# Stage 2 — 실사 오류 분석

실사 PnP 반환은 {stage1['PnP_valid']}/27이지만 ADD 통과는 {stage1['ADD_pass']}/27이다. 예측 대응점 inlier 비율 평균은 {stage1['inlier_ratio_mean_all_frames']*100:.4f}%이다. PBR Val20은 ADD 1/20, inlier 평균 4.0090%였다. PBR은 GT visible BBox, 실사는 GT CAD amodal BBox를 사용했으므로 두 차이를 외관 도메인 차이만의 인과 효과로 해석하지 않는다.

이전 육안 주석으로 도구 가림이 관찰된17장과 가림이 없거나 적은8장을 구분했을 때 두 집합 모두 ADD 통과는 0이다. 나머지2장은 미분류다. 정량 visible Mask나 가시율 GT는 없으며, 표식은 원인 판정을 위한 정성 자료다.

000051·000092의 GT/예측 overlay를 직접 확인했다. 방향·크기 차이가 보이지만 wireframe은 실물 가림을 모델링하지 않는다. 낮은 좌표 일반화, 외관/가림, ROI 방식, GT 정밀도는 원인 후보이며 각 효과를 분리 측정하지 않았다.

{link(R/'stage2/real_frame_analysis.csv')} · {link(R/'stage2/priority_examples.png')} · {link(R/'stage2/analysis_summary.json')}
'''
save(R/'stage2/실사_오류분석.md',stage2_text)
comparison_table='\n'.join(f"| {label} | {a} | {b} |" for label,a,b in [
('Mask IoU',fmt(bf['mask_iou'],6),fmt(af['mask_iou'],6)),('Front XYZ L2',fmt(bf['xyz_l2']),fmt(af['xyz_l2'])),('Back XYZ L2',fmt(bb['xyz_l2']),fmt(ab['xyz_l2'])),('Front Z0 hard bit 오류',fmt(float(bf['Z0_hard_error'])*100,2)+'%',fmt(float(af['Z0_hard_error'])*100,2)+'%'),('Back Z0 hard bit 오류',fmt(float(bb['Z0_hard_error'])*100,2)+'%',fmt(float(ab['Z0_hard_error'])*100,2)+'%'),('Front 유효 PnP',bf['PnP_valid']+'/20',af['PnP_valid']+'/20'),('Front ADD@0.1d',bf['ADD_pass']+'/20',af['ADD_pass']+'/20'),('Front RE 평균(유효 반환)',fmt(bf['rotation_deg_mean_valid'])+'°',fmt(af['rotation_deg_mean_valid'])+('°' if af['rotation_deg_mean_valid'] else '')),('Front TE 평균(유효 반환)',fmt(bf['translation_mean_valid']),fmt(af['translation_mean_valid']))])
base_geom=next(x for x in geometry if x['model']=='baseline');aug_geom=next(x for x in geometry if x['model']=='rotation')
stage3_text=f'''# Stage 3 — 동기 회전 증강 단일 실험

변경: Train crop에 `uniform[-20°,20°]` in-plane 회전만 추가했다. RGB bilinear, Mask·Front/Back XYZ·유효 support nearest, 변환 뒤 HCCE 재생성이다. 기존 Loss는 유효 support를 별도 가중하지 않으므로 이번에도 support는 검증용으로 추적하고 Loss 수식은 바꾸지 않았다.

- CPU 정합성: Train80의0° 입력/라벨 최대차이 {align['identity_max_tensor_abs_diff']}; 128/256 affine 차이 {align['shared_affine_128_256_max_difference']}; HCCE 복호화 차이 {align['GT_HCCE_roundtrip_max_uint8_error']}. 독립 CPU 모델 smoke 1 optimizer step 및 두 번째 Loss forward 통과.
- 초기 무작위 모델의 모든 state tensor가 Baseline epoch0과 일치했다. Train80/Val20, seed42, batch4, Adam lr0.0002, AMP, Loss, ValLoss 기반 checkpoint 및 early stopping 규칙을 고정했다. 별도 RNG의 회전 각도는 step 로그에 기록했다.
- 단 1회 GPU 학습: {train['completed_epochs']}epoch, 성공 optimizer {train['successful_optimizer_steps']}회, AMP skip {train['amp_skipped_steps']}회, 종료 `{train['stop_reason']}`. 주 checkpoint는 PBR ValLoss 기준 epoch{train['best_val_loss']['epoch']}, Baseline은 epoch155다.
- 같은 선택/early stopping 규칙을 사용했지만 **실제 학습 길이와 선택 epoch가 달라** 회전 자체의 순수 효과를 분리해 확정할 수 없다. 추가 checkpoint 선택이나 재학습은 하지 않았다.

## PBR Val20 비교

| 지표 | Baseline | 회전 증강 |
|---|---:|---:|
{comparison_table}

평균 RE/TE의 '판단 불가'는 0이 아니라 유효 PnP가 0개이기 때문이다. 기준 Mask·XYZ·ADD는 악화됐고 Front Z0 bit 오류만 감소했다. 이는 단일 Z plane 개선이 전체 6D Pose 개선을 보장하지 않는 사례다.

Val에서 예측 Front XYZ의 프레임당 unique 점 평균은 {base_geom['unique_xyz_mean']:.1f}→{aug_geom['unique_xyz_mean']:.1f}, X좌표 표준편차 평균은 {base_geom['x_std_mean']:.4f}→{aug_geom['x_std_mean']:.4f}이다. 예측 대응점의 X축 붕괴 및 낮은 다양성이 PnP 실패와 양립한다. 이것만으로 회전 증강, checkpoint 선택, 다른 학습 요인 중 단일 원인을 확정하지 않는다.

학습 초기 pseudo-low Loss 문제는 이전 진단에서 확인됐으며 이번에도 epoch10의 낮은 Val Loss가 좋은 좌표를 보장하지 않았다. epoch40의 Val Mask IoU는 {train['best_val_mask_iou_secondary']['metrics']['val']['mask_iou_mean']:.6f}로 기록됐지만, 모델 선택 규칙을 사후 변경하지 않았고 해당 checkpoint로 Pose 평가·추가 실험을 하지 않았다.

## 선택 후 실사27 개발 비교

| 지표 | Baseline | 회전 증강 |
|---|---:|---:|
| 유효 PnP | {stage1['PnP_valid']}/27 | {real['PnP_valid']}/27 |
| ADD@0.1d | {stage1['ADD_pass']}/27 | {real['ADD_pass']}/27 |
| 평균 inlier 비율, 실패 포함 | {stage1['inlier_ratio_mean_all_frames']*100:.4f}% | {real['mean_inlier_ratio_all']*100:.4f}% |
| ADD 평균, 유효 반환만 | {stage1['valid_pose_error_stats']['add']['mean']:.4f} (n=27) | {fmt(real['mean_ADD_valid'])} (n={real['PnP_valid']}) |

두 평균의 분모가 다르므로 직접적인 평균 정확도 개선/악화로 비교하지 않는다. 증강 모델이 반환한 동일 5프레임에 한정한 ADD 평균은 Baseline {fmt(paired_base)}→증강 {fmt(paired_new)}이나, 작은 선택된 하위집합의 탐색값일 뿐이다. 실사27은 학습·체크포인트 선택·튜닝에 사용하지 않았다.

PnP 구현의 성공/유효 정의는 solver success, R/t 유한, CAD 양의 Z다. 이 정의를 그대로 유지한 Train Front frame83에는 inlier1개와 TE 약6.29e17의 극단적으로 큰 **유한** 값이 포함됐다. 따라서 Train Front의 유효 반환 조건부 ADD/TE 평균은 심하게 왜곡되며, 이를 실용적 Pose 성공으로 해석하지 않는다. NumPy norm overflow 경고가 실행 로그에 기록됐고 CSV의 비어 있지 않은 평가 오차에는 비유한 값이 없음을 별도로 확인했다. 실패 Pose는 분모에서 제외하지 않고 ADD 실패로 센다.

{link(R/'stage3/pbr_comparison.csv')} · {link(R/'stage3/real_comparison_per_frame.csv')} · {link(R/'stage3/pnp_correspondence_geometry.csv')}
'''
save(R/'stage3/회전증강_평가.md',stage3_text)
main=f'''# SMT obj04 HCCEPose 자율 연구 실행 보고서

작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}  
실험 루트: `{R}`  
보고서 생성까지 약 {elapsed:.2f}분 / 최대 90분.

**결론: 기존 Baseline은 실사 개발용 27장에서 ADD@0.1d 0/27이었다. 지정한 회전 증강 단일 실험은 선택된 checkpoint에서 PBR Val의 좌표 및 Pose를 악화시켰고, 실사 ADD는 0/27로 개선되지 않았다.**

## 1. 기존 Baseline과 사전 병목

기존 Baseline은 PBR Train80/Val20(생성 블록 분리), random ResNet34 seed42, batch4/crop256, Adam lr0.0002, AMP float16, GT visible BBox crop이다. 총195epoch/3900 optimizer step 중 PBR Val Loss로 epoch155/3100step을 선택했다. 원본 Loss의 NumPy scalar→tensor 문제는 이전 작업 복사본의 명시적 float 변환을 사용했다.

이전 Train/Val 진단: Front XYZ L2 4.2057/29.4046, ADD 59/80(73.75%)/1/20(5%), Val Mask IoU 0.935782. GT XYZ→HCCE→crop→PnP Oracle은 Val20의 ADD20/20이었다. Val Front Z0 hard bit 오류는44.57%로, Train2.75%에 비해 컸다. Oracle 수치는 네트워크 실제 성능이 아니다.

이번 실험 전 동일 checkpoint를 PBR Val20에 다시 적용한 결과, 프레임별 ADD와 Mask IoU의 최대차이는 모두 {repro['max_ADD_frame_difference']:.1f}/{repro['max_mask_IoU_frame_difference']:.1f}였다. {link(R/'checks/pbr_repro_gate.json')}.

## 2. Stage 1 — 카메라·GT 확인과 실사 Baseline

27개의 RGB·Pose NPY·GT·카메라 JSON 프레임 집합이 일치하고, 원본85파일 해시가 이전 manifest와 같았다. 각 NPY와 GT의 R/t, 회전행렬, JSON K가 일치했다. 실사 CAD는 PBR 학습용 CAD 복사본과 해시가 같고, GT 적용 시 모든 CAD 점이 카메라 앞에 있으며 투영 bbox가 영상 안에 있었다.

실사 원본에서 OpenCV Brown-Conrady `(k1,k2,p1,p2,k3)`로 명시적 K,D,newK=K·무 crop/resize 보정 영상을 새 PNG로 생성했다. 그 영상과 기존 `image_undistorted`의 평균 픽셀 차이는0.4342/255였다. 기존 UND의 정확한 생성 이력은 여전히 미확인이고, 이번 모델 입력에는 출력 K가 명시된 새 영상만 사용했다. 이 비교는 카메라의 물리적 정밀도 인증이 아니다.

GT CAD 투영 bbox를 1.5배 정사각형으로 확장한 Oracle ROI를 전 프레임에 적용했다. 결과는 실사 전체 **27/27 유효 PnP, 0/27 ADD 통과**, 평균 ADD {stage1['valid_pose_error_stats']['add']['mean']:.4f}, 평균 RE {stage1['valid_pose_error_stats']['rotation_deg']['mean']:.4f}°, 평균 TE {stage1['valid_pose_error_stats']['translation']['mean']:.4f}, inlier 비율 평균 {stage1['inlier_ratio_mean_all_frames']*100:.4f}%다. ADD/TE 길이는 CAD 원본 수치 단위이며 실측 mm라고 확정하지 않는다. ROI는 GT를 이용하므로 YOLO를 포함한 E2E 성능이 아니다.

이전 CAD 재투영 검토의 판단 불가 5프레임(066,068,070,072,073)도 제외하지 않았다. GT 자체는 거시적 시각 정합성 기준의 개발용 후보이며, 정밀 공정 허용오차나 독립 최종 테스트 GT로 인증되지 않았다. {link(R/'stage1/실사_Baseline_평가.md')}.

## 3. Stage 2 — 실패 분석

| 환경 | ADD 통과 | 평균 inlier 비율 | 조건 |
|---|---:|---:|---|
| PBR Val20 | 1/20 | 4.0090% | GT visible BBox |
| 실사27 | 0/27 | {stage1['inlier_ratio_mean_all_frames']*100:.4f}% | GT CAD 투영 amodal BBox |

PnP 반환과 ADD 성공은 다르다. 기존 육안 가림 분류상 도구 가림 17장, 가림이 없거나 적은 8장 모두 ADD 통과가0이었다. 2장은 미분류다. 가림을 정량화할 visible Mask GT는 없다. 이 결과는 가림만으로 실패를 설명하기 어렵다는 관찰이지 가림 영향이 없다는 증거가 아니다.

PBR과 실사는 ROI 정의도 다르다. 반사·조명·배경·가림과 객체 자세/형상, 미검증 GT 정밀도, ROI 차이, 학습 좌표 일반화가 후보이며 각 원인은 인과 분리되지 않았다. 000051과000092의 overlay를 시각 확인했으며 예측과 GT 방향·크기 차이가 나타난다. 가림을 모델링하지 않은 wireframe의 한계가 있다. {link(R/'stage2/실사_오류분석.md')}, {link(R/'stage2/priority_examples.png')}.

### 실사 프레임별 Baseline

| 프레임 | 이전 가림 표식 | GT 육안 품질 | 유효 PnP | ADD | inlier 비율 | ADD |
|---|---|---|---|---:|---:|---|
{chr(10).join(rt)}

가림 표식은 이전 사람이 검토한 보고서의 정성 기록이며 자동 생성 visible Mask가 아니다. 전체 원자료: {link(R/'stage2/real_frame_analysis.csv')}.

## 4. Stage 3 — 회전 증강 단일 실험

RGB·Mask·Front/Back XYZ·유효 support에 같은 회전을 적용하고, RGB는 bilinear, label은 nearest, HCCE는 변환된 XYZ에서 다시 생성했다. Crop 내 회전 각도는 표본별 uniform[-20°,20°], 독립 RNG seed42다. 다른 초기화·분할·Loss·optimizer·batch·ValLoss 선택 및 early-stopping 규칙은 유지했다. 지원하지 않는 valid-mask 가중치는 새로 Loss에 삽입하지 않았다.

CPU 정합성 및 CPU 독립 smoke가 통과했고, 실제 GPU 학습은 **단 1회**다. 초기 state는 Baseline epoch0과 전부 일치했다. 유한 Loss로 {train['completed_epochs']}epoch, optimizer {train['successful_optimizer_steps']}회, AMP skip {train['amp_skipped_steps']}회를 수행하고 기존 규칙에 의해 early stop했다. 실사 이미지는 학습에 사용하지 않았다. Baseline은 epoch155에서 선택됐지만 회전 모델은 epoch{train['best_val_loss']['epoch']}에서 선택됐다. 같은 규칙의 결과이지만 평가 checkpoint의 유효 학습량이 다르므로 회전 자체의 효과와 학습 단계의 효과를 완전히 분리할 수 없다.

### PBR Val20 — 고정된 주 checkpoint 비교

| 지표 | Baseline epoch155 | 회전 증강 epoch{train['best_val_loss']['epoch']} |
|---|---:|---:|
{comparison_table}

회전 모델의 Val Front PnP는0/20이므로 평균 RE/TE는 계산할 수 없다. Mask·Front/Back 좌표·ADD는 악화됐다. Front Z0 오류율은 감소했지만 Back Z0 오류율은 증가했다. 두 모델 모두 PBR Val을 모델 선택에 사용한 개발 결과이며 독립 테스트가 아니다. {link(R/'stage3/pbr_comparison.png')}, {link(R/'stage3/validation_loss_curve.png')}.

예측 Front 3D 대응점의 Val 프레임당 unique 평균은 {base_geom['unique_xyz_mean']:.1f}→{aug_geom['unique_xyz_mean']:.1f}, X 표준편차 평균은 {base_geom['x_std_mean']:.4f}→{aug_geom['x_std_mean']:.4f}였다. 회전 모델의 거의 한 축에 몰린 대응점은 PnP 실패와 일치하는 증거다. 원인 자체는 증강·체크포인트 시점·Loss 게이팅 등을 구분할 추가 대조 없이 확정하지 않는다.

### 고정 checkpoint의 실사27 비교

| 지표 | Baseline | 회전 증강 |
|---|---:|---:|
| 유효 PnP | {stage1['PnP_valid']}/27 | {real['PnP_valid']}/27 |
| ADD@0.1d | {stage1['ADD_pass']}/27 | {real['ADD_pass']}/27 |
| 평균 inlier 비율, 반환 실패 포함 | {stage1['inlier_ratio_mean_all_frames']*100:.4f}% | {real['mean_inlier_ratio_all']*100:.4f}% |
| 평균 ADD, 유효 PnP만 | {stage1['valid_pose_error_stats']['add']['mean']:.4f} (n=27) | {fmt(real['mean_ADD_valid'])} (n={real['PnP_valid']}) |

유효 Pose 평균의 분모가 다르므로 두 평균을 직접 비교해 실사 성능 차이를 확정하지 않는다. 양쪽 유효인 동일5프레임의 Baseline/회전 평균 ADD는 {fmt(paired_base)}/{fmt(paired_new)}이나 선택된 소표본 탐색 수치다. 실사는 checkpoint 선택 후 동일 보정 영상·ROI로 평가했고 학습·선택·튜닝에는 사용하지 않았다. {link(R/'stage3/real_comparison.png')}, {link(R/'stage3/real_comparison_per_frame.csv')}.

동일 유효 정의(성공·유한·양의 깊이)에서는 Train Front frame83의 극단적인 유한 TE≈6.29e17과 inlier1개도 포함됐다. 이는 유효 반환과 물리적으로 합리적인 Pose가 다름을 보여주며 조건부 평균을 왜곡한다. 비유한 오차는 CSV에서 발견되지 않았고, 출력된 NumPy overflow 경고와 모든 실패는 보존했다. 임의의 사후 거리 cutoff는 추가하지 않았다.

세부: {link(R/'stage3/회전증강_평가.md')}, {link(R/'stage3/pbr_comparison.csv')}, {link(R/'stage3/pnp_correspondence_geometry.csv')}.

## 5. 결론·한계·다음 연구

**[확인]** 기존 모델은 PBR Val에서 낮은 좌표 일반화와 5% ADD, 실사 개발27에서는 0% ADD를 보였다. 지정한 회전 증강 1회는 기존 선택 규칙에서 PBR Pose·좌표를 악화시켰고 실사 ADD 통과 수를 늘리지 못했다. 개선 효과는 입증되지 않았다.

**[한계]** 하나의 seed, 합성 Val 두 블록과 상관된 실사27프레임, 조건부 GT 및 Oracle ROI다. Train/Val Loss가 예측 Mask에 의존해 낮은 Loss와 좌표 품질이 일치하지 않을 수 있고, 두 주 checkpoint의 학습량이 다르다. 실사 Mask IoU는 실제 visible Mask GT가 없어 계산하지 않았다. 물리 mm 단위와 GT 정밀도도 별도 확인이 필요하다.

**[제안·미실행]** 다음 연구는 추가 실사 튜닝 전에 checkpoint 선택 규칙의 좌표 붕괴 민감도를 별도 실험으로 검증한다. 예를 들어 고정 PBR 검증에서 좌표별 분산·Z0/XYZ 오차와 PnP 유효율을 함께 기록하고, 미리 정한 조건으로만 checkpoint 선택을 비교한다. 이 제안은 Loss/epoch/augmentation을 자동 탐색하라는 뜻이 아니며 다음 실행 전 실험 설계가 필요하다. 이후 새로운 촬영 세션의 독립 실사 테스트와 실제 가림 Mask, 검증된 detector를 확보한다.

## 6. 산출물·보존

| 경로 | 내용 |
|---|---|
| {link(R/'checks/stage1_preflight.json')} | 원본85·매핑·K/GT/CAD 게이트 |
| {link(R/'stage1/real_pose_per_frame.csv')} | Baseline 실사27 결과, 실패 포함 |
| {link(R/'stage2/real_frame_analysis.csv')} | 프레임별 가림 표식·오류 분석 |
| {link(R/'checks/pbr_repro_gate.json')} | 기존 PBR Val 재현 |
| {link(R/'checks/augmentation_alignment.json')} / {link(R/'checks/augmentation_cpu_smoke.json')} | CPU 정합성 및 smoke |
| {link(R/'checks/rotation_training.json')} | 단일 학습과 선택·자원 기록 |
| {link(R/'stage3/pbr_comparison.csv')} | Train/Val Mask·XYZ·Z0·Pose 비교 |
| {link(R/'stage3/real_comparison_per_frame.csv')} | 실사27 동일 프레임 비교 |
| {link(R/'checks/final_integrity.json')} | 원본·이전 작업 해시 재검증 |
| `{R/'stage1/overlays'}` / `{R/'stage3/real_overlays'}` | Baseline/증강 GT 대비 CAD overlay |
| `{R/'logs'}` / `{R/'stage3/checkpoints'}` | 실행 원문·회전 모델 checkpoint |

원본740파일, 이전 세미나366파일, 이전 Baseline61파일, 이전 Pose 진단102파일, 이전 Train/Val 진단63파일의 해시가 기존 manifest와 일치했다. 기존 venv 재사용, 추가 패키지 설치·다운로드 없음. GPU 작업은 순차 실행했고 다른 compute PID가 있으면 중단하는 guard를 사용했다. 종료 시 compute 조회: `{integrity['종료 시 GPU compute 조회'] or '실행 중인 compute process 없음'}`.
'''
save(R/'reports/SMT_obj04_자율연구_최종보고서.md',main)
short=f'''# SMT obj04 자율 실험 핵심 결과

**실사 Baseline ADD는 0/27이며, 회전 증강 1회는 PBR·실사 Pose 개선을 입증하지 못했다.**

| 항목 | Baseline | 회전 증강 |
|---|---:|---:|
| PBR Val Mask IoU | {fmt(bf['mask_iou'],4)} | {fmt(af['mask_iou'],4)} |
| PBR Val Front XYZ L2 | {fmt(bf['xyz_l2'])} | {fmt(af['xyz_l2'])} |
| PBR Val Front Z0 오류 | {float(bf['Z0_hard_error'])*100:.2f}% | {float(af['Z0_hard_error'])*100:.2f}% |
| PBR Val Front PnP 유효 / ADD 통과 | {bf['PnP_valid']}/20 / {bf['ADD_pass']}/20 | {af['PnP_valid']}/20 / {af['ADD_pass']}/20 |
| 실사27 PnP 유효 / ADD 통과 | {stage1['PnP_valid']}/27 / {stage1['ADD_pass']}/27 | {real['PnP_valid']}/27 / {real['ADD_pass']}/27 |

GT CAD 기반 Oracle ROI 조건이다. 실사 GT 5프레임은 이전 시각 검토에서 판단 불가였지만 제외하지 않았다. 실사 GT의 정밀도·물리 단위와 독립성은 미확인이다.

- PBR 재현성: 기존 checkpoint의 Val 프레임별 ADD/Mask IoU 차이0.
- 회전 증강: CPU 정합성·smoke 통과, GPU 학습1회, epoch50에서 early stop. 같은 ValLoss 규칙으로 epoch10 선택(기존 epoch155). 비교 시 선택 epoch와 실제 학습량 차이를 설명한다.
- Val 증강 모델의 XYZ 대응점 X표준편차 평균0.1759로 Baseline25.1331보다 작았다. 좌표 붕괴 가능성을 확인했으나 원인을 단정하지 않는다.
- 회전 모델은 Val Front PnP 0개이므로 RE/TE 평균은 계산 불가다. 실사 유효 PnP 평균도 Baseline n=27, 증강 n=5로 분모가 다르다.
- 이후 checkpoint 선택과 좌표 붕괴의 관계를 별도 통제 실험으로 검증하고, 독립 실사 테스트를 확보한다.

발표용 그림: {link(R/'stage3/pbr_comparison.png')}, {link(R/'stage3/real_comparison.png')}, {link(R/'stage2/priority_examples.png')}.  
상세: {link(R/'reports/SMT_obj04_자율연구_최종보고서.md')}.
'''
save(R/'reports/세미나_핵심결과.md',short)
for p in [R/'reports/SMT_obj04_자율연구_최종보고서.md',R/'reports/세미나_핵심결과.md',R/'stage1/실사_Baseline_평가.md',R/'stage2/실사_오류분석.md',R/'stage3/회전증강_평가.md']:
 for x in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(x).is_file(),x
jsonout(R/'checks/report_validation.json',{'status':'PASS','reports':5,'stage_gates':'all PASS','links_resolved':True,'elapsed_minutes':elapsed,'original_assets_unchanged':True})
print(json.dumps({'final_report':str(R/'reports/SMT_obj04_자율연구_최종보고서.md'),'elapsed_minutes':elapsed,'integrity':integrity,'pbr_val_baseline':bf,'pbr_val_rotation':af,'real':real},ensure_ascii=False,indent=2))
