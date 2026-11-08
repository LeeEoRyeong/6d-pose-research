"""실측 JSON/CSV로 한국어 진단 보고서·세미나 요약·도표를 신규 생성."""
from pathlib import Path
import json,csv,hashlib,re,datetime,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];BASE=R.parent;T=BASE/'runs/baseline_seed42_20261103_214353/trained_only';M=R/'matched_batch4';SUP=R/'supplement';CHANNEL=R/'channel_controls'
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def save(p,v):
    with p.open('x') as f:f.write(v)
def js(p,v):save(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def link(p):return f'[{p.name}]({p})'
def cp(a,b):
    assert not b.exists();shutil.copyfile(a,b)
cp(SUP/'tables/Predicted_XYZ_errors_raw_target.csv',R/'tables/Predicted_XYZ_error_batch4.csv')
cp(M/'tables/PnP_candidate_comparison.csv',R/'tables/PnP_candidate_comparison_batch4.csv')
A=read(R/'checks/summary.json')['stages'][:5];C=read(M/'checks/summary.json')['stages'];EX=read(SUP/'checks/error_summary.json');CS=read(CHANNEL/'checks/summary.json')['stages'];SS=read(SUP/'checks/summary.json')['stages'];A2K=read(R/'cropK_equal_units/checks/summary.json')['stages'][0];REL=read(SUP/'checks/relationship.json');CROWS=read(M/'checks/prediction_results.json')['candidates'];ER=rows(SUP/'tables/Predicted_XYZ_errors_raw_target.csv');GEO=rows(SUP/'tables/independent_CAD_ray_check.csv');BIT=rows(SUP/'tables/HCCE_bit_channel_errors.csv')
# SHA-256 checks: originals, earlier seminar, prior Baseline artifacts.
integrity={}
for label,p,root in [('originals',BASE/'checks/original_manifest.json',None),('previous_seminar',BASE/'checks/artifact_manifest.json',BASE),('previous_baseline',T.parent/'checks/final_artifact_manifest.json',T.parent)]:
    manifest=read(p);bad=[]
    for v in manifest:
        f=Path(v['path']) if root is None else root/v['path']
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=v['sha256']:bad.append(str(f))
    integrity[label]={'checked':len(manifest),'mismatches':bad}
assert not any(x['mismatches'] for x in integrity.values()),integrity
integrity['GPU_compute_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
js(R/'checks/final_integrity.json',integrity)
# Bar plot log mean ADD, labels reflect actual measurements.
im=Image.new('RGB',(1200,650),'white');d=ImageDraw.Draw(im);d.text((50,22),'GT Oracle -> network prediction | Mean ADD (PBR native units, log scale)',fill='black');x0,y0,x1,y1=150,85,1150,535
for exp in range(-5,3):
 y=y1-(exp+5)/7*(y1-y0);d.line((x0,y,x1,y),fill='#dddddd');d.text((25,y-7),f'10^{exp}',fill='black')
for i,(name,v) in enumerate([('A0 exact CAD',A[0]['add_mean']),('A1 GT HCCE',A[1]['add_mean']),('A2 GT crop',A[2]['add_mean']),('Pred Front batch4',C[0]['add_mean'])]):
 x=x0+80+i*230;y=y1-(np.log10(v)+5)/7*(y1-y0);d.rectangle((x,y,x+100,y1),fill='#3275a8' if i<3 else '#cc5544');d.text((x-15,y-22),f'{v:.6g}',fill='black');d.text((x-30,565),name,fill='black')
im.save(R/'figures/GT_to_prediction_ADD.png')
# Correlation on 20 frame means, fixed annotation of 40 and 90.
im=Image.new('RGB',(1100,700),'white');d=ImageDraw.Draw(im);d.text((60,20),f'Front XYZ error vs ADD | Val20 | Pearson r={REL["pearson_FrontGTmask_error_vs_ADD"]:.4f}',fill='black');d.text((60,42),'GT mask, valid labels; each point = one frame; descriptive correlation',fill='black')
front={int(e['frame']):float(e['xyz_l2_mean']) for e in ER if e['branch']=='front' and e['scope']=='GTmask'};cpd={c['frame']:c for c in CROWS if c['stage']=='front'}
for i in range(6):
 x=100+900*i/5;y=580-480*i/5;d.line((100,y,1000,y),fill='#dddddd');d.text((40,y-7),str(i*30),fill='black');d.text((x-5,600),str(i*12),fill='black')
for frame,err in front.items():
 x=100+err/60*900;y=580-cpd[frame]['add']/150*480;d.ellipse((x-5,y-5,x+5,y+5),fill='#d05040' if frame in (40,90) else '#3075aa');d.text((x+7,y-12),f'{frame:06d}',fill='black')
d.text((370,635),'GT-mask Front XYZ mean error (native units)',fill='black');d.text((40,75),'ADD (native units)',fill='black');im.save(R/'figures/XYZ_error_ADD_relation.png')
# HCCE planes: first bit is the high-order plane; Gray-like dependencies matter.
im=Image.new('RGB',(1250,560),'white');d=ImageDraw.Draw(im);d.text((35,18),'HCCE hard-bit error | Val20 GT mask | mean of per-frame rates',fill='black');d.text((35,40),'Plane index0 = first/high-order HCCE plane; planes are not independent binary digits',fill='black')
for bit in range(8):d.text((225+bit*120,72),str(bit),fill='black')
for row,(branch,axis) in enumerate(( (b,a) for b in ('front','back') for a in 'xyz')):
 d.text((30,118+row*65),f'{branch} {axis.upper()}',fill='black')
 for bit in range(8):
  value=float(np.mean([float(q['hard_bit_error']) for q in BIT if q['branch']==branch and q['axis']==axis and int(q['bit_index'])==bit]));x=180+bit*120;y=100+row*65;color=(int(80+170*min(value/.5,1)),int(225-150*min(value/.5,1)),100);d.rectangle((x,y,x+112,y+55),fill=color);d.text((x+22,y+20),f'{value*100:.1f}%',fill='black')
im.save(R/'figures/HCCE_bit_errors.png')
# SO(3) numerical floor and geometry aggregate.
ind={k:{'mean':float(np.mean([float(q[k]) for q in GEO])),'max':max(float(q[k]) for q in GEO)} for k in ['front_cameraZ_error_mean','front_cameraZ_error_max','back_cameraZ_error_mean','back_cameraZ_error_max','A0_RE_with_SO3_GT_deg']}
js(R/'checks/independent_geometry_summary.json',ind)
CAUSES=[
('CAD·카메라 좌표계','PBR 내부 거시 불일치 근거 없음','A0 ADD20/20, 평균0.00001962; PBR GT 라벨 min/max 독립 ray400개 확인','실사 K/newK·절대 물리단위는 이번 검증 대상 아님','현 PBR K·CAD 유지; 실사 provenance 별도 확보'),
('XYZ 라벨 생성','정밀 잔차·경계 누락 확인, 주 오차 원인 근거 약함','A1 ADD20/20 평균0.164691, A2도20/20; GTmask 경계의 무효 라벨356pixel','VisPy bitwise 비교·전체표면 point-to-mesh 미완료','경계 support 분리 및 원본 VisPy 소수 표본 비교'),
('HCCE encode/decode','시험 범위에서 통과','각 축256값 원본 encode→threshold→decode 최대 차이0; A1 uint8 차이0; XYZ float 복원 최대차이2.6866e-6','예측 bit 오류는 존재, encoder 역함수 오류와 구분','원본 encoder/decoder 회귀테스트 유지'),
('Crop→원본 좌표','거시 오류 근거 없음, 작은 샘플링 잔차','A2 GT20/20 평균ADD0.288577; nearest-pixel 대조0.148635; cropK 대조도20/20','half-pixel/최근접·양자화 잔차는 남음','임의 offset 변경 금지; affine K·임계값 단위 명시'),
('네트워크 좌표 예측','현재 가장 큰 병목, Z가 특히 중요','GTmask Front/Back 평균3D오차29.4046/29.1845; GT Z치환 Oracle10/20','왜 Z 예측이 실패하는지(학습실패·일반화·손실/초기화)는 미확정','다음 run 전 Train80 동일 좌표/bit 측정으로 학습 대 일반화 분리'),
('Mask','기여 가능하나 단독 주원인 근거 약함','예측XYZ+GTmask1/20, GTXYZ+예측mask20/20','다른 가림/검출 조건에서 중요하지 않다는 뜻 아님','Mask 경계/무효 support를 명시하고 좌표 개선 우선'),
('PnP·대응점 품질','예측 대응점 품질 저하 및 원본 inlier 집계 문제 확인','front 실제 평균inlier120.75/3095.30, 평균 비율4.01%; Back/Mixed는 Front로 inlier재계산','집계 수정만으로 정확도 개선된다고 확정 불가; 실제max-inlier ADD2/20','작업 복사본에서 inlier 계산을 실제 Points3D로 통일하고 회귀검증'),
('미확정 원인','추가 근거 필요','단일 checkpoint·합성 Val20·같은 Val 다중 후보 비교','대칭 모호성·PBR 외관·증강·pretrained·일반화 효과 미검증','새 독립 테스트와 한 요인 대조 실험; 성능 향상 가설로 표시')]
with (R/'tables/cause_diagnosis.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['원인','판정','확인된 증거','미확인·한계','다음 조치']);w.writerows(CAUSES)
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(read(R/'checks/run_info.json')['start']);elapsed=(now-start).total_seconds()/60
# Formatting helpers.
def stage_table(data):
 return '\n'.join(f"| {x['stage']} | {x['valid']}/{x['n']} | {x['ADD_pass']}/{x['n']} ({x['ADD_rate']*100:.0f}%) | {x['add_mean']:.6g} / {x['add_median']:.6g} | {x['rotation_deg_mean']:.4f} | {x['translation_mean']:.6g} | {x['inliers_mean']:.2f} / {x['inlier_ratio_mean']*100:.2f}% |" for x in data)
def n(x):return f'{x:.6f}'
error_table='\n'.join(f"| {x['branch']} | {x['scope']} | {x['pixels']:,} | {x['xyz_l2_mean_frame_mean']:.4f} | {x['x_MAE_frame_mean']:.4f} | {x['y_MAE_frame_mean']:.4f} | {x['z_MAE_frame_mean']:.4f} | {x['hard_bit_error_rate_frame_mean']*100:.2f}% |" for x in EX)
cause_table='\n'.join('| '+' | '.join(c)+' |' for c in CAUSES)
example_table=[]
for frame in (40,90):
 p=cpd[frame];f=next(e for e in ER if int(e['frame'])==frame and e['branch']=='front' and e['scope']=='GTmask');b=next(e for e in ER if int(e['frame'])==frame and e['branch']=='back' and e['scope']=='GTmask')
 example_table.append(f"| {frame:06d} | {float(f['xyz_l2_mean']):.4f} | {float(b['xyz_l2_mean']):.4f} | {p['add']:.4f} | {p['rotation_deg']:.4f} | {p['translation']:.4f} | {p['inliers_actual']}/{p['correspondences']} ({p['inlier_ratio_actual']*100:.2f}%) | {'성공' if p['add_pass'] else '실패'} |")
text=f'''# HCCEPose SMT obj04 Pose 오차 원인 분석

작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}

**결론: GT XYZ→PnP는 Val20 전체에서 통과했고, 큰 오차는 네트워크 예측 XYZ를 사용했을 때 발생했다. 현재 가장 강한 병목 증거는 좌표 예측이며, 특히 Z 축이다.** Mask만 GT로 치환해도 ADD는1/20에 머물렀고, XYZ를 GT로 치환하면 기존 예측 Mask로도20/20이었다. GT를 사용한 결과는 모두 **Oracle 진단**이며 실제 모델 성능이 아니다.

- 진단 루트: `{R}`
- 기존 모델: `{T/'checkpoints/best_val_loss.pt'}` (epoch155)
- 원본 소스: `/RESEARCH_DATA/HCCEPose` — 수정 없음
- 데이터: `/RESEARCH_DATA/HCCEPose_experiments/realistic_final100/train_pbr/000000`
- 기존 venv 재사용, 추가 설치·다운로드·재학습 없음. GPU에서는 제한된 checkpoint 추론만 수행했다.
- 수치·CSV·그림은 이번 신규 진단 결과다. 기존 Baseline 학습·실사27 검증 수치는 구분해 재사용했다.
- 보고서 생성 시 계산 시작 이후 약{elapsed:.2f}분. 사용자 요청 접수부터 포함한 전체 작업도60분 이내이며 이후 신규 실험을 종료한다.

## 1. 질문별 답변

1. **오차가 처음 발생한 단계:** A0의 수치 오차 후 A1의8bit 양자화/라벨 표본화에서 작은 비영 오차가 처음 확인된다. A2까지는 ADD20/20을 통과했다. **실용적으로 큰 Pose 오차는 예측 HCCE를 복호화한 XYZ를 사용할 때 발생했다.**
2. **정답 XYZ에서도 실패하는가:** 이번 GT Front full-resolution·crop 및 GT Back crop은 모두 유효 PnP20/20, ADD20/20이었다. 모든 모델·카메라·형상에서 성공한다는 보장은 아니다.
3. **예측 XYZ 오차:** GTmask 내부 프레임별 평균 L2의20프레임 평균은 Front29.4046, Back29.1845(PBR 원본 단위). GT 양자화 좌표와 같은128×128 픽셀에서 비교했다.
4. **가장 큰 병목:** 예측 좌표/대응점 품질. Front Z MAE16.3539, 첫 Z HCCE plane 오류율44.55%. GT Z만 치환한 Oracle ADD10/20은 Z의 중요성을 지지한다. Y 치환도7/20으로 올라 복합 오류임을 함께 보여준다.
5. **다음 실험의 첫 조치:** 추가 학습 전 Train80에도 같은 좌표/bit 진단을 적용해 학습 자체의 실패와 Val 일반화 실패를 분리한다. 구현 수정 우선순위는 작업 복사본의 PnP inlier를 실제 사용한 `Points_3D`로 일관되게 계산하고 GT/Back/Mixed 회귀검증하는 것이다. 이 수정만으로 Front Baseline의 정확도가 개선된다고 주장하지 않는다.
6. **발표 가능한 결론:** “Mask IoU0.9358이어도 좌표 예측 오차가 크면 ADD5%에 머문다. GT 좌표 Oracle은100%여서 이번 Val20의 주요 오차를 좌표 예측 단계로 좁혔다.” 실사 정확도나 개선 모델 성능 확보 주장은 할 수 없다.

## 2. 원본 알고리즘과 좌표 규약

| 항목 | 실제 규약·근거 |
|---|---|
| Front / Back | 카메라 ray의 최소/최대 표면 깊이. 원본 렌더러 GL_LESS/GL_GREATER 의미, `HccePose/bop_loader.py:rendering_bop_dataset_back_front` |
| 3D 좌표 | object-to-camera `Xc=R Xm+t`; 역변환 행벡터 `(Xc-t)@R` |
| 정규화 | `(Xm-min_xyz)/size_xyz`, clip[0,1], round×255, uint8. min=(-45.25,-45.25,-32.65763855), size=(90.5,90.5,65.31527710) |
| 채널 | cv2 읽기/쓰기 배열의 channel0=X,1=Y,2=Z. PNG viewer의 일반 RGB 표시와 혼동하지 않음 |
| HCCE | 축당8개 soft plane,3축24개, Front/Back 총48개. 원본 `train_bop_dataset_back_front.hcce_encode` |
| 복호화 | soft GT를>0.5로 이진화 후 원본 `HccePose_BF_Net.hcce_decode`, 누적 절댓값 차분×2의 거듭제곱. `/255*size+min`으로 모델 좌표 복원 |
| 예측 경로 | 원본 `inference_batch` sigmoid threshold0.5. 모델의 raw 회귀 출력과 threshold 후 bit·3D는 서로 다른 평가 대상 |
| 영상·K | PBR1280×720, 무왜곡, 프레임별 `scene_camera.json` K. 원본 K를 사용하며 실사 K/newK 가정은 이번 진단에 사용하지 않음 |
| Crop | GT visible BBox pad1.5, RGB256, code/Mask128. 원본 affine crop과 nearest label 표본화 |
| 원본 좌표 복원 | `u=x0+j*w/128`, `v=y0+i*h/128`, 원본 `inference_batch`와 동일. 이 좌표에 원본 K 적용 |
| 단위·지표 | PBR 원본 좌표 단위; provenance에 mm→m0.001 기재, 실제 물리 치수 독립 계측 없음. diameter139.378694855, ADD 임계13.9378694855 |

`hcce_encode`의 soft target `2*code-1`과 네트워크 raw 출력의 MAE, threshold 후 hard-bit 오류, 복호화 XYZ의 L2/축 MAE를 분리했다. 총 학습 Loss와 혼동하지 않는다. HCCE plane은 독립적인 일반 binary digit이 아니며 앞 plane 오류는 뒤 복호화에도 영향을 줄 수 있다.

## 3. 실험 A — GT Oracle Round-trip

네트워크 예측을 사용하지 않았다. 모든 R/t는 실제 원본 `solve_PnP` 호출 반환값이다. GT 행렬을 반환값으로 복사하지 않았다.

- **A0:** CAD1439개 꼭짓점을 GT R/t/K로 정확히 투영하고 2D–3D를 PnP에 제공. 가려진 CAD 점도 포함한 기하 Oracle이다.
- **A1:** 기존 full-size Front PNG의 유효 GT visible 픽셀을 원본 HCCE로 encode→threshold→decode한 후 정수 원본 픽셀과 PnP. 라벨의 명시적 zero-invalid는 제외했다.
- **A2:** 기존 학습과 같은 crop·128출력·원본 좌표 복원. `GTmask`는 baseline과 같은 Mask support, `valid`는 추가로 라벨 유효 영역과 교집합한 대조다.
- **nearest control:** crop이 실제 선택한 원본 nearest 픽셀 좌표를 대응점으로 사용해 재표본화 위치의 작은 차이를 분리했다.

| 경로 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
{stage_table(A)}

A1에서 HCCE decode uint8와 원본 label의 차이는20장 모두0이다. 축별0..255 exhaustive 검사도 최대 오차0이다. GT를 원래3D로 복원하는 단계는 통과했다.

평균 correspondence reprojection error는 A0={A[0]['reprojection_mean_px_mean']:.8f}px, A1={A[1]['reprojection_mean_px_mean']:.6f}px, A2 GTmask={A[2]['reprojection_mean_px_mean']:.6f}px, A2 valid={A[3]['reprojection_mean_px_mean']:.6f}px, nearest control={A[4]['reprojection_mean_px_mean']:.6f}px다. 이는 **PnP에 제공한 3D 대응점을 반환 Pose로 투영한 residual**이며 실제 이미지의 CAD contour 정합성 오차와 다르다.

### 라벨·crop 추가 대조

- GT Back crop도 ADD20/20, 평균ADD{SS[0]['add_mean']:.6f}, 평균RE{SS[0]['rotation_deg_mean']:.6f}deg였다.
- GTmask crop 내 명시적 무효 라벨은20장 합계356pixel이다. 제거하면 A2 mean residual이1.175377→0.429055px로 줄지만 제거 전에도 ADD20/20이다. 작은 경계 잔차를 숨기지 않는다.
- crop-K 대조는 `Kcrop=A K`, `A=[[128/w,0,-x0*128/w],[0,128/h,-y0*128/h],[0,0,1]]`를 사용했다. RANSAC threshold도1.5 원본px→`1.5*128/w` crop px로 변환했다. ADD20/20, 평균ADD{A2K['add_mean']:.6f}. 중간 보조 대조의1.5 crop-px 고정 결과는 임계값 단위가 달라 별도 폴더에 남겼고 주 비교에는 사용하지 않는다.
- 독립 Moller–Trumbore ray20개/프레임, 총400개를 CAD 삼각면과 교차시켰다. 같은+0.5 ray의 최소/최대 깊이와 양자화 Front/Back을 비교한 camera-Z 평균 오차는{ind['front_cameraZ_error_mean']['mean']:.6f}/{ind['back_cameraZ_error_mean']['mean']:.6f}, 전체 최대는{ind['front_cameraZ_error_max']['max']:.6f}/{ind['back_cameraZ_error_max']['max']:.6f}다. 내부 표본만 검사했으며 전체 표면 검증은 아니다.
- A0의 raw trace RE 약0.01439deg는 float32 GT R이 엄밀한 SO(3)가 아닌 영향이 포함된다. GT R을 SVD로 SO(3)에 투영한 **별도 수치 대조**의 평균 RE는{ind['A0_RE_with_SO3_GT_deg']['mean']:.9f}deg다. 원본 GT는 수정하지 않았다.
- CPU renderer와 원본 VisPy의 bitwise 비교, 실사 카메라·CAD 물리 정합성은 이번에도 미완료다. A0 자기일관성만으로 실물 GT 정밀도를 인증하지 않는다.

근거: {link(R/'tables/GT_XYZ_Oracle_PnP.csv')}, {link(R/'tables/GT_HCCE_Roundtrip.csv')}, {link(R/'checks/hcce_exhaustive.json')}, {link(SUP/'tables/independent_CAD_ray_check.csv')}, {link(R/'cropK_equal_units/tables/cropK_equal_threshold_Oracle.csv')}. 프레임별 K·BBox·support는 {link(R/'checks/geometry.json')}.

## 4. 실험 B — 예측 HCCE와 복호화 XYZ 오차

**주 결과는 기존 Baseline과 동일 batch4·CUDA AMP float16이다.** epoch155 가중치 로딩 후 eval/inference_mode만 사용했다. same affine·nearest crop·same min/size·same128×128 index에서 GT/예측을 비교했다. Front와 Back은 각자 GT branch와 비교했다.

GT XYZ는 기존 양자화 라벨의 모델 좌표이고 예측 XYZ는 원본 decoder 출력이다. 명시적 무효 GT label pixel은3D 오차 계산에서 제외하고 n을 CSV에 기록했다. `PredMask_validLabel`은 예측 Mask와 GT label 유효영역의 교집합이며, **GT가 없는 background의 좌표 오차를 만들어 계산한 것이 아니다.**

| branch | 평가 영역 | 전체 유효 pixel | 평균3D L2 | X MAE | Y MAE | Z MAE | hard bit 오류율 |
|---|---|---:|---:|---:|---:|---:|---:|
{error_table}

표는 **프레임별 평균의20장 평균**이다. Front GTmask의 pixel-weighted 평균L2는29.532293이고 표의29.404647과 구분한다. 프레임별 median/p95·raw soft target MAE도 CSV에 있다. 평균median을 전체pixel의 pooled median이라고 해석하지 않는다.

- Front GTmask raw 출력 대 `2*HCCE-1` 평균MAE0.468275, Back0.468172. 동적 가중치·Mask Loss를 포함한 총 Loss 값이 아니다.
- Front 첫 HCCE plane 오류율은 X15.27%, Y9.57%, **Z44.55%**; Back Z는43.11%였다. Z의 큰 coarse 오류가 확인된다.
- Front 예측을 Back GT에 잘못 대입해도 평균L2는29.7721로 더 작아지지 않았다(정상29.4046). Back도 교차 비교30.4567이 정상29.1845보다 크다. 단순 Front/Back 파일 뒤바뀜이 주원인이라는 근거를 얻지 못했다.
- Z 상반영역 점유는 Front GT약50.09%/예측54.32%, Back GT50.93%/예측47.34%다. 전체가 한쪽 부호로 붕괴했다고 단정할 수 없다.
- Front GTmask 평균좌표오차와 Front ADD의 Pearson r={REL['pearson_FrontGTmask_error_vs_ADD']:.6f}(n=20). 기술적 양의 관계이며 인과나 독립 통계 검정을 의미하지 않는다.

### 성공40 / 실패90 비교

| 프레임 | Front 평균XYZ오차 | Back 평균XYZ오차 | ADD | RE(deg) | TE | 실제 inlier / 입력 | ADD 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(example_table)}

성공40도 대부분의 좌표가 정확하다는 뜻이 아니다. 두 표본 모두 correspondence inlier가 소수이며 RANSAC 반환 성공만으로 품질을 보장하지 않는다.

그림: {link(M/'figures/XYZ_GT_prediction_000040.png')}, {link(M/'figures/XYZ_GT_prediction_000090.png')}, {link(R/'figures/HCCE_bit_errors.png')}, {link(R/'figures/XYZ_error_ADD_relation.png')}.

CSV 주본: {link(R/'tables/Predicted_XYZ_error_batch4.csv')}, {link(SUP/'tables/HCCE_bit_channel_errors.csv')}. 초기 `tables/Predicted_XYZ_error.csv`는 batch1 진단이다. 그 파일의 `soft_hcce_MAE`는 tanh 변환을 추가한 보조값이므로 학습 target의 raw MAE로 쓰지 않는다. **raw 오차의 확정치는 supplement의 `raw_output_target_MAE` 열**이다. 초기 파일도 삭제·덮어쓰지 않고 차이를 명시했다.

### Batch 재현성 확인

초기 단일 이미지 진단은 Front ADD2/20이었다. 기존과 같은 batch4에서는 **1/20, ADD평균62.2473843023, RE81.0261168561deg, TE60.9093847786**으로 기존 결과가 재현됐다. AMP·batch 변화가 일부 threshold/RANSAC 결과에 영향을 줄 수 있어 이를 정확도 개선으로 보고하지 않는다. 주 표는 batch4이며 batch1 CSV/JSON은 별도 보조 기록이다.

## 5. 실험 C — PnP 후보 비교

모든 후보는 동일 checkpoint·Val20이다. 원본 RANSAC EPNP,150iterations, 프레임별 RNG seed=42+frame, 기본1.5 원본pixel. Front/Back/Mixed 모두 GT를 이용해 Pose 후보를 생성하지 않았다. GT Mask·GT XYZ가 포함된 후보만 명시적으로 Oracle로 분리했다.

- `front`, `back`: 각 branch 좌표.
- `mixed_bf`: 원본처럼 선택된 대응점 순서의 번갈아 절반을 Back으로 치환.
- `mixed_bfu`: 원본의 Front/Back 및 ray상 보간 혼합.
- `topConfidence256/1024`: GT 없이 mask probability×양 branch 평균 bit margin 중 작은 값으로 순위를 매겨 일부 대응점 선택. 보정된 confidence는 아니다.
- threshold3/5는 원본pixel 단위다. 개선을 보장하는 권장 설정이 아니라 진단 후보다.

| 후보 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
{stage_table(C)}

**후보 선택 대조 — 고정4개(Front/Back/BF/BFU) 중 선택:**

| 선택 방식 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
{stage_table(SS[2:])}

`source_maxReportedInliers`는4후보 중 원본 reported count 최대를 고르는 진단 규칙이며 원본 keypoint subset prior까지 복원한 전체 Tester 재현은 아니다. `actual_maxUsedInliers`는 GT 없는 실제 대응점 inlier 최대 선택이다. `GT_minADD_Oracle`은 **프레임별 GT ADD가 최소인 후보를 고른 제한된4후보의 Oracle 상한**이며 실제 추론 규칙이 아니다. 그것조차3/20이다. 후보만 바꿔 좌표 품질 문제를 해소했다고 볼 수 없다.

### 확인된 원본 inlier 집계 문제

원본 `solve_PnP`는 Back/Mixed의 `Points_3D`로 Pose를 풀고, op2 반환 전 `cv2.projectPoints(pred_front, ...)`로 inlier를 재집계한다. 이번 Back/BF/BFU는 각각20/20프레임에서 실제 사용한 대응점 기준 count와 달랐다. 실제 count는 OpenCV에 넘긴 float32 `Points_3D`를 capture하여 반환 Pose로 재투영하고 동일 threshold로 다시 계산했다. RANSAC raw inlier count도 CSV에 별도 저장했다.

Front는 이 branch 불일치의 영향을 받지 않으므로 이 버그가 **Front Baseline ADD5%의 원인이라고 주장하지 않는다.** 일부 임계점에서는 float32/float64 재투영의 경계 판정으로1개 count 차이도 생길 수 있다. 집계 수정 후에도 실제 max-inlier 대조는2/20이고 평균ADD는81.5700으로 낮지 않았다.

근거: {link(R/'tables/PnP_candidate_comparison_batch4.csv')}, {link(M/'tables/PnP_candidates_per_frame.csv')}, {link(SUP/'tables/PnP_candidate_selectors.csv')}, 원본 `/RESEARCH_DATA/HCCEPose/HccePose/PnP_solver.py:solve_PnP`, `solve_PnP_comb`.

## 6. 좌표·Mask 치환을 통한 병목 분리

| 치환 조건 | ADD 통과 | 해석 |
|---|---:|---|
| 예측XYZ + 예측Mask | 1/20 | 실제 checkpoint의 기존 Baseline 조건 |
| 예측XYZ + GTmask | 1/20 | Mask만 정답으로 바꿔도 주 오차는 남음, Oracle |
| GTXYZ + 예측Mask | 20/20 | 같은 예측Mask로도 충분한 GT 대응점 확보 가능, Oracle |
| 예측Front + GTvalidMask | 1/20 | 아래 축 치환의 통제조건, Oracle Mask |
| GT X만 치환 | 2/20 | Oracle, 실제 성능 아님 |
| GT Y만 치환 | 7/20 | Oracle, 실제 성능 아님 |
| GT Z만 치환 | 10/20 | Oracle, 평균ADD21.6841, Z 좌표 중요성의 추가 증거 |

축 치환은 정답Mask·유효라벨 조건으로 통제하며 다른 두 축은 예측을 유지했다. 혼합 좌표는 물리적 단일 표면과 항상 일치하지 않아 정식 개선 알고리즘이 아니다. Z만으로 전체 원인을 설명할 수도 없다.

근거: {link(CHANNEL/'tables/GT_channel_replacement_Oracle.csv')}.

## 7. 원인별 판정과 미확정 가설

| 원인 | 판정 | 확인된 증거 | 미확인·한계 | 다음 조치 |
|---|---|---|---|---|
{cause_table}

**다음 변경 제안은 미실행이다.** 먼저 실제 대응점 inlier 집계를 복사본에서 바로잡아 비교 지표를 안정화한다. 학습 연구는 Z 중심의 raw/bit/XYZ 검증 지표를 추가하고 Train80 대 Val20을 비교하는 것부터 시작한다. 그 결과 없이 epochs·lr·PnP threshold를 무작정 바꾸지 않는다. 이후 pretrained/증강/좌표 손실 후보를 한 요소씩 비교하고 개선 효과는 새 결과로 검증한다. 대칭이나 단위를 성능이 좋아지는 방향으로 임의 변경하지 않는다.

## 8. 산출물과 재현

| 파일 | 내용 |
|---|---|
| {link(R/'tables/GT_XYZ_Oracle_PnP.csv')} | A0 각 프레임의 실제 PnP 결과·실패 원인 |
| {link(R/'tables/GT_HCCE_Roundtrip.csv')} | A1/A2 및 nearest 대조 |
| {link(R/'tables/Predicted_XYZ_error_batch4.csv')} | 주 좌표·raw HCCE 오차 |
| {link(R/'tables/PnP_candidate_comparison_batch4.csv')} | 주 후보별 mean/median ADD·RE·TE·inliers |
| {link(R/'tables/cause_diagnosis.csv')} | 원인별 판정 표 |
| {link(R/'checks/returned_poses.json')} | A와 초기 batch1의 실제 반환R/t |
| {link(M/'checks/returned_poses.json')} | 주 batch4 후보의 실제 반환R/t |
| {link(R/'figures/GT_to_prediction_ADD.png')} | GT 경로 대 예측 경로 평균ADD |
| {link(M/'figures/front_000040_overlay.png')} | 성공40, GT녹색/예측자홍 CAD wireframe |
| {link(M/'figures/front_000090_overlay.png')} | 실패90, 같은 표시 |
| {link(R/'figures/A1_full_HCCE_000090_overlay.png')} | 같은90의 GT 라벨 Oracle PnP overlay |
| {link(R/'checks/final_integrity.json')} | 원본·이전 산출물 SHA-256 보존 확인 |

주 평가 CSV에는 실패를0오차로 채우지 않고 명시적 failure를 남기는 정책을 적용했다. 이번 후보들은 모두 유효반환20/20이라 반환 실패 표본은 없었고, 이는 실패처리 경로 전체가 테스트됐다는 의미는 아니다. ADD 실패와 PnP 반환 실패를 구분한다. 성공률 분모는 항상20이다.

실행 순서와 코드(이미 생성된 결과를 덮어쓰지 않고 새 경로로만 재현):

```bash
{BASE/'env/bin/python'} -B {R/'scripts/diagnose.py'}
{BASE/'env/bin/python'} -B {R/'scripts/matched_batch4.py'}
{BASE/'env/bin/python'} -B {R/'scripts/supplement.py'}
{BASE/'env/bin/python'} -B {R/'scripts/channel_controls.py'}
{BASE/'env/bin/python'} -B {R/'scripts/cropK_equal_units.py'}
```

위 스크립트는 절대 기존 학습 파일을 수정하지 않으며 같은 출력이 있으면 재사용 또는 실패할 수 있으므로 현재 폴더에 그대로 재실행하지 않는다. 새 진단 root의 경로 설정과 예산을 먼저 고정한다. `logs/`에 각 실행 원문을 보존했다. 코드의 원본 함수는 AST 추출로 재사용했고 네트워크는 이전 학습의 명시적 float 타입 호환 복사본을 사용했다. 이번 진단 중 CUDA busy 확인에서 다른 compute PID가 없었으며, 최종 query는 `{integrity['GPU_compute_processes_after'] or '실행 중인 compute process 없음'}`이다.

원본740파일, 이전 세미나366파일, 이전 Baseline61파일의 SHA-256이 모두 기존 manifest와 같았다. 기존 source·데이터·checkpoint·report·venv를 보존했다. 실사27은 이번 학습/평가/후보선택에 사용하지 않았다.

## 9. 세미나 발표 결론과 제한

발표 문장: **“정답 XYZ를 동일한 HCCE·crop·PnP 경로에 넣으면 Val20 전체가 통과한다. 예측 좌표는 평균 약29 PBR단위 오차이며, 정답Mask 치환만으로는 ADD5%가 개선되지 않았다. 주요 병목은 좌표 예측, 특히 Z 축으로 좁혀졌다.”**

GT Z 치환50%, Mixed15%, GT 후보선택15%는 서로 조건이 다른 Oracle/탐색 대조다. 기존 Baseline 정확도가5%에서50%로 개선됐다고 발표하면 안 된다. 독립 테스트·실사 추론·추가 학습·구조 개선·원본 VisPy 대조는 미수행이다.
'''
save(R/'reports/HCCEPose_Pose오차_원인분석.md',text)
seminar=f'''# 세미나 추가 실험 요약 — 6D Pose 병목 진단

**핵심 메시지:** GT 좌표는 PnP를 통과하지만 예측 좌표는 큰 오차를 보인다. PBR Val20의 가장 큰 병목을 좌표 예측, 특히 Z로 좁혔다. 재학습·실사 모델 평가 결과가 아니다.

## 실제 확인 수치

| 조건 | ADD@0.1d | 평균ADD |
|---|---:|---:|
| A0 CAD 기하 Oracle | 20/20 | 0.00001962 |
| A1 GT HCCE full-size | 20/20 | 0.164691 |
| A2 GT HCCE crop | 20/20 | 0.288577 |
| 실제 epoch155 Front, batch4 | 1/20 | 62.247384 |
| 예측XYZ + GTmask Oracle | 1/20 | 61.755297 |
| GTXYZ + 예측Mask Oracle | 20/20 | 0.270927 |

GTmask의 평균XYZ L2는 Front29.4046 / Back29.1845(PBR단위), Front Z MAE16.3539다. Front 첫 Z HCCE plane 오류율44.55%. GT Z축만 치환하면10/20이나 **실제 모델 성능이 아닌 Oracle 원인 분석**이다.

## 발표용 3장 구성

### 1. GT→PnP 대조로 기하 경로 분리

- 한 문장: “같은 K·CAD·crop 경로에 GT XYZ를 넣으면20/20 통과한다.”
- 숫자: A0/A1/A2 ADD20/20, encoder256값 최대오차0.
- 그림: {link(R/'figures/GT_to_prediction_ADD.png')}, {link(R/'figures/A1_full_HCCE_000090_overlay.png')}.
- 설명: 각 결과는 실제 solver 반환R/t 평가다. Oracle은 GT를 사용한다. 작은 양자화·crop 잔차는 남는다.

### 2. Mask가 아닌 좌표 예측의 병목

- 한 문장: “정답Mask로 바꿔도 ADD1/20이며, 예측Mask에 GTXYZ를 주면20/20이다.”
- 숫자: Front29.4046 / Back29.1845, Front X/Y/Z MAE12.3937/8.6497/16.3539.
- 그림: {link(M/'figures/XYZ_GT_prediction_000040.png')}, {link(M/'figures/XYZ_GT_prediction_000090.png')}, {link(R/'figures/HCCE_bit_errors.png')}.
- 설명: 같은128픽셀·같은모델좌표에서 비교했다. Z coarse plane 오류44.55%. GT Z치환50%는 원인 분석용 상한 대조다.

### 3. 실제 PnP 실패와 다음 조치

- 한 문장: “유효 Pose 반환은 정확도를 뜻하지 않으며, 대응점 품질을 먼저 개선해야 한다.”
- 숫자: 실제Front20/20반환이나 ADD1/20, 평균inlier120.75/비율4.01%. Mixed BFU3/20은 같은Val의 탐색 결과.
- 그림: {link(M/'figures/front_000040_overlay.png')}, {link(M/'figures/front_000090_overlay.png')}, {link(R/'figures/XYZ_error_ADD_relation.png')}.
- 설명: 원본 Back/Mixed inlier가 Front 기준으로 집계되는 문제를 별도 확인했다. 이것은 Front Baseline5%의 직접 원인은 아니다. 먼저 집계 회귀검증과 Train80 좌표진단으로 학습/일반화 문제를 분리한다.

## 말할 수 있는 것 / 아직 말할 수 없는 것

- 확인: Val20 GT 기하·HCCE·crop 경로 통과, epoch155 batch4 기존수치 재현, 예측XYZ 큰오차, Z치환Oracle 민감도.
- 확인 안 됨: Z 실패의 학습상 원인, 실제모델 개선, 실사27 정확도, 독립 테스트 성능, VisPy bitwise 동등성.
- 초기 batch1 Front2/20은 AMP·batch 조건이 다른 보조 기록이다. 기존 batch4 주 결과1/20을 교체하지 않는다.
- 원본·기존산출물 해시 일치, 기존checkpoint/venv 보존, 추가학습·설치 없음.

기존5슬라이드에서1–3은 유지하고4에는 GT→PnP와 XYZ비교를,5에는 실제실패overlay·inlier 집계문제·다음좌표진단을 넣는 것이 좋다. 별도의50% 개선 모델이 생겼다고 표시하지 않는다.

상세: {link(R/'reports/HCCEPose_Pose오차_원인분석.md')}
'''
save(R/'reports/세미나_추가실험_요약.md',seminar)
# Validate all markdown file links resolve before declaring completion.
for p in (R/'reports').glob('*.md'):
 links=re.findall(r'\]\((/[^)]+)\)',p.read_text());missing=[x for x in links if not Path(x).is_file()];assert not missing,(p,missing)
js(R/'checks/report_validation.json',{'reports':2,'markdown_links_resolved':True,'report_generated_utc':now.isoformat(),'computation_to_report_minutes':elapsed,'primary_inference_batch':4,'originals_and_previous_artifacts_unchanged':True})
print(json.dumps({'reports':[str(p) for p in (R/'reports').glob('*.md')],'integrity':integrity,'minutes_since_computation_start':elapsed},ensure_ascii=False,indent=2))
