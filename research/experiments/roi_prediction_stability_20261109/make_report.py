import csv,json,pathlib,collections
P=pathlib.Path(__file__).resolve().parent;O=P/'full';s=json.loads((O/'condition_summary.json').read_text());data=list(csv.DictReader((O/'per_frame.csv').open()));frames=list(csv.DictReader((O/'frame_sensitivity.csv').open()));m=json.loads((O/'manifest.json').read_text());sm=json.loads((P/'smoke/smoke_check.json').read_text())
fail=[]
for r in data:
 reason=[]
 if r['baseline_pnp_valid']!='True':reason.append('baseline_pnp_failure_reference_missing')
 if r['pnp_valid']!='True':reason.append('condition_pnp_failure')
 if r['rotation_sym_deg'] and float(r['rotation_sym_deg'])>30:reason.append('prediction_rotation_change_gt_30deg')
 if reason:fail.append({'frame':r['frame'],'condition':r['condition'],'observed':';'.join(reason),'baseline_pnp_valid':r['baseline_pnp_valid'],'pnp_valid':r['pnp_valid'],'rotation_sym_deg':r['rotation_sym_deg'],'translation_sym_aligned_cad_units':r['translation_sym_aligned_cad_units'],'xyz_median_cad_units':r['xyz_median_cad_units'],'cause':'unknown'})
with (O/'failure_cases.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(fail[0]));w.writeheader();w.writerows(fail)
def fmt(v):return '—' if v is None else f'{v:.2f}'
table='\n'.join(f"| {x['condition']} | {x['pnp_valid']}/27 | {x['matched_pose_count']} | {fmt(x['rotation_sym_deg']['median'])} / {fmt(x['rotation_sym_deg']['p95'])} | {fmt(x['translation_sym_aligned_cad_units']['median'])} / {fmt(x['translation_sym_aligned_cad_units']['p95'])} | {fmt(x['xyz_median_cad_units']['median'])} / {fmt(x['xyz_median_cad_units']['p95'])} | {fmt(x['time_total_ms']['median'])} |" for x in s)
report=f'''# GT 비의존 ROI 예측 안정성 분석

2026-11-09. 이 결과는 **같은 영상에서 ROI 변화에 따른 모델 예측의 일관성**이다. GT pose와 비교하지 않았고, 예측이 안정적이라는 사실은 정확도·성능 향상·GT 신뢰성을 증명하지 않는다. 앞선 GT 신뢰성 NO-GO는 유지한다. 학습은 실행하지 않았고 BB100과 oracle ROI는 제외했다.

## 입력과 방법

실사 개발27의 RAW RGB를 기존 K,D로 undistort하고 newK=기존 K로 둔 결과가 기존 controlled PNG와 프레임별 픽셀 일치함을 확인했다. 기존 YOLO11n이 출력해 저장해 둔 **예측 bbox**를 사용했다. 이번에는 detector를 재실행하지 않았고, GT pose·GT 기반 ROI·BB100 데이터는 분석에 사용하지 않았다. 기존 detector 결과 CSV의 GT 관련 열은 계산에서 참조하지 않았다. 공식 EfficientNet-B4 체크포인트 strict=True 로드, FP16, 원본 crop/256 전처리, front XYZ PnP, 1.5 px·150 iteration·seed=42+frame을 고정했다.

원래 YOLO bbox에 기존 pad 1.5를 적용한 **정사각 ROI**를 기준으로 크기 0.85·1.00·1.20·1.40을 비교했다. 위치는 기준 1.00 정사각 ROI의 폭/높이 대비 x 또는 y를 각각 −10%, −5%, +5%, +10% 이동했다. 축을 독립적으로 변화시켜 기준 포함 12조건 × 27장 = 324회다.

같은 영상 기준 1.00 예측에 대한 각 조건의 차이를 계산했다. 기본 PnP가 실패한 프레임 68은 해당 조건 PnP가 성공해도 pose 변화의 기준이 없어 결측으로 두었다. 각 조건의 PnP 실패도 pose 변화 분포에서 제외하고 별도로 수를 표시한다. XYZ는 PnP 성공 여부와 관계없이 기준과 변경 ROI의 예측 mask가 겹친 영역에서 front XYZ 차이를 계산했으며 공유 pixel이 없으면 결측이다. 두 출력 grid를 이미지 좌표로 맞춘 뒤 nearest-neighbor 재표본화했으므로 경계·양자화 및 교집합 선택의 영향을 받는다. 1.00의 0은 기준을 자기 자신과 비교한 결과다.

회전은 (1) 일반 SO(3) 각도와 (2) 공식 대칭 중 `R_base @ S_R`와의 각도가 최소인 대칭 고려 각도를 따로 기록한다. 병진은 (1) 예측 t의 직접 차이와 (2) **회전 각도가 최소인 동일 대칭**의 `t_base + R_base @ S_t`를 반영한 차이를 따로 기록한다. 대칭 변환은 `T_base @ S` 순서이며 회전 기준으로 대칭을 선택했으므로 병진까지 동시에 최소화한 값은 아니다. `per_frame.csv`와 `condition_summary.csv`에 두 종류 모두 있다. 모델 CAD 단위는 BOP mm이나 이 실사 환경의 물리적 mm 대응 정확도는 독립 검증되지 않았다.

## 조건별 결과

회전·병진은 기준/변경 양쪽 PnP가 유효한 paired 샘플만의 중앙값 / p95다. XYZ는 유효 공유 mask가 있는 프레임별 중앙값의 분포다. PnP는 전체 27장 분모의 유효 반환 수이며 정확도 지표가 아니다. 시간은 각 조건의 crop→전처리/H2D→모델+decode/D2H→PnP의 GPU 동기화 중앙값 ms다.

| 조건 | 유효 PnP | pose 비교 수 | 대칭 회전 ° 중앙 / p95 | 대칭 정렬 병진 CAD 단위 중앙 / p95 | XYZ CAD 단위 중앙 / p95 | 시간 중앙 ms |
|---|---:|---:|---:|---:|---:|---:|
{table}

일반 SO(3) 회전 및 일반 병진 분포, 공유 픽셀 비율, 최대값과 조건별 실패 프레임은 `full/condition_summary.csv`에 기록했다. `full/stability_distributions.png`는 대칭 회전·병진·XYZ 분포, `full/pnp_valid.png`는 유효 반환 수다. 프레임별 기준/변경 값과 각 시간 구간은 `full/per_frame.csv`, 최대 변화를 보인 조건은 `full/frame_sensitivity.csv`, 실패·큰 변화는 `full/failure_cases.csv`에 있다.

## 관찰 및 한계

- 기준 PnP는 26/27이고 68이 실패했다. scale 0.85 및 x/y 이동 조건은 25/27로 66과 68이 실패했다. 1.20·1.40은 26/27이다. 66은 기준에서 반환됐지만 11개 변화 조건 중 2개에서만 반환됐다. 해당 프레임의 실패 원인은 확인하지 못했다.
- 66의 scale 1.20은 대칭 정렬 회전 132.00°, 병진 1644.94 CAD 단위 차이를 보였고, scale 1.40의 공유-mask front XYZ 중앙 차이는 82.69 CAD 단위였다. 이는 **예측 간 차이**이며 어느 예측이 맞는지는 알 수 없다. 92와 96의 y −10%, 78과 94의 1.40배도 대칭 회전 변화가 컸다. 세부 값은 프레임 CSV에 있다.
- scale 1.20의 66에서 비항등 대칭이 회전 기준으로 선택됐다. 나머지 비교의 대칭 인덱스 분포는 요약 CSV에 있으며, 대칭 고려값이 작아져도 pose 정확도를 판정하지 않는다.
- 조건별 비교 표본은 PnP 반환에 따라 25 또는 26개이며 전체 27장을 분모로 한 안정성 성공률이 아니다. 작은 유효 표본의 p95는 극단값에 민감하다. 촬영 연속성, 동일 pose 및 개발 데이터 재사용으로 독립 표본이 아니다.
- 예측 변화의 원인으로 ROI 잘림, 배경, XYZ 또는 PnP 중 하나를 단정할 근거는 없다. `full/overlays/`의 자홍 CAD는 예측 pose만 표시하고 GT를 포함하지 않는다.

## Smoke·자원·시간

실행 전 RTX 5090의 사용량은 391 MiB/32607 MiB였고 compute process가 없었다. 026·068·094 × 12조건 smoke는 통과했다. 1.00 조건의 예측 PnP/행렬/병진은 기존 저장 예측과 일치했고, 026에서 동일 seed 3회 반복 시 mask·XYZ·PnP가 동일했다. 이 1프레임 반복은 전체 이미지의 알고리즘 안정성을 증명하지 않는다. 공식 대칭 합성 검증도 통과했다. 전체 실행의 PyTorch peak allocated={m['peak_allocated_bytes']} bytes, reserved={m['peak_reserved_bytes']} bytes. 주요 입력 {len(m['input_sha256'])}개 해시는 실행 전후 일치했고 전 파일 전수 불변성 검사는 아니다.

시간은 3회 warmup 뒤 GPU 동기화해 매 회 측정했으며 detector, 이미지 읽기, undistort, overlay, 실행 초기화는 제외한다. PnP 유효와 실패가 섞인 실험 측정치로 실시간 end-to-end latency가 아니다. GT와 무관한 ROI 민감도 진단으로만 사용한다.

## 후속 검증

1. 독립 자료로 개발27과 BB의 calibration K,D·pose·물리 단위·시간 정합을 검증한다. 이 전까지 GT 정확도와 ROI 정책 개선 평가는 NO-GO다.
2. 같은 ROI·같은 저장된 XYZ 대응점으로 PnP seed를 여러 개 바꿔 반복해 solver 변동을 분리하고, 같은 seed의 반복도 여러 프레임에서 확인한다. 특히 66의 유효/실패 전환을 조사한다.
3. 검증된 별도 세션에서 사전 정의한 ROI 변화 조건을 paired 방식으로 평가한다. 본 결과는 배율 선택의 참고 자료이며 독립 검증을 대체하지 않는다.
''';(P/'연구_보고서.md').write_text(report,encoding='utf8')
(P/'README.md').write_text('''# 재실행

기존 결과 보존을 위해 새 빈 폴더에서 코드를 복사해 실행한다. 실행은 GPU와 기존 경로의 이미지·K,D·YOLO 검출 CSV·CAD·공식 가중치가 필요하다. BB 및 GT pose는 입력으로 쓰지 않는다. 기존 환경의 `/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv`를 사용한다.

```bash
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/roi_prediction_stability_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/roi_prediction_stability_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/stability.py" --smoke > "$TASK_NEW/smoke.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/stability.py" > "$TASK_NEW/full.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/summarize.py" > "$TASK_NEW/summary.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/make_report.py"
```

`stability.py`는 compute process가 GPU에 있거나 smoke가 통과하지 않으면 전체 실행을 중단한다. 추론 지표와 시각화의 한계는 `연구_보고서.md`를 확인한다.
''',encoding='utf8')
print('report',len(fail),'failure or sensitivity observations')
