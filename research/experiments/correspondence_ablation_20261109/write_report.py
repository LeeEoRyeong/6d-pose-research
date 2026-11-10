import json,csv,pathlib
P=pathlib.Path(__file__).resolve().parent;O=P/'full'
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
M=read(O/'method_summary.csv');PAIR=read(O/'paired_vs_all.csv');RAN=read(O/'matched_random_comparison.csv');ORDER=read(O/'order_summary.csv');prec=read(O/'dedup_precision_indices.csv')
fmt=lambda v,n=3:'—' if v=='' else f'{float(v):.{n}f}'
items=['A_all','B_dedup3','B_dedup5','B_dedup7','C_grid4','C_grid8','C_grid16','D_dedup5_grid4','D_dedup5_grid8','D_dedup5_grid16'];lines=[]
for k in items:
 a=next(x for x in M if x['scope']=='all' and x['method']==k and x['kind']=='deterministic');b=next(x for x in PAIR if x['scope']=='all' and x['method']==k);lines.append(f"| {k} | {fmt(a['median_selected_count'],0)} | {a['cases_with_any_success']}/104 | +{b['pnp_gain_vs_A']} / −{b['pnp_loss_vs_A']} | {b['both_pnp_returned']} | {fmt(b['median_full_error_delta_vs_A_px'])} | {fmt(a['median_pnp_ms'],2)} |")
matched=[]
for k in ['B_dedup5','C_grid4','C_grid8','C_grid16','D_dedup5_grid4','D_dedup5_grid8','D_dedup5_grid16']:
 x=next(r for r in RAN if r['scope']=='all' and r['target']==k);matched.append(f"| {k} | {x['cases_both_all5_success']} | {fmt(x['median_full_error_delta_selected_minus_random_mean_px'])} | {fmt(x['median_full_inlier_ratio_delta_selected_minus_random_mean'],4)} |")
o=next(x for x in ORDER if x['method']=='A_all')
report=f'''# Phase 9: 2D–3D 대응점 선택 Ablation

2026-11-09. 기존 104개 고정 예측 입력에서 대응점 선택만 바꿔 PnP의 반환과 내부 재투영 일관성을 비교했다. **중복 제거와 성긴 격자 선택에서 기존 전체 점 대비 일관된 개선 근거를 찾지 못했다. 동일 데이터에서 격자·반올림 파라미터를 추가 탐색하는 것은 중단을 권고한다.** GT pose를 사용하지 않았으며 실제 pose 정확도와 GT 신뢰성에 대한 NO-GO는 유지한다.

## 실험과 검증

Phase 8이 기록한 캐시·입력 CSV 해시를 재검증했고, 전체 점으로 기존 OpenCV EPNP RANSAC 1.5px/150회, `42+frame` seed를 적용할 때 **104개 입력 모두의 반환 여부, 성공 시 R/t와 최종 재투영 inlier 수**가 기존 cache와 일치했다. 32·66·68 기준 및 66/0.85배의 4조건, 총 300회 smoke 후 전체 104조건에서 **7,800회** 추론했다. 변경된 XYZ 값은 없고 2D–3D 인덱스만 선택했다. 사용한 캐시와 관련 입력의 Phase 8 해시는 실험 전후 일치했다. 기존 모델·데이터는 수정하지 않았고 학습하지 않았다.

104조건은 개발27 기준 27개와 선별된 이상 프레임의 변경 ROI 77개다. 동일 프레임의 여러 ROI가 들어 있으므로 104개의 독립 영상이나 모집단 대표 표본이 아니다. 기존 GT pose나 oracle ROI는 이용하지 않았다.

- **A:** 모든 대응점을 원래 순서로 사용한다.
- **B:** XYZ를 CAD 좌표에서 소수 3·5·7자리로 반올림해 중복 그룹을 만들고, 각 그룹에서 **원래 배열 인덱스가 가장 작은 점**을 대표로 택한다. 대표점은 다시 원래 순서대로 정렬한다. 양자화된 예측점의 중복이 잘못된 예측을 뜻하지는 않는다.
- **C:** 기존 padded square ROI의 이미지 좌표를 4×4·8×8·16×16 격자로 나누고, 각 비어 있지 않은 칸에서 **칸 중심에 가장 가까운 점 한 개**를 택한다. 거리가 같으면 원래 인덱스가 작은 점을 택하고 결과는 원래 순서로 정렬한다. 즉 각 칸의 동일 개수라는 뜻이지 모든 격자에 점이 있다는 뜻은 아니다.
- **D:** B의 5자리 중복 제거 후 C를 적용한다.
- **E:** B5 및 C/D 각 격자 방법과 **같은 최종 점 수**의 원본 대응점을 복원 없이 무작위 선택한다. 각 입력·대상 방법별 별도 결정적 NumPy 난수 스트림 5개를 사용하고 선택한 인덱스는 원래 순서로 정렬한다. 반복 추출 간 점이 겹칠 수 있으며 서로 독립 영상 5장을 뜻하지 않는다.

원래 solver는 대응점이 **4개 이하**면 실행하지 않고 실패를 반환한다. 모든 방법에 K와 입력 좌표계, EPNP, RANSAC 임계값/횟수를 고정했다. 각 방법의 같은 선택 집합에 대해 원래 순서와 별도 난수 스트림으로 만든 순열 3개도 비교했다. OpenCV 내부 RNG가 외부 `cv2.setRNGSeed`를 실제로 반영하는지는 앞선 실험에서 확인되지 않았으므로, 이 순서 실험은 서로 다른 입력 순서를 통제한 실험이며 독립적인 RANSAC 난수 상태를 보장한다고 주장하지 않는다.

## 기존 전체 점과 paired 비교

아래 표의 ±는 **같은 입력에서 A 대비 PnP 반환이 새로 발생 / 사라진 건수**다. 재투영 Δ는 A와 해당 방법이 모두 유효 pose를 반환한 동일 입력에서, **원래 전체 대응점**에 대한 오차 중앙값 `(방법−A)`의 중앙값이다. 음수가 실제 정확도 향상을 뜻하지 않는다. PnP 시간은 4개 CPU worker가 동시에 실행한 OpenCV 호출 시간으로 선택·재투영 계산은 제외한다.

| 방법 | 선택 점 수 중앙값 | 반환 / 104 | 반환 변화 + / − | 양쪽 반환 수 | 전체 점 오차 Δ px 중앙값 | PnP 시간 중앙 ms |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

전체 점 A는 83/104에서 반환됐다. B5는 81/104로 새 반환 0·기존 반환 손실 2건이고, 양쪽 반환 81건의 전체 점 재투영 오차 Δ 중앙값은 **−0.005px**이다. C16은 72/104, D16은 77/104이며 양쪽 반환 입력에서 전체 점 오차 Δ 중앙값은 각각 **+0.119px**, **+0.131px**이다. 성공 집합이 다르므로 조건별 성공 사례의 단순 중앙값만으로 우열을 판단하지 않는다. 세부 범위별 비교와 inlier·대칭 pose·병진 변화는 `method_summary.csv`, `paired_vs_all.csv`, `per_trial.csv`에 있다. 회전은 순수 SO(3)와 공식 대칭 고려 결과를 별도 저장한다. 대칭 병진은 최소 회전 대칭의 이동까지 반영한다. CAD 단위는 BOP mm이지만 실사 물리 단위 정확도는 미검증이다.

## 같은 점 수 무작위 대조

각 행은 선택 방법과 동일 점 수의 E 대조군을 비교한다. 점 수 감소 효과와 선택 규칙 효과를 분리하기 위한 것으로, **해당 결정적 선택과 5개 무작위 시도가 모두 반환된 입력만** 사용했다. 재투영 Δ는 `(선택 방법의 원래 전체 점 오차 중앙값 − 5개 무작위 시도의 원래 전체 점 오차 중앙값 평균)`을 프레임·조건별로 구한 뒤 중앙값을 취했다. 공통 1.5px inlier 비율도 같은 방식으로 비교했다. 반환 실패가 있는 입력은 이 paired 수치에서 제외되므로 표본 선택 편향이 남는다.

| 대상 방법 | 6개 시도 모두 반환된 입력 | 전체 점 오차 Δ 중앙값 px | 전체 점 1.5px inlier 비율 Δ 중앙값 |
|---|---:|---:|---:|
{chr(10).join(matched)}

B5의 무작위 대비 Δ는 **−0.008px(79개 입력)**, C16은 **−0.025px(72개)**, D16은 **−0.031px(70개)**다. 이 조건부 차이와 제한된 집합만으로 선택 규칙의 일반적인 개선을 주장할 수 없다. 무작위 5회 반환 여부와 각 시도의 전체·선택 점 오차는 `per_trial.csv`, 케이스당 무작위 평균은 `per_case_summary.csv`에 있다.

선택 점만의 오차는 실제보다 좋아 보일 수 있다. 예를 들어 32 기준의 C4는 선택 7점 오차 중앙값 **0.82px**지만 같은 pose를 원래 1,941점에 적용하면 중앙값은 **2.28px**다. 68 기준의 C4는 pose를 반환했어도 원래 201점의 오차 중앙값이 **463.80px**, 전체 점의 1.5px inlier가 0개였다. 특정 프레임의 반환만으로 결과를 개선됐다고 판단하지 않는다.

## 순서와 정밀도

A의 동일 점 집합을 순열 3개로 바꾸면 **{o['success_changed_any_of_3']}/104 입력**에서 기존 순서와 반환 여부가 한 번 이상 달랐다. 기존과 순열 중 양쪽 pose가 하나 이상 반환된 {o['valid_pose_comparison_cases']}개 입력의 최대 대칭 회전 변화는 중앙값 {fmt(o['max_sym_rotation_median_deg'],2)}°, p95 {fmt(o['max_sym_rotation_p95_deg'],2)}°였다. 이는 매 입력의 세 순열 중 최대값 분포이며 GT 오차가 아니다. A의 반환 여부가 달라진 사례는 66 기준·1.20·x+10%·y+5%·y+10%, 68 x−10%·y+10%다. `order_summary.csv`와 `per_trial.csv`에 다른 방법도 기록했다.

반올림 3·5·7자리는 **104/104 입력에서 선택 점 수뿐 아니라 대표 인덱스 집합까지 동일**했고, 결정적 PnP 결과도 일치했다. 이 캐시의 이산 XYZ 분해능에서 본 세 정밀도는 중복 판단 결과를 바꾸지 않았다는 뜻이며 다른 모델이나 데이터에는 확장할 수 없다. `dedup_precision_indices.csv` 참조.

## 66·68 및 연구 판단

66 기준에서 A는 203점으로 pose를 반환했지만 원래 전체 점의 최종 1.5px inlier는 0개, 재투영 중앙값은 26.50px이다. B는 중복 제거 후 21점이며 반환하지 않았다. 68 기준은 A 201점과 B 19점에서 반환하지 않았다. C4의 반환도 위의 전체 점 재투영 검사에서 낮은 일관성을 보였다. 한편 이 검사는 입력 예측점 사이의 내부 관계를 보는 것이므로 낮은 오차도 실제 GT pose를 보증하지 않는다.

현재 실험 범위에서는 **더 많은 격자·반올림 파라미터 탐색보다** 예측 mask와 XYZ가 어떤 실제 표면을 가리키는지 독립적으로 검증하고, 반환 후 전체 대응점 재투영 품질을 별도로 보고하는 것이 다음 우선순위다. 운영 품질 gate의 임계값과 XYZ·PnP 수정 효과는 독립 데이터와 GT 확인 후 검증해야 한다. 입력 순서에 따른 차이를 조사하려면 점 집합과 순서를 고정한 기준에서 실제 독립 RANSAC 표본 제어가 가능한 구현을 별도 시험한다.

## 산출물과 재현

`full/per_trial.csv`는 7,800회 전부의 선택 점 수, R/t, raw·최종 inlier, 선택/전체 점 재투영, 대칭/일반 pose 변화 및 시간을 보존한다. `full/ablation_summary.png`는 반환과 matched random 대조를 보여준다. 설정과 재실행 명령은 `실험_설정.json`, `README.md`를 참조한다. 추가 모델 학습과 GT 기반 정확도 평가는 미실행이다.
''';(P/'연구_보고서.md').write_text(report,encoding='utf8')
config={'data':'Phase 7/8 validated 104 caches','scope':'27 baseline + 77 anomaly ROI; not independent','threshold_px':1.5,'iterations':150,'solver':'OpenCV EPNP','cv_seed':'42+frame, external RNG effect not established','dedup_decimals':[3,5,7],'dedup_representative':'first original index','grid_sizes':[4,8,16],'grid_selection':'nearest occupied cell centre, lowest index tie','random_matched_repeats':5,'sampling_randomness':'NumPy generator hashed per frame/condition/target/replicate','order_shuffles_per_method':3,'workers':4,'no_ground_truth':True,'training':False};(P/'실험_설정.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
(P/'README.md').write_text('''# 재실행

기존 Phase 7의 104개 캐시와 Phase 8 manifest가 필요하며 원본을 수정하지 않는다. 아래는 새 디렉터리에 코드와 결과를 생성한다. CPU 4 worker를 사용하며 GPU 및 GT pose는 사용하지 않는다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/correspondence_ablation_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/correspondence_ablation_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/ablate.py" --smoke > "$TASK_NEW/smoke.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/ablate.py" --workers 4 > "$TASK_NEW/full.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/summarize.py" > "$TASK_NEW/summarize.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/paired_baseline.py"
"$TASK_PY" -B "$TASK_NEW/verify_precision.py"
"$TASK_PY" -B "$TASK_NEW/write_report.py"
```

실험 결과는 GT 정확도를 뜻하지 않으며 보고서의 분모·성공 조건 및 독립성 제한을 확인한다.
''',encoding='utf8');print('REPORT READY')
