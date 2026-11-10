import csv,json,pathlib,collections,numpy as np
P=pathlib.Path(__file__).resolve().parent;O=P/'full'
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
r=read(O/'quality_per_case.csv');g=read(O/'gate_summary.csv');meta=json.loads((O/'summary.json').read_text());manifest=json.loads((O/'manifest.json').read_text());yes=[x for x in r if x['pnp_valid']=='True'];no=[x for x in r if x['pnp_valid']=='False']
def median(key,z):
 a=[float(x[key]) for x in z if x[key]!=''];return float(np.median(a)) if a else None
labels={'none':'기존 PnP 반환만','points32':'대응점 ≥32','inliers20':'최종 inlier ≥20','ratio10pct':'최종 inlier 비율 ≥10%','median3px':'전체 점 오차 중앙값 ≤3px','spatial_rank':'2D rank≥2 및 3D rank≥2','combined':'위 조건 모두'}
gtable='\n'.join(f"| {labels[x['gate']]} | {x['accepted']}/104 | {x['rejected_among_returned']}/83 |" for x in g)
ctable=[]
for f in [32,66,68,78,92,94,96]:
 x=next(q for q in r if int(q['frame'])==f and q['condition']=='baseline');ctable.append(f"| {f} | {x['point_count']} | {x['pnp_valid']} | {x['raw_ransac_inliers'] or '—'} / {x['final_inliers'] or '—'} | {float(x['duplicate_3d_fraction_round_1e5']):.1%} | {float(x['cov3_planarity_ratio_e3_e1']):.6f} | {float(x['reproj_median_all_px']):.2f} |" if x['reproj_median_all_px'] else f"| {f} | {x['point_count']} | {x['pnp_valid']} | — | {float(x['duplicate_3d_fraction_round_1e5']):.1%} | {float(x['cov3_planarity_ratio_e3_e1']):.6f} | — |")
report=f'''# Phase 8: XYZ 대응점 품질 및 PnP 실패 진단

2026-11-09. GT 없이 기존 예측의 내부 일관성과 기하를 조사했다. **PnP 실패 21건 중 20건은 solver 최소 조건인 4점보다 많은 대응점이 있었고, 반환된 83건 중 7건은 최종 1.5px 재투영 inlier가 20점 미만이었다.** 점 개수나 PnP 반환만으로는 대응점의 재투영 일관성을 보증할 수 없다. GT 신뢰성 NO-GO는 유지하며 어떤 pose가 실제로 정확한지는 판정하지 않는다.

## 입력·방법과 무결성

Phase 7의 **104개 캐시**를 재사용했다: 개발27의 기준 ROI 27건과 이상 7프레임(32·66·68·78·92·94·96)의 변경 ROI 77건. 이 104건은 독립된 104장도 무작위 표본도 아니며 이상 프레임이 과대표집됐다. 실사·모델의 GT pose를 읽거나 재학습하지 않았다. bbox는 YOLO 예측에 기반한다.

4조건 smoke(32 기준, 66 기준·0.85배, 68 기준) 이후 전체를 실행했다. 104개 NPZ의 mask에 의한 XYZ·2D 점 추출, 유한성, bbox, 이전 mask 픽셀 수를 검사했고, 원래 `42+frame` seed와 1.5px/150회의 OpenCV EPNP를 다시 실행해 PnP 반환, R/t, raw inlier 개수 및 최종 재투영 inlier 인덱스가 캐시와 일치했다. XYZ 교집합 중앙값도 이전 ROI 결과와 일치했다. 캐시와 입력 CSV **{len(manifest['source_sha256'])}개 파일**의 해시는 이번 분석 전후 일치했다. 이전 캐시 생성 시 NPZ 자체의 해시는 기록되지 않았으므로 과거부터의 바이트 불변성을 증명하는 것은 아니다. 원본 코드는 수정하지 않았다.

## 대응점 공간 및 기하

2D 공분산은 ROI 폭·높이로 정규화한 이미지 좌표에 대해, 3D 공분산은 모델 예측 XYZ의 CAD 좌표에 대해 계산했다. 모두 중심을 뺀 뒤 `XᵀX/N`이다. 고유값은 내림차순, 유효 rank는 최대 고유값의 10⁻⁶ 초과 개수, 조건수는 최대/최소이며 최소값이 최대의 10⁻¹² 이하이면 미정으로 표시한다. 2D convex hull은 ROI 면적으로 정규화했다. 3D 범위는 축별 max−min, 중복은 좌표를 CAD 단위 소수 5자리로 반올림한 동일 XYZ의 비율이다. HCCE의 이산 양자화 때문에 중복 자체가 오류는 아니다. BOP CAD 단위는 mm이나 실사에서의 물리적 mm 대응 정확도는 미검증이다.

| 관측 지표 | PnP 반환 83건 | PnP 실패 21건 |
|---|---:|---:|
| 대응점 수 중앙값 | {median('point_count',yes):.0f} | {median('point_count',no):.0f} |
| ROI 대비 2D hull 면적 중앙값 | {median('cov2_hull_area_fraction',yes):.4f} | {median('cov2_hull_area_fraction',no):.4f} |
| 중복 예측 XYZ 비율 중앙값 | {median('duplicate_3d_fraction_round_1e5',yes):.1%} | {median('duplicate_3d_fraction_round_1e5',no):.1%} |
| 3D 최소/최대 공분산 고유값 비율 중앙값 | {median('cov3_planarity_ratio_e3_e1',yes):.6f} | {median('cov3_planarity_ratio_e3_e1',no):.6f} |

관측은 선택된 개발 입력의 기술 통계이며 실패 원인에 대한 독립적인 통계 검증이 아니다. 실패 21건 중 20건이 4점 초과, 19건이 100점 이상이었다. 실패 21건 중 20건은 2D rank 2, 19건은 3D rank 3이므로 **형식적인 full rank조차 성공을 보장하지 않는다**. 작은 세 번째 고유값은 평면적 가시 표면에서도 자연스럽게 생긴다. 따라서 rank 2나 큰 조건수만으로 수치적 실패 또는 잘못된 XYZ로 확정하지 않는다. 공분산은 비선형 PnP의 완전한 condition number도 아니다.

## 우선 프레임의 기준 ROI

중복 비율은 반올림 기준의 예측 XYZ에 대한 것이다. raw inlier는 OpenCV RANSAC 반환 인덱스, 최종 inlier는 **반환 pose를 모든 점에 다시 투영해 1.5px 미만인 인덱스**다. 실패하여 pose가 없는 경우 raw/final과 재투영 오차는 0이 아닌 결측으로 표기한다.

| 프레임 | 점 수 | PnP 반환 | raw / 최종 inlier | 중복 XYZ | 3D λ3/λ1 | 전체 점 재투영 중앙값 px |
|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(ctable)}

66 기준은 203점과 3D rank 3을 가졌지만 XYZ의 약 89.7%가 중복되고 OpenCV raw inlier 5점과 달리 최종 재투영 1.5px inlier는 0점이다. 68 기준도 201점 중 약 90.5%가 중복되고 PnP가 실패했다. 이러한 관측은 고유 대응점 부족·공간 분포·불일치가 **가능한 설명**임을 보여주지만, 이 실험만으로 각각의 인과 기여도를 분리할 수는 없다. 특히 raw와 최종 inlier 차이는 확인된 결과이며 그 내부 알고리즘 원인은 확인하지 않았다.

## ROI에 따른 예측 XYZ 변화

기준 ROI에서 비교 이미지 좌표를 생성하고 다른 ROI의 XYZ/mask를 nearest-neighbor로 맞춘 뒤 **양쪽 예측 mask의 교집합**에서 XYZ 유클리드 차이의 중앙값과 p95를 계산했다. 이는 실제 동일 표면점이나 GT와의 차이가 아니다. 경계, 양자화·재표본화, mask 불일치 및 교집합이 적은 경우에 영향을 받는다. 모든 케이스의 값·공유 픽셀 수는 `quality_per_case.csv`, 공간 delta 배열은 `spatial/`, 예시는 `visualizations/*_xyz_delta.png`에 있다.

- 66 / 1.20배: 공유 57픽셀, 중앙값 **11.50**, p95 **88.08** CAD 단위.
- 68 / 1.40배: 공유 11픽셀, 중앙값 **68.14**, p95 **68.16**. 픽셀 수가 작아 전체 물체로 일반화할 수 없다.
- 32 / 1.40배: 공유 1,405픽셀, 중앙값 **79.30**, p95 **122.50**. 예측 출력의 변화이며 실제 XYZ 오류라고 부를 수 없다.

기준 crop grid 대비 예측 mask 픽셀 수가 절반 미만인 조건은 104건 중 6건, 4점 이하인 조건은 1건이다. 서로 다른 ROI의 128×128 출력 grid에서 픽셀 수를 비교한 것이므로 실제 영상상 가시면적이나 visible GT의 축소율이 아니다. 66/0.85의 단일 예측 픽셀은 입력 개수 부족에 해당한다.

## 진단용 품질 gate

기존 PnP 1.5px/150회 결과를 변경하지 않고 반환 pose를 **보류**하는 예시다. 32점·20점·10%·3px 및 rank 기준은 탐색용 가설이다. 모든 gate는 PnP가 반환되지 않으면 보류한다. 아래 반환값 중 보류 수는 GT 오류 검출 수가 아니다.

| 가상 gate | 전체 104건 중 통과 | 기존 PnP 반환 83건 중 보류 |
|---|---:|---:|
{gtable}

복합 gate가 66/104만 통과하고 반환 17건을 보류했지만, 검증된 정답이 없어 이 중 실제로 정확한 pose를 얼마나 버렸는지 또는 부정확한 pose를 얼마나 걸렀는지 알 수 없다. 즉 precision/recall·정확도 향상이나 실운영 기준을 주장하지 않는다. 평면적 물체를 무조건 거부하지 않도록 3D rank 2는 허용했으며, 실제 gate는 독립적인 검증 데이터와 다양한 실제 가림 조건으로 보정해야 한다. `gate_summary.csv`, `gate_counts.png` 참조.

## 질문에 대한 답과 우선순위

1. **점 개수 부족만으로 설명되는가?** 아니다. 21건 중 20건이 solver 최소 점 수를 넘었다. 다만 66/0.85처럼 1점으로 확실히 부족한 사례도 있다.
2. **충분한데도 왜 실패하는가?** 66·68에서 중복 XYZ가 약 90%이고 편평한 예측 분포가 보인다. 그러나 충분한 수의 고유한 정답 대응점이라는 뜻은 아니며, 가림·ROI·양자화·예측 오류·PnP 내부 표본 중 정확한 인과 원인은 미확정이다.
3. **ROI가 XYZ에 미치는 영향은?** 선택 조건에서 위와 같은 큰 예측 변화가 관찰된다. mask 교집합 밖은 비교할 수 없고, 예측 변화가 실제 정답 오차를 뜻하지 않는다.
4. **PnP 반환이 놓치는 위험은?** 반환 83건 중 7건이 최종 1.5px inlier 20점 미만이다. 66 기준처럼 raw 5점·최종 0점인 사례도 있다. 재투영 일관성도 실제 pose 정확도의 보증은 아니다.
5. **모델과 PnP 중 무엇이 우선인가?** 이 선택 데이터에서는 mask 소실과 XYZ·대응점 불일치가 실제로 관측돼, 우선 **검출 ROI·mask·XYZ 입력 품질을 검증하고 PnP 반환을 보류하는 진단**이 타당하다. PnP 설정 탐색과 seed 문제도 남아 있으나 단독 설정 완화로 1점 입력을 복구할 수 없다. 어느 쪽을 학습·수정하면 실제 성능이 개선되는지는 GT와 독립 세션 없이는 결론낼 수 없다.

다음 실험은 독립 세션의 calibration·pose·visible 기준 검증, 동일 cache로 중복 축소·대응점 공간 분포에 대한 통제된 ablation, 독립 RANSAC 샘플 제어 검증 순서로 제안한다. 실제 성능 및 운영 gate 검증은 NO-GO다.
''';(P/'연구_보고서.md').write_text(report,encoding='utf8')
config={'input_cases':104,'pnp':'OpenCV EPNP 1.5px/150, seed 42+frame','rank_rule':'eigenvalue > max_eigenvalue*1e-6','condition_undefined':'min <= max*1e-12 or max <= 1e-20','duplicate_rule':'round XYZ and UV to 1e-5 units','gate_hypotheses':{'min_points':32,'min_final_inliers':20,'min_final_inlier_ratio':.1,'max_all_point_median_reprojection_px':3,'min_2d_rank':2,'min_3d_rank':2},'ground_truth_used':False,'training':False,'selection_bias':'27 baseline and 77 selected anomalous frames/conditions'};(P/'실험_설정.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(P/'README.md').write_text('''# 재실행

기존 Phase 7 캐시와 연구 환경이 필요하며 GT pose와 GPU는 사용하지 않는다. 새 빈 디렉터리에서 실행해 이전 결과를 보존한다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/xyz_correspondence_quality_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/xyz_quality_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/analyze.py" --smoke > "$TASK_NEW/smoke.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/analyze.py" > "$TASK_NEW/full.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/summarize.py" > "$TASK_NEW/summarize.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/write_report.py"
```

실사 이미지와 예측을 포함한 NPZ·시각화는 연구 서버에만 보관한다. 데이터·GT·카메라 불확실성과 gate 한계는 `연구_보고서.md`를 확인한다.
''',encoding='utf8');print('REPORT READY')
