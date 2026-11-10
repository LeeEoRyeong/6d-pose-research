from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent;O=P/'full'
def read(f):
 with f.open(newline='',encoding='utf8') as x:return list(csv.DictReader(x))
M=read(O/'setting_summary.csv');D=read(O/'input_diagnostics.csv');meta=json.loads((O/'summary_meta.json').read_text());cache=json.loads((O/'cache_check.json').read_text());S=read(O/'seed_summary.csv')
configs=[('1.5','150','기존'),('2.0','1000','후보 1'),('3.0','150','후보 2')]
def fmt(x,n=2):return '—' if x=='' else f'{float(x):.{n}f}'
lines=[]
for scope,scope_label in [('baseline','기준 ROI'),('changed_roi','변경 ROI')]:
 for th,it,label in configs:
  r=next(x for x in M if x['scope']==scope and x['threshold_px']==th and x['iterations']==it)
  lines.append(f"| {scope_label} | {label} {th}px/{it} | {r['returned']}/{r['unique_inputs']} | {fmt(r['median_strict_1p5_ratio'],3)} | {fmt(r['median_all_reprojection_px'])} | {fmt(r['median_sym_rotation_vs_original_deg'])} / {fmt(r['p95_sym_rotation_vs_original_deg'])} | {fmt(r['median_sym_translation_vs_original'])} / {fmt(r['p95_sym_translation_vs_original'])} | {fmt(r['median_pnp_ms_per_input'])} |")
rows=[]
for i in [32,66,68,78,92,94,96]:
 r=next(x for x in D if x['frame']==str(i) and x['condition']=='baseline');rows.append(f"| {i} | {r['points']} | {r['original_pnp_valid']} | {r['original_reprojected_inliers_1p5px'] or '—'} | {fmt(r['original_reproj_median_all_px'])} |")
report=f'''# HCCEPose PnP 실패 원인 및 설정 실험

2026-11-09. 이 실험은 **GT 없이 고정 예측 대응점의 PnP 반환과 일관성**을 살폈다. 기존 GT 신뢰성 NO-GO는 유지한다. 새로운 설정이 pose를 더 자주 반환하더라도 GT 정확도나 실제 성능 개선을 뜻하지 않는다. Official EfficientNet-B4·YOLO 및 데이터는 변경하지 않았고 학습하지 않았다.

## 입력, 범위와 smoke

이전 `roi_prediction_stability_20261109`의 개발27 기준 및 ROI 변화를 사용했다. 원본 코드 `HccePose/PnP_solver.py`의 `solve_PnP(pnp_op=2)`는 예측 mask에서 front XYZ와 image 2D 좌표를 추출하고 OpenCV `SOLVEPNP_EPNP` RANSAC(기본 1.5px, 150회)을 실행한다. 그 후 **반환된 pose에 다시 투영한 오차 <1.5px**를 별도로 계산해 inlier로 반환한다. OpenCV가 RANSAC에서 반환한 inlier와 최종 pose의 재투영 기준 inlier는 같지 않을 수 있다.

캐시는 27장 기준 ROI와 이상 7프레임(32, 66, 68, 78, 92, 94, 96)의 11개 변경 ROI, 중복 없이 총 **104개 입력**이다. 32·78은 이전 결과에서 큰 예측 변화가 있어 추가했다. 각 `full/cache/*.npz`에는 ROI xywh, 256 RGB crop, mask·front/back XYZ, 2D grid, PnP 입력 2D–3D 대응점, K, 기존 R/t와 기존 재투영 inlier·오차가 있다. 이 캐시는 실사 이미지와 모델 예측을 포함하는 로컬 연구 산출물이다. `cache_inventory.csv`는 대응점 수·유효성·inlier·재투영 통계를 기록한다. 기존 YOLO 검출 bbox를 재사용했으며 oracle ROI·GT pose는 이용하지 않았다.

개발27 RAW를 기존 K,D,newK=K로 보정한 이미지가 controlled PNG와 픽셀 일치했다. 104개 cache 조건에서 이전 ROI 결과의 mask 픽셀 수·PnP 반환·pose 변화가 재현됐다. 같은 점을 이용한 직접 OpenCV 호출은 기존 solver의 반환 여부·R/t와 최종 재투영 inlier가 일치했다. 별도 smoke 6조건(026,066,068,094 관련, 66의 1점 조건 포함)과 72회 PnP smoke를 통과했다. 공식 체크포인트 및 사용 이미지·카메라·기존 결과 등 {{len(cache['source_sha256'])}}개 주요 파일의 해시는 전후 일치했다. 저장소 모든 파일의 불변성을 검증한 것은 아니다.

첫 전체 캐시는 66/0.85에서 이전 결과대로 대응점이 1개인데, 작성한 검사 코드가 5개 초과를 강제해 중단됐다. 1개를 유효한 **입력 부족 실패 사례**로 기록하도록 실험 사본만 수정하고 smoke부터 다시 실행했다. 실패 로그와 부분 결과는 `full_cache_failed_insufficient_gate/`에 보존했다. 좌표 불일치로 확인된 것은 아니다.

## Seed 안정성

104개 고정 입력에서 기존 설정으로 `cv2.setRNGSeed(0..19)`를 호출한 **2,080회**의 성공 여부와 pose는 각 입력 안에서 모두 동일했다. 설정 grid에서도 1,560개 입력·설정 그룹 각각에서 20개 seed 결과가 동일했다. 1.5px/150회 grid와 seed 단계의 성공·R/t도 전부 일치했다. 따라서 계산상 관측된 seed 간 회전·병진 분산은 0이지만, **이 OpenCV 경로에서 호출자가 바꾼 seed가 서로 독립적인 RANSAC 표본을 만들었다고 입증되지 않았다**. seed 효과가 본질적으로 없다는 결론이나 강건성 증명으로 사용하지 않는다. `seed_summary.csv`와 `seed.csv`에는 일반·대칭 고려 회전, 일반·대칭 정렬 병진 차이가 저장돼 있다.

## 고정 대응점의 RANSAC 설정

기준 ROI 27개와 변경 ROI 77개 각각에 임계값 1·1.5·2·3·5px, 반복 150·500·1000, seed 0–19를 적용해 **31,200회** 실행했다. 각 조건의 20 seed 결과가 같으므로 반환 통계는 중복 20회를 독립 표본으로 세지 않고 **고유 입력 27개 또는 77개**를 분모로 썼다. 아래 재투영은 설정에 관계없이 최종 pose에서 1.5px 미만인 점의 비율이며, 서로 다른 inlier 임계값을 직접 비교할 때 생기는 착시를 줄이기 위한 공통 진단이다. 반환 실패는 재투영 및 pose 차이 중앙값에서 제외했다.

| 범위 | 설정 | 유효 반환 | 최종 1.5px 비율 중앙값 | 전체 점 오차 중앙값의 중앙값 px | 기존 pose 대비 대칭 회전 중앙 / p95 ° | 기존 pose 대비 대칭 병진 중앙 / p95 CAD 단위 | PnP 시간 중앙 ms |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

회전·병진 변화는 기존 1.5px/150회와 비교 조건 양쪽에서 반환된 입력만 paired로 계산했다. 대칭은 공식 `T_reference @ S`를 회전에 대해 최소화하고 같은 S의 이동도 적용한다. 모델 단위는 BOP CAD mm이지만 실제 실사 pose의 물리 mm 정확도는 미검증이다. inlier 반환율·오차 표본 수는 `setting_summary.csv`에 있다. 시간은 4개 CPU worker가 동시에 실행하며 OpenCV `solvePnPRansac` 호출만 잰 값으로 재투영 계산을 포함하지 않는다. 순서·부하에 영향을 받아 단독 처리 latency로 해석하면 안 된다.

**개선 후보는 2px/1000회와 3px/150회 두 개만 탐색 후보로 남긴다.** 같은 개발 입력에서 선택한 것으로 독립 검증이 아니다. 두 설정은 반환 수가 증가하지만, 변경 ROI에선 기존 반환 pose와의 대칭 회전 차이 p95가 각각 61.06°와 80.43°여서 예측 일관성 향상을 확인하지 못했다. 특히 68번 기준 ROI에서 3px/150회는 반환하지만 201점 중 최종 1.5px 이내는 2점이고 전체 점 재투영 오차 중앙값이 약 **69.63px**이다. 2px/1000회도 최종 1.5px 이내 4점, 전체 중앙값 27.64px이다. 두 후보 모두 **운영 채택 NO-GO**다. 재투영 inlier 비율 역시 XYZ가 틀리면 정확도 GT가 될 수 없다.

## 프레임별 원인 분해

아래 inlier는 원본 solver의 **최종 pose에서 재계산한 1.5px 이내 점 수**다. 실패 시 재투영 오차는 결측이다.

| 기준 ROI 프레임 | 대응점 수 | PnP 반환 | 최종 1.5px 이내 점 수 | 전체 대응점 재투영 중앙값 px |
|---:|---:|---|---:|---:|
{chr(10).join(rows)}

- **A, crop 변화:** ROI 변경에 따른 256×256 crop의 평균 절대 픽셀값 차이를 `input_diagnostics.csv`에 기록했다. 66의 0.85/1.20은 기준 crop 대비 각각 30.4/28.5(0–255)다. 이는 crop 좌표가 달라진 이미지 배열의 차이이며 물체가 잘렸다는 증명은 아니다.
- **B, XYZ 변화:** 같은 이미지 좌표의 두 예측 mask 교집합에 대해 nearest-neighbor로 맞춘 front XYZ 차이를 계산했다. 결과가 이전 ROI 실험 CSV와 전부 일치했다. 66의 1.20에서는 중앙 11.50, 1.40에서는 82.69 CAD 단위로 변화했다. 이는 예측끼리의 차이이지 정답 XYZ 오차가 아니다.
- **C, 대응점:** 66 기준 예측 mask는 203점을 만들지만 최종 pose에 대한 1.5px 재투영 점은 **0점**, 전체 재투영 중앙값은 26.50px다. OpenCV가 RANSAC에서 반환한 raw inlier 수는 5개여서 최종 pose 재투영 0개와 다르다. 그 이유를 단정하지 않으며 반환값만으로 pose를 신뢰하면 안 된다는 실측 근거다. 66/0.85의 예측 mask는 1점이라 solver 최소 요건을 충족하지 못한다. 68 기준은 201점이 있어도 1.5px/150회에서 실패했다. 점 개수만으로 품질을 판단할 수 없다.
- **D, RANSAC 민감도:** seed 설정 값 변화에 따른 출력 변이는 관측되지 않았지만 seed가 실제 표본을 제어하는지는 불명이다. 임계값·반복 변경에는 일부 반환 여부와 pose가 달라졌다. 66/0.85처럼 입력이 1개면 RANSAC 설정만으로 복구할 수 없다.
- **E, 대칭:** 순수 SO(3)와 공식 이산 대칭을 고려한 변화량을 별도 저장했다. 대칭은 물체의 동등 표현 차이를 분리하는 도구이고, 이 실험에는 어떤 표현이 GT에 맞는지 판정할 기준이 없다.

프레임별 전체 2D–3D 입력과 수치는 `cache_inventory.csv`, `input_diagnostics.csv`, `seed_summary.csv`, `grid_baseline.csv`, `grid_roi.csv`에 있다. 66의 ROI·예측 mask·정규화 XYZ를 비교한 `frame66_crop_mask_xyz.png`는 예측 입력의 변화를 보여줄 뿐 visible GT가 아니다. 원인 중 실제 가림, 잘못된 ROI, XYZ 정확도 및 pose 정확도는 별도 검증 없이 확정할 수 없다.

## 완료·미실행 및 다음 단계

완료: 사전 점검, smoke, 104개 입력 캐시, 고정 대응점 seed 반복, 기준·이상 ROI 설정 grid, 재투영 분석과 시각화. 미실행: 모델 학습, BB 분석, GT 기반 pose 정확도 또는 독립 세션 검증, 실시간 latency benchmark. 20개 seed는 독립 RANSAC 실행 표본이라는 근거가 없으므로 seed 분산 평가의 목적은 **부분 충족**이다.

다음으로 OpenCV가 실제로 서로 다른 RANSAC 표본을 사용하도록 허용하는 별도 API/샘플링 구현을 독립 검증하고, 대응점 분포·퇴화·최종 inlier 수를 실패로 판정하는 품질 gate를 독립 데이터에서 보정한다. 이후 camera K,D, GT pose 및 물리 단위를 검증하고 별도 촬영 세션에서 정책을 평가한다. 이 실험의 높은 PnP 반환율만으로는 완화된 설정을 배포하지 않는다.
'''.replace("{len(cache['source_sha256'])}",str(len(cache['source_sha256'])))
(P/'연구_보고서.md').write_text(report,encoding='utf8')
config={'frames_baseline':27,'abnormal_frames':[32,66,68,78,92,94,96],'changed_roi_conditions_per_abnormal':11,'cached_inputs':104,'seed':list(range(20)),'reprojection_px':[1.0,1.5,2.0,3.0,5.0],'iterations':[150,500,1000],'solver':'OpenCV SOLVEPNP_EPNP solvePnPRansac; original op2','workers':4,'uses_ground_truth':False,'training':False,'input_hashes':'full/cache_check.json'};(P/'실험_설정.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
(P/'README.md').write_text('''# 재실행

실사 영상 및 체크포인트가 있는 기존 연구 환경이 필요하다. 이전 결과를 덮어쓰지 않으려면 새 디렉터리에서 실행한다. smoke 통과 및 GPU 여유 확인 뒤 전체 캐시를 생성하고 CPU 4 worker로 PnP를 계산한다. CPU 반복은 수 분이 걸릴 수 있다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/pnp_stability_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/pnp_stability_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/cache_inputs.py" --smoke > "$TASK_NEW/smoke_cache.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/run_pnp.py" --smoke > "$TASK_NEW/smoke_pnp.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/cache_inputs.py" > "$TASK_NEW/full_cache.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/run_pnp.py" --phase seed > "$TASK_NEW/seed.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/run_pnp.py" --phase grid_baseline > "$TASK_NEW/grid_baseline.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/run_pnp.py" --phase grid_roi > "$TASK_NEW/grid_roi.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/summarize.py" > "$TASK_NEW/summarize.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/write_report.py"
```

모든 결과는 GT 없이 예측 입력 일관성을 분석한다. 상세 범위와 미실행 사항은 `연구_보고서.md` 참조.
''',encoding='utf8');print('REPORT READY')
