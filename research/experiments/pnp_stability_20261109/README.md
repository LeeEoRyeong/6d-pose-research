# 재실행

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
