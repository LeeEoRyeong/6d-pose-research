# 재실행

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
