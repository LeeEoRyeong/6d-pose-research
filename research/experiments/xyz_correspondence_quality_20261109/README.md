# 재실행

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
