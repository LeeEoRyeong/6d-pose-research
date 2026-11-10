# 재실행

Phase 10 결과 및 RAW·라벨 JPEG가 기존 경로에 있어야 한다. CPU에서 실행하며 새 빈 디렉터리에 결과를 만든다. 데이터 출처는 보고서처럼 가설과 실행 증거를 구분한다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/bb100_calibration_provenance_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/bb100_calibration_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/compare.py" > "$TASK_NEW/run.log" 2>&1
"$TASK_PY" -u -B "$TASK_NEW/jpeg_and_groups.py" > "$TASK_NEW/jpeg.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/report.py"
```

그림의 RAW는 저장된 JPEG 후보이고 센서 Bayer RAW라는 뜻이 아니다. 임의의 보정 파라미터를 학습에 적용하지 않는다.
