# 재실행

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
