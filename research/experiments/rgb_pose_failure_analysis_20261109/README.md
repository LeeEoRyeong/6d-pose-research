# 분석 재실행

기존 결과와 입력 데이터는 보존해야 하므로 새 빈 디렉터리에서 실행한다. Python 환경과 두 이전 실험 및 GPU가 필요하다. 다른 프로세스가 GPU를 점유하면 `analyze.py`는 시작을 중단한다. BB failure contact-sheet 육안 관찰 기록은 `finalize.py`에 남아 있으며 가림 GT가 아니다.

```bash
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/rgb_pose_failure_analysis_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/rgb_pose_failure_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC/analyze.py" "$TASK_SRC/finalize.py" "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/analyze.py" > "$TASK_NEW/run.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/finalize.py"
```

출력: `연구_보고서.md`, `frame_comparison.csv`, `roi_comparison.csv`, `bb100_failure_analysis.csv`, `failure_statistics.json`, `overlays/`. 좌표·마스크·정확도 제한은 보고서를 참고한다.
