# BB100 GT 및 카메라 감사 재현

원본 데이터 경로가 있는 기존 연구 서버와 Python/OpenCV 환경이 필요하다. 새 빈 디렉터리를 만들고 코드를 복사해 실행하면 기존 결과를 덮어쓰지 않는다. 처리된 오버레이는 카메라 보정과 pose의 정답 인증이 아니다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/bb100_gt_camera_verification_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/bb100_audit_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/audit.py" > "$TASK_NEW/audit.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/verify_overlay.py"
"$TASK_PY" -B "$TASK_NEW/report.py"
```

프레임 검수표, 오버레이, 그룹과 임시 split 및 미확인 사유는 `연구_보고서.md`를 참조한다.
