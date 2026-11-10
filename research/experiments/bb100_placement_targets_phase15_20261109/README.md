# Phase 15 재현 및 검수

상세 판정과 제한은 `연구_보고서.md`를 참조한다. 타깃은 전부 미검수 진단용이다.

Windows에서는 `BB100_placement_review_windows.zip` 전체를 복사해 해제한 뒤 `index.html`을 연다. 개별 HTML만 복사하면 이미지가 보이지 않는다. CSV 내보내기를 자주 사용한다.

CPU 재현 시 기존 파일을 덮어쓰지 않도록 `build.py`, `review_template.html`, `targets.py`, `test_targets.py`, `finalize.py`를 새 빈 디렉터리에 복사한다. 지정된 서버 원본 경로와 기존 Python 환경이 필요하다.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B build.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B test_targets.py
```

`targets.DiagnosticTargets.training_sample`은 의도적으로 실패한다. 대칭 canonicalization과 visible supervision은 아직 검증되지 않았으므로 학습용 데이터 로더로 사용하지 않는다.
