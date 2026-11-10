# Phase 16A-1

Windows 오프라인 검수는 `BB100_visible_mask_review_windows.zip` 전체를 복사해 압축 해제하고 `index.html`을 연다. 모든 결과는 GT가 아닌 후보이며 상세 해석은 `연구_보고서.md` 참조.

재현은 `generate.py`, `template.html`, `package.py`, `topology.py`를 새로운 빈 디렉터리에 복사해 실행한다. 기존 실험 및 SAM 가상환경·체크포인트 경로가 필요하다. 첫 명령은 GPU에서 6장 추론하며 사용 중인 다른 GPU compute 작업이 있으면 시작하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/bb_sam_candidates_20261108/.venv/bin/python -B generate.py
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B package.py
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B topology.py
```

`package.py` 및 `topology.py`는 기존 결과 덮어쓰기를 피하도록 새 디렉터리에서 실행해야 한다. `topology.py`의 브라우저 테스트 PASS 기록은 별도 브라우저 테스트가 수행됐을 때만 유효하다.
