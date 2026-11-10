# Phase 15B 실행 안내

전체 결과와 한계는 `연구_보고서.md`를 확인한다. Windows에서 ZIP 전체를 복사·압축 해제하고 `index.html`을 열어 수동 검수한다. 자동 판정은 GT가 아니며 CSV 가져오기에서 기존 편집은 보존된다.

CPU 재현 시 `analyze.py`, `template.html`, `tests.py`, `finalize.py`를 **새 빈 디렉터리**에 복사한 후 다음을 실행한다. 기존 실험의 절대경로와 Python 환경이 필요하다.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B analyze.py
CUDA_VISIBLE_DEVICES='' /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B tests.py
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/hcce15b_mpl /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B finalize.py
```

`test_browser.py`는 별도 headless geckodriver가 45678 포트에서 실행될 때 브라우저 가져오기·내보내기·보존을 검증한다. 현재 코드에서 페이지 URL은 이번 결과 경로다.
