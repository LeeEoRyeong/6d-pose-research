# Phase 16A-2F

상세 결과와 제한은 `연구_보고서.md`를 확인한다. Windows 검수는 ZIP 전체를 압축 해제해 `index.html`을 연다. 원본 SAM·CAD와 탐색 차집합은 모두 GT가 아니다.

독립 `.venv`는 기존 PyTorch 환경을 `.pth`로 읽기 참조한다. 재현 시 `run_pairs.py`와 `package.py`, `template.html`을 새 빈 실험 디렉터리에 복사하고 그곳에 원본 LiteFlowNet checkout, 가중치 및 별도 가상환경을 준비해야 한다. 경로가 스크립트 내 기존 연구 데이터로 고정돼 있다. 아래 첫 명령은 GPU 추론이며, 다른 사용자 compute 작업이 있으면 중단한다. 현재 저장된 결과를 덮어쓰지 않도록 반드시 빈 결과 디렉터리에서 실행한다.

```bash
cd pytorch-liteflownet
CUDA_VISIBLE_DEVICES=0 CUPY_CACHE_DIR=../cupy_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python -B ../run_pairs.py
cd ..
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B package.py
```

원본 1024×436 예제 크기 assertion을 우회하는 래퍼의 제한과 모델 라이선스를 보고서에서 확인한다.
