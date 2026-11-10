# Phase 14 실행 안내

CPU 조사 결과는 연구_보고서.md와 JSON/CSV에 있다. 현재 BLOCKED이며 학습 명령은 제공되지 않는다. `prepare.py`, `checks.py`는 기존 파일 덮어쓰기를 방지하기 위해 출력에 exclusive create를 사용하므로 재현 시 새 빈 디렉터리에 두 스크립트를 복사하고 그곳에서 아래를 실행한다. 원본 데이터 경로는 서버 절대경로다.

```bash
CUDA_VISIBLE_DEVICES='' /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B prepare.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B checks.py
```

E0–E3 YAML은 사전 설계 명세이고 실행 설정으로 사용할 수 없다. 실행 가능한 파인튜닝 명령은 loader/trainer 구현 및 차단 해소 후에 작성한다.
