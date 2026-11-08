# SAM 2.1 BB 후보 실험

결과와 한계: [한국어 보고서](결과_보고서.md). 043/075/235/247 네 장의 SAM 후보만 생성하며 GT나 학습 데이터로 승인하지 않는다.

공식 `sam2_source` checkout과 `sam2.1_hiera_tiny.pt`는 이 디렉터리 안에 있다. `.venv`는 Hydra 등은 분리 설치했지만 기존 `obj04_seminar_v1/env`의 PyTorch·OpenCV를 읽기 전용으로 참조한다. 기존 환경 이동/삭제 시 경로를 다시 구성해야 한다.

재실행 시 GPU로 네 장을 다시 추론하고 새 결과 디렉터리를 만든다:

```bash
cd /RESEARCH_DATA/HCCEPose_experiments/bb_sam_candidates_20261108
.venv/bin/python -B run_candidates.py
```

현재 후보에는 구멍 채움 및 그리퍼 선택 실패가 있어 GPU 혼합 학습과 100장 자동 확장은 보류한다.
