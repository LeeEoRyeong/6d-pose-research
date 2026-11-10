# RobotSeg 6장 실험

상세 결과는 `연구_보고서.md`. Windows에서는 `BB100_RobotSeg_6frame_review.zip` 전체를 풀고 `index.html`을 연다. 각 마스크는 GT가 아닌 후보이다.

재현은 기존 결과를 덮어쓰지 않도록 새 실험 디렉터리에 공식 고정 commit의 소스, 별도 가상환경과 스크립트를 준비해야 한다. 체크포인트 입력 경로는 `/home/oylee/projects/HCCEPose/robotseg.pt`이며 코드에서 SHA-256을 검사한다. 아래 명령은 GPU 추론을 실행하므로 GPU가 비어 있는지 확인해야 한다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B run_inference.py --frames 235 --tag smoke
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B run_inference.py --frames 75 99 221 340 359 --tag remaining
```

이 설정은 단일 이미지 자동 `gripper` 프롬프트이며 공식 데모의 guided refinement와 동적 multimask 후처리는 사용하지 않는다. 참조된 기존 Python 환경의 패키지 버전에 의존한다.
