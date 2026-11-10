# RobotSeg ROI·Guided Prompt 실험

결론과 제약은 `연구_보고서.md` 참조. `BB100_RobotSeg_ROI_guided_review.zip` 전체를 압축 해제하고 `index.html`을 연다. 모든 mask는 미검수 후보이다.

재현 시 `run.py`, `package.py`, `template.html`을 기존 파일이 없는 새 결과 디렉터리에 복사한다. 이 스크립트는 기존 RobotSeg 실험의 source·가상환경, 원본 checkpoint 및 기존 SAM/CAD 경로를 참조한다. 첫 명령은 GPU 추론을 실행하고 다른 GPU compute 작업이 있으면 중단한다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/oylee/projects/HCCEPose_experiments/bb100_robotseg_phase16a2r_resume_20261110/.venv/bin/python -B run.py
/home/oylee/projects/HCCEPose_experiments/bb100_robotseg_phase16a2r_resume_20261110/.venv/bin/python -B package.py
```
