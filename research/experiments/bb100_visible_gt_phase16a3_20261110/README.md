# BB100 Visible Mask 편집기

10장은 아직 모두 미저장 DRAFT다. 상세 사용 규약과 한계는 `연구_보고서.md` 참조.

서버에서:

```bash
cd /home/oylee/projects/HCCEPose_experiments/bb100_visible_gt_phase16a3_20261110
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B app.py --port 7867
```

Windows에서 SSH 터널을 연다:

```bash
ssh -L 7867:127.0.0.1:7867 oylee@<연구서버주소>
```

Windows Chrome에서 `http://127.0.0.1:7867`을 연다. 1/2/3은 전경/지우개/ignore, Ctrl+Z/Y는 undo/redo다. 후보 선택 후 초기화하고 직접 수정한다. 사용자 검수 및 명시적 승인 전에는 ACCEPTED가 아니다. Ignore는 학습과 평가에서 제외해야 한다.

사용자 승인 이후 평가(평가 중 편집 저장 중지):

```bash
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B evaluate.py
```

`BB100_visible_GT_candidate_review_windows.zip`은 **오프라인 비교만** 지원하며 편집기는 서버와 SSH 터널이 필요하다.
