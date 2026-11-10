# BB100 전체 Visible Mask 검수

상세 연구 결과는 `연구_보고서.md` 참조. 기존 10장 ACCEPTED는 이전 실험의 해시 일치 스냅샷이며 나머지 90장은 DRAFT다. 두 편집기는 자동 동기화되지 않으므로 이후에는 새 7870 편집기만 사용한다.

서버 실행:

```bash
cd /home/oylee/projects/HCCEPose_experiments/bb100_visible_gt_phase16a4_20261110
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B app.py --port 7870
```

Windows 터미널에서 `ssh -L 7870:127.0.0.1:7870 oylee@<연구서버주소>`를 실행하고 Chrome으로 `http://127.0.0.1:7870`에 접속한다. 새 D 후보는 무조건 GT가 아니다. 특히 090, 093, 095의 그리퍼 제거량이 커 B/C와 대조한다.

승인 상태: `annotations/frame_status.csv`. `0/128/255`에서 128은 ignore. 사용자가 직접 확인해야만 ACCEPTED로 저장한다. 10장의 기존 승인본과 초기 후보는 바뀌지 않는다.

진행 후 BOP 준비 manifest 새 스냅샷:

```bash
/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python -B create_manifest.py
```

HCCEPose 학습은 아직 금지·미준비 상태다.
