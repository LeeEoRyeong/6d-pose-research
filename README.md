# HCCEPose RGB-only 6D Pose 연구 백업

ITODD obj04를 대상으로 PBR baseline, 공식 사전학습 HCCEPose와 SMT 환경 RGB-only 추론을 연구한 코드와 결과다. 원본 저장소와 연구 이력은 별도로 보존한다. 아래 수치는 개발 데이터의 조건부 결과이며 독립 테스트 성능이 아니다.

## 현재 연구 결과

| 평가 조건 | PnP | 명시적 대칭 ADD@0.1d |
|---|---:|---:|
| 공식 EfficientNet-B4 + 실사 개발27 oracle ROI | 27/27 | 21/27 |
| YOLO11n + 공식 모델, 실사 개발27 | 26/27 | 17/27 |
| 기존 1 epoch YOLO + 공식 모델 | 0/27 | 0/27 |
| 공식 모델 + BB100 oracle ROI | 100/100 | 55/100 |

YOLO는 검출 27/27이고 25/27은 CAD amodal bbox와 IoU≥0.5였다. BB100은 단일 세션이며 distortion 및 pose 품질이 완전히 검증되지 않았다. CAD amodal은 visible GT가 아니다. HCCEPose fine-tuning은 하지 않았다.

## 최근 연구 자료

- [공식 체크포인트 호환성 감사](research/experiments/itodd_official_audit_20261108/결과_보고서.md)
- [공식 이산 대칭 및 기존 모델 비교](research/experiments/itodd_symmetry_eval_20261108/연구_보고서.md)
- [ROI 강건성 및 BB100 진단](research/experiments/official_rgb_pose_advanced_20261108/연구_보고서.md)
- [YOLO + HCCEPose 파이프라인](research/experiments/yolo_hcce_rgb_pipeline/연구_보고서.md)과 [추론 코드](research/experiments/yolo_hcce_rgb_pipeline/run_inference.py)
- [BB 약지도 검토](research/experiments/bb_weak_supervision_20261108/결과_보고서.md) 및 [SAM 후보 실험](research/experiments/bb_sam_candidates_20261108/결과_보고서.md)

이 저장소는 BB100 승인 마스크와 Pose 스냅샷을 포함한다. RGB·PBR 이미지, CAD, 모델 및 체크포인트와 RGB 오버레이는 포함하지 않는다. 이전 백업의 일부 CSV는 예측 pose와 이미지 식별 정보를 제외했으며, 2026-11-11 스냅샷은 해시와 서버 경로 등 재현 메타데이터를 포함한다. 복사본의 로컬 경로는 `/RESEARCH_DATA` 등 자리표시자로 바뀌었으므로 사용 전 별도 설정이 필요하다. 코드가 원본 HCCEPose 및 외부 모델·데이터에 의존하므로 저장소만으로 즉시 재현할 수 없다.

## 출처와 라이선스

[원본 HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)의 LGPL 2.1 고지와 원본 commit은 [출처 문서](provenance/UPSTREAM.md)와 [라이선스](LICENSES/HCCEPose-LGPL-2.1.txt)에 남겼다. `research/`의 일부 파일은 upstream 파생물·스냅샷이고 일부는 연구용 코드다. 제3자 의존성은 [별도 라이선스 안내](provenance/THIRD_PARTY.md)를 확인한다. 원본 Git 이력은 이 연구 전용 저장소에 포함되지 않으며 upstream과 로컬 원본에 보존된다.

## 이전 백업 안내


실사 BB 객체 4의 Front/Back XYZ 및 visible mask, PBR 학습 및 평가를 검토한 연구 코드와 보고서다. 원본 프로젝트와 개인 데이터는 별도 보관한다. 이전 백업에서는 실사·PBR 이미지, 원본 pose, CAD 모델, 마스크, 체크포인트 및 가상환경을 제외했다. 이후 BB100 승인 마스크 및 Pose 스냅샷은 추가되었다.

## 구조

- `research/local/`: 원본 작업공간에서 작성한 분석 보고서와 BB 사전검증 코드.
- `research/experiments/obj04_seminar_v1/`: PBR baseline, 진단, 평가 및 GT-mask loss 실험 코드·설정·선택 지표.
- `research/experiments/real_bb100_20261108/`: BB pose 변환과 투영 검증 도구의 코드만 포함.
- `provenance/`: 원본 버전과 백업 정제 내역.
- `LICENSES/`: 원본 프로젝트의 라이선스.

## 사용 시 주의

연구 기록은 기존 환경의 실행 상태를 보존하기 위한 것이다. 데이터와 모델이 없으므로 이 저장소만으로 학습·평가를 실행할 수 없다. 소스 및 설정의 `/RESEARCH_DATA/...`, Windows `C:\RESEARCH_DATA\...`는 비식별화된 자리표시자이며, 사용자가 소유한 외부 데이터 경로와 일치하도록 별도 설정해야 한다. 원본 프로젝트 의존 코드 및 Python 환경도 필요하다. 일부 보고서의 이미지·원본 데이터 링크는 의도적으로 백업에서 빠져 있다. 스크립트를 실행하기 전에 입력과 출력 경로를 확인해 기존 파일이 덮어쓰이지 않게 한다.

BB100 visible mask는 100장 사용자 승인 상태이며 저장본 무결성 검사는 미저장 편집 여부 확인 전 CONDITIONAL이다. 픽셀 단위 독립 정확도는 미인증이며 학습 준비는 별도 검증이 필요하다. CAD 투영 mask를 검증된 visible mask로 간주하지 않는다. 평가 결과의 실사 GT 및 oracle ROI 제한은 각 보고서를 참고한다.

## BB100 검수 스냅샷 (2026-11-11)

[승인 마스크·Pose 및 저장 이력](research/bb100_gt_snapshot_20261111/README.md)과 [Phase 16A-5 무결성 보고서](research/experiments/bb100_gt_integrity_phase16a5_20261111/연구_보고서.md)를 확인한다. RGB 원본, CAD, 가중치, 후보 마스크와 RGB 포함 저장 이력 오버레이는 별도 백업해야 한다. 최신 스냅샷에는 로컬 경로, 이미지 해시, 촬영 시각을 포함하는 메타데이터가 있으므로 접근 권한을 관리한다.
