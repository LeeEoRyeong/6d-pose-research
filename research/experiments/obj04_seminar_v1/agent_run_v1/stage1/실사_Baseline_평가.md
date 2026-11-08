# Stage 1 — 실사27 Baseline 평가

- 결과: 27프레임 매핑·K/GT/CAD 선행조건 통과. 모델은 epoch155, CAD 기반 GT Oracle ROI.
- 데이터: 원본 RGB를 K,D로 보정해 출력 K=원본 K인 PNG를 신규 생성. 기존 UND 생성 이력은 확정하지 않았다. 두 보정 영상의 평균 픽셀 MAE 0.4342 (8-bit)이며, 이는 픽셀 수준 재현 비교이지 캘리브레이션 정밀도 자체가 아니다.
- 결과: 유효 PnP 27/27, ADD@0.1d 0/27, 평균 ADD 129.0455, RE 151.5618°, TE 110.3013(PBR/CAD 수치 단위). 평균 inlier 비율 0.4817%.
- 전체27 포함, 066/068/070/072/073은 기존 시각 검토의 판단 불가 표식을 유지. 실사 GT는 조건부 개발 평가로서 정밀 mm·deg 정확도와 독립 최종 테스트 지위를 보장하지 않는다.

프레임별: [real_pose_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage1/real_pose_per_frame.csv). 카메라·GT 체크: [stage1_preflight.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/stage1_preflight.json). 오버레이: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage1/overlays`.
