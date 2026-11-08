# SMT obj04 HCCEPose 자율 연구 실행 보고서

작성: 2026-11-03T23:15:57.923658+09:00  
실험 루트: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1`  
보고서 생성까지 약 12.48분 / 최대 90분.

**결론: 기존 Baseline은 실사 개발용 27장에서 ADD@0.1d 0/27이었다. 지정한 회전 증강 단일 실험은 선택된 checkpoint에서 PBR Val의 좌표 및 Pose를 악화시켰고, 실사 ADD는 0/27로 개선되지 않았다.**

## 1. 기존 Baseline과 사전 병목

기존 Baseline은 PBR Train80/Val20(생성 블록 분리), random ResNet34 seed42, batch4/crop256, Adam lr0.0002, AMP float16, GT visible BBox crop이다. 총195epoch/3900 optimizer step 중 PBR Val Loss로 epoch155/3100step을 선택했다. 원본 Loss의 NumPy scalar→tensor 문제는 이전 작업 복사본의 명시적 float 변환을 사용했다.

이전 Train/Val 진단: Front XYZ L2 4.2057/29.4046, ADD 59/80(73.75%)/1/20(5%), Val Mask IoU 0.935782. GT XYZ→HCCE→crop→PnP Oracle은 Val20의 ADD20/20이었다. Val Front Z0 hard bit 오류는44.57%로, Train2.75%에 비해 컸다. Oracle 수치는 네트워크 실제 성능이 아니다.

이번 실험 전 동일 checkpoint를 PBR Val20에 다시 적용한 결과, 프레임별 ADD와 Mask IoU의 최대차이는 모두 0.0/0.0였다. [pbr_repro_gate.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/pbr_repro_gate.json).

## 2. Stage 1 — 카메라·GT 확인과 실사 Baseline

27개의 RGB·Pose NPY·GT·카메라 JSON 프레임 집합이 일치하고, 원본85파일 해시가 이전 manifest와 같았다. 각 NPY와 GT의 R/t, 회전행렬, JSON K가 일치했다. 실사 CAD는 PBR 학습용 CAD 복사본과 해시가 같고, GT 적용 시 모든 CAD 점이 카메라 앞에 있으며 투영 bbox가 영상 안에 있었다.

실사 원본에서 OpenCV Brown-Conrady `(k1,k2,p1,p2,k3)`로 명시적 K,D,newK=K·무 crop/resize 보정 영상을 새 PNG로 생성했다. 그 영상과 기존 `image_undistorted`의 평균 픽셀 차이는0.4342/255였다. 기존 UND의 정확한 생성 이력은 여전히 미확인이고, 이번 모델 입력에는 출력 K가 명시된 새 영상만 사용했다. 이 비교는 카메라의 물리적 정밀도 인증이 아니다.

GT CAD 투영 bbox를 1.5배 정사각형으로 확장한 Oracle ROI를 전 프레임에 적용했다. 결과는 실사 전체 **27/27 유효 PnP, 0/27 ADD 통과**, 평균 ADD 129.0455, 평균 RE 151.5618°, 평균 TE 110.3013, inlier 비율 평균 0.4817%다. ADD/TE 길이는 CAD 원본 수치 단위이며 실측 mm라고 확정하지 않는다. ROI는 GT를 이용하므로 YOLO를 포함한 E2E 성능이 아니다.

이전 CAD 재투영 검토의 판단 불가 5프레임(066,068,070,072,073)도 제외하지 않았다. GT 자체는 거시적 시각 정합성 기준의 개발용 후보이며, 정밀 공정 허용오차나 독립 최종 테스트 GT로 인증되지 않았다. [실사_Baseline_평가.md](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage1/실사_Baseline_평가.md).

## 3. Stage 2 — 실패 분석

| 환경 | ADD 통과 | 평균 inlier 비율 | 조건 |
|---|---:|---:|---|
| PBR Val20 | 1/20 | 4.0090% | GT visible BBox |
| 실사27 | 0/27 | 0.4817% | GT CAD 투영 amodal BBox |

PnP 반환과 ADD 성공은 다르다. 기존 육안 가림 분류상 도구 가림 17장, 가림이 없거나 적은 8장 모두 ADD 통과가0이었다. 2장은 미분류다. 가림을 정량화할 visible Mask GT는 없다. 이 결과는 가림만으로 실패를 설명하기 어렵다는 관찰이지 가림 영향이 없다는 증거가 아니다.

PBR과 실사는 ROI 정의도 다르다. 반사·조명·배경·가림과 객체 자세/형상, 미검증 GT 정밀도, ROI 차이, 학습 좌표 일반화가 후보이며 각 원인은 인과 분리되지 않았다. 000051과000092의 overlay를 시각 확인했으며 예측과 GT 방향·크기 차이가 나타난다. 가림을 모델링하지 않은 wireframe의 한계가 있다. [실사_오류분석.md](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage2/실사_오류분석.md), [priority_examples.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage2/priority_examples.png).

### 실사 프레임별 Baseline

| 프레임 | 이전 가림 표식 | GT 육안 품질 | 유효 PnP | ADD | inlier 비율 | ADD |
|---|---|---|---|---:|---:|---|
| 000026 | 도구 가림 관찰 | 거시 정합 | 성공 | 138.94 | 0.18% | 미통과 |
| 000032 | 도구 가림 관찰 | 거시 정합 | 성공 | 95.80 | 0.39% | 미통과 |
| 000038 | 가림 없거나 적음 | 거시 정합 | 성공 | 54.70 | 0.37% | 미통과 |
| 000039 | 가림 없거나 적음 | 거시 정합 | 성공 | 160.72 | 0.35% | 미통과 |
| 000050 | 미분류 | 거시 정합 | 성공 | 87.79 | 0.61% | 미통과 |
| 000051 | 가림 없거나 적음 | 거시 정합 | 성공 | 71.55 | 0.38% | 미통과 |
| 000056 | 가림 없거나 적음 | 거시 정합 | 성공 | 74.78 | 0.63% | 미통과 |
| 000061 | 도구 가림 관찰 | 거시 정합 | 성공 | 183.53 | 0.36% | 미통과 |
| 000065 | 도구 가림 관찰 | 거시 정합 | 성공 | 78.22 | 0.28% | 미통과 |
| 000066 | 도구 가림 관찰 | 판단불가 | 성공 | 167.28 | 0.24% | 미통과 |
| 000068 | 도구 가림 관찰 | 판단불가 | 성공 | 243.70 | 0.22% | 미통과 |
| 000070 | 도구 가림 관찰 | 판단불가 | 성공 | 78.79 | 0.36% | 미통과 |
| 000072 | 도구 가림 관찰 | 판단불가 | 성공 | 157.26 | 0.28% | 미통과 |
| 000073 | 도구 가림 관찰 | 판단불가 | 성공 | 100.04 | 0.30% | 미통과 |
| 000078 | 도구 가림 관찰 | 거시 정합 | 성공 | 142.74 | 0.34% | 미통과 |
| 000083 | 도구 가림 관찰 | 거시 정합 | 성공 | 139.38 | 0.29% | 미통과 |
| 000088 | 미분류 | 거시 정합 | 성공 | 166.77 | 0.13% | 미통과 |
| 000091 | 도구 가림 관찰 | 거시 정합 | 성공 | 121.95 | 0.62% | 미통과 |
| 000092 | 도구 가림 관찰 | 거시 정합 | 성공 | 88.16 | 0.63% | 미통과 |
| 000094 | 도구 가림 관찰 | 거시 정합 | 성공 | 176.47 | 0.67% | 미통과 |
| 000096 | 도구 가림 관찰 | 거시 정합 | 성공 | 70.56 | 0.51% | 미통과 |
| 000098 | 도구 가림 관찰 | 거시 정합 | 성공 | 124.07 | 0.27% | 미통과 |
| 000099 | 도구 가림 관찰 | 거시 정합 | 성공 | 121.94 | 0.43% | 미통과 |
| 000108 | 가림 없거나 적음 | 거시 정합 | 성공 | 76.91 | 1.31% | 미통과 |
| 000118 | 가림 없거나 적음 | 거시 정합 | 성공 | 262.04 | 1.01% | 미통과 |
| 000119 | 가림 없거나 적음 | 거시 정합 | 성공 | 217.56 | 0.92% | 미통과 |
| 000126 | 가림 없거나 적음 | 거시 정합 | 성공 | 82.58 | 0.95% | 미통과 |

가림 표식은 이전 사람이 검토한 보고서의 정성 기록이며 자동 생성 visible Mask가 아니다. 전체 원자료: [real_frame_analysis.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage2/real_frame_analysis.csv).

## 4. Stage 3 — 회전 증강 단일 실험

RGB·Mask·Front/Back XYZ·유효 support에 같은 회전을 적용하고, RGB는 bilinear, label은 nearest, HCCE는 변환된 XYZ에서 다시 생성했다. Crop 내 회전 각도는 표본별 uniform[-20°,20°], 독립 RNG seed42다. 다른 초기화·분할·Loss·optimizer·batch·ValLoss 선택 및 early-stopping 규칙은 유지했다. 지원하지 않는 valid-mask 가중치는 새로 Loss에 삽입하지 않았다.

CPU 정합성 및 CPU 독립 smoke가 통과했고, 실제 GPU 학습은 **단 1회**다. 초기 state는 Baseline epoch0과 전부 일치했다. 유한 Loss로 50epoch, optimizer 999회, AMP skip 1회를 수행하고 기존 규칙에 의해 early stop했다. 실사 이미지는 학습에 사용하지 않았다. Baseline은 epoch155에서 선택됐지만 회전 모델은 epoch10에서 선택됐다. 같은 규칙의 결과이지만 평가 checkpoint의 유효 학습량이 다르므로 회전 자체의 효과와 학습 단계의 효과를 완전히 분리할 수 없다.

학습 후 추가 CPU 검증에서 독립 역 affine 참조의 유효 픽셀 382,368개에 대한 Mask·Front·Back·valid의 최근접 값 불일치는 각각 0개였다. RGB bilinear와 Mask nearest의 합성 표식 16개에서 중심 최대 차이는 128 해상도 기준 0.1693픽셀 미만이었다. 반픽셀 동률 근처 0.05픽셀은 보간 규칙 차이를 피하기 위해 제외했다. 이는 학습 전 필수 게이트와 별도로 실행한 확인이다. [독립 정합성 검사](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/augmentation_spatial_independent.json).

### PBR Val20 — 고정된 주 checkpoint 비교

| 지표 | Baseline epoch155 | 회전 증강 epoch10 |
|---|---:|---:|
| Mask IoU | 0.935782 | 0.840029 |
| Front XYZ L2 | 29.4046 | 61.8816 |
| Back XYZ L2 | 29.1845 | 61.7875 |
| Front Z0 hard bit 오류 | 44.57% | 39.25% |
| Back Z0 hard bit 오류 | 43.12% | 47.37% |
| Front 유효 PnP | 20/20 | 0/20 |
| Front ADD@0.1d | 1/20 | 0/20 |
| Front RE 평균(유효 반환) | 81.0261° | 판단 불가 |
| Front TE 평균(유효 반환) | 60.9094 | 판단 불가 |

회전 모델의 Val Front PnP는0/20이므로 평균 RE/TE는 계산할 수 없다. Mask·Front/Back 좌표·ADD는 악화됐다. Front Z0 오류율은 감소했지만 Back Z0 오류율은 증가했다. 두 모델 모두 PBR Val을 모델 선택에 사용한 개발 결과이며 독립 테스트가 아니다. [pbr_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/pbr_comparison.png), [validation_loss_curve.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/validation_loss_curve.png).

예측 Front 3D 대응점의 Val 프레임당 unique 평균은 1845.0→75.1, X 표준편차 평균은 25.1331→0.1759였다. 회전 모델의 거의 한 축에 몰린 대응점은 PnP 실패와 일치하는 증거다. 원인 자체는 증강·체크포인트 시점·Loss 게이팅 등을 구분할 추가 대조 없이 확정하지 않는다.

### 고정 checkpoint의 실사27 비교

| 지표 | Baseline | 회전 증강 |
|---|---:|---:|
| 유효 PnP | 27/27 | 5/27 |
| ADD@0.1d | 0/27 | 0/27 |
| 평균 inlier 비율, 반환 실패 포함 | 0.4817% | 0.0390% |
| 평균 ADD, 유효 PnP만 | 129.0455 (n=27) | 155.7488 (n=5) |
| 평균 RE, 유효 PnP만 | 151.5618° (n=27) | 136.9421° (n=5) |
| 평균 TE, 유효 PnP만 | 110.3013 (n=27) | 135.9352 (n=5) |

유효 Pose 평균의 분모가 다르므로 두 평균을 직접 비교해 실사 성능 차이를 확정하지 않는다. 양쪽 유효인 동일5프레임의 Baseline/회전 평균 ADD는 125.8301/155.7488이나 선택된 소표본 탐색 수치다. 실사는 checkpoint 선택 후 동일 보정 영상·ROI로 평가했고 학습·선택·튜닝에는 사용하지 않았다. [real_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/real_comparison.png), [real_comparison_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/real_comparison_per_frame.csv).

동일 유효 정의(성공·유한·양의 깊이)에서는 Train Front frame83의 극단적인 유한 TE≈6.29e17과 inlier1개도 포함됐다. 이는 유효 반환과 물리적으로 합리적인 Pose가 다름을 보여주며 조건부 평균을 왜곡한다. 비유한 오차는 CSV에서 발견되지 않았고, 출력된 NumPy overflow 경고와 모든 실패는 보존했다. 임의의 사후 거리 cutoff는 추가하지 않았다.

세부: [회전증강_평가.md](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/회전증강_평가.md), [pbr_comparison.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/pbr_comparison.csv), [pnp_correspondence_geometry.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/pnp_correspondence_geometry.csv).

## 5. 결론·한계·다음 연구

**[확인]** 기존 모델은 PBR Val에서 낮은 좌표 일반화와 5% ADD, 실사 개발27에서는 0% ADD를 보였다. 지정한 회전 증강 1회는 기존 선택 규칙에서 PBR Pose·좌표를 악화시켰고 실사 ADD 통과 수를 늘리지 못했다. 개선 효과는 입증되지 않았다.

**[한계]** 하나의 seed, 합성 Val 두 블록과 상관된 실사27프레임, 조건부 GT 및 Oracle ROI다. Train/Val Loss가 예측 Mask에 의존해 낮은 Loss와 좌표 품질이 일치하지 않을 수 있고, 두 주 checkpoint의 학습량이 다르다. 실사 Mask IoU는 실제 visible Mask GT가 없어 계산하지 않았다. 물리 mm 단위와 GT 정밀도도 별도 확인이 필요하다.

**[제안·미실행]** 다음 연구는 추가 실사 튜닝 전에 checkpoint 선택 규칙의 좌표 붕괴 민감도를 별도 실험으로 검증한다. 예를 들어 고정 PBR 검증에서 좌표별 분산·Z0/XYZ 오차와 PnP 유효율을 함께 기록하고, 미리 정한 조건으로만 checkpoint 선택을 비교한다. 이 제안은 Loss/epoch/augmentation을 자동 탐색하라는 뜻이 아니며 다음 실행 전 실험 설계가 필요하다. 이후 새로운 촬영 세션의 독립 실사 테스트와 실제 가림 Mask, 검증된 detector를 확보한다.

## 6. 산출물·보존

| 경로 | 내용 |
|---|---|
| [stage1_preflight.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/stage1_preflight.json) | 원본85·매핑·K/GT/CAD 게이트 |
| [real_pose_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage1/real_pose_per_frame.csv) | Baseline 실사27 결과, 실패 포함 |
| [real_frame_analysis.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage2/real_frame_analysis.csv) | 프레임별 가림 표식·오류 분석 |
| [pbr_repro_gate.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/pbr_repro_gate.json) | 기존 PBR Val 재현 |
| [augmentation_alignment.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/augmentation_alignment.json) / [augmentation_cpu_smoke.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/augmentation_cpu_smoke.json) | CPU 정합성 및 smoke |
| [rotation_training.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/rotation_training.json) | 단일 학습과 선택·자원 기록 |
| [pbr_comparison.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/pbr_comparison.csv) | Train/Val Mask·XYZ·Z0·Pose 비교 |
| [real_comparison_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/real_comparison_per_frame.csv) | 실사27 동일 프레임 비교 |
| [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/final_integrity.json) | 원본·이전 작업 해시 재검증 |
| `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage1/overlays` / `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/real_overlays` | Baseline/증강 GT 대비 CAD overlay |
| `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/logs` / `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/stage3/checkpoints` | 실행 원문·회전 모델 checkpoint |

원본740파일, 이전 세미나366파일, 이전 Baseline61파일, 이전 Pose 진단102파일, 이전 Train/Val 진단63파일의 해시가 기존 manifest와 일치했다. 기존 venv 재사용, 추가 패키지 설치·다운로드 없음. GPU 작업은 순차 실행했고 다른 compute PID가 있으면 중단하는 guard를 사용했다. 종료 시 compute 조회: `실행 중인 compute process 없음`.
