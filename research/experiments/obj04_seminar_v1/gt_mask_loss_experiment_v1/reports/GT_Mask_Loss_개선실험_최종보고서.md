# HCCEPose SMT obj04 — GT Mask 좌표 Loss 개선 실험 최종보고서

작성: 2026-11-04T00:01:11.395731+09:00  
실험 루트: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1`  
보고서 작성 시 전체 약 11.61분 / 최대90분.

**결과: PBR Val에서 Front/Back 평균 XYZ 오차는 29.2946→29.9958로 증가했다. Front ADD 통과는 1/20→3/20(+2개 증가), 실사 개발27 ADD 통과는 0/27→0/27(변화 없음)이다.** 원본 코드는 유지하고 GT 유효 Mask의 좌표 Loss 게이트만 바꾼 단일 학습 결과다.

## 1. 연구 목적과 기존 문제

기존 Baseline은 PBR Train80/Val20의 obj04, random ResNet34, GT BBox Oracle crop으로 학습했다. Val Front/Back XYZ L2는 29.4046/29.1845, Front Z0 오류는 44.57%, Front ADD는 1/20이었다. 기존 실사 개발27 ADD는 0/27이다.

이전 분석에서 예측 Mask를 좌표 Loss의 게이트로 쓰면 놓친 GT 객체 픽셀의 코드 오차가 빠지는 사실을 확인했다. GT Mask를 이용한 고정 가중치 반사실에서 일부 checkpoint의 Loss 순서가 달랐지만, 이는 재학습 효과의 증거가 아니었다. 이번에는 **좌표 L1의 공간 게이트만** 변경한 실제 단일 학습으로 가설을 시험했다.

## 2. 기존 Loss와 변경 수식

기존 좌표 L1은 `round(sigmoid(pred_mask)).detach()`를 Front/Back 코드와 target에 곱한 후 batch·height·width 전체 평균을 취한다. GT 객체이지만 예측 Mask가 놓친 픽셀은 좌표 항에 기여하지 않는다. Mask Loss 자체는 `L1(sigmoid(pred_mask), gt_mask)`이다.

이번 변경은 branch별 `G_front = GT_visible_mask ∩ valid_front`, `G_back = GT_visible_mask ∩ valid_back`을 Front/Back L1의 공간 게이트로 사용한다. Label validity는 실제 렌더의 유한 near/far depth support를 쓰며 단순히 XYZ 값이 0이 아니라고 가정하지 않는다.

```text
L_front_axis = weighted_mean_planes(mean_BHW(abs(pred_code * G_front
                                                  - (2 * GT_code - 1) * G_front)))
L_back_axis  = 같은 식에서 G_back 사용
L_total = L_mask + 3 * sum(L_front_axis) + 3 * sum(L_back_axis)
```

plane별 동적 weight와 길이100 history는 **이전과 동일하게 predicted Mask** 기반이다. Mask Loss·HCCE encoding/복호화·channel order·target 변환·3배 계수·batch/crop 전체 평균·AMP는 유지했다. 유효 픽셀 수로 나누는 새 정규화는 추가하지 않았다. 유효 픽셀 수 변화는독립변수의 일부다.

근거: 원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`, 기존 호환 복사본 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/code/network_model_explicit.py`, 새 복사본 [network_gtmask.py](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/code/network_gtmask.py), 정확한 차이 [loss_change.diff](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/loss_change.diff).

## 3. 유효 support·학습 전 검증

기존 CPU near/far CAD renderer에서 100프레임의 finite-depth support를 재구성하고, 원본 Front/Back 양자화 PNG 전체와 각 픽셀을 다시 계산해 **100/100 정확히 일치**시켰다. 이 데이터에서는 유효한 양자화 전체0 픽셀 Front/Back 각각 0/0개, 두 branch support 차이 0픽셀이었다. 별도 branch support 이미지를 저장하고 GT visible Mask와 각각 교차했다. 원본 VisPy 구현과의 bitwise 비교는 이전과 마찬가지로 미완료다.

| 확인 | 실제 결과 |
|---|---|
| 기존 epoch155 Val Total Loss | 기록 1.537051797, 재계산 1.537051797, 차이 0.0 |
| 동일 Mask 인공 배치 | CPU/GPU 기존·새 Loss와 gradient 일치; 상세 [loss_equivalence.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/loss_equivalence.json) |
| 예측 Mask가 GT 전경 전체 누락 | 기존 좌표 항0, 변경 항은 Front/Back 양수; GT 유효영역에 gradient 전달 |
| 배경·무효 support | 각 branch의 게이트 밖 좌표 gradient 0; branch별로 다른 support 인공 사례 통과 |
| Mask Loss·dynamic weight/history | Mask Loss 동일; 동일 predicted Mask에서 gate가 달라도 dynamic weight와 history 일치 |
| GPU smoke | optimizer 적용0회, Front/Back gradient 유한·0 아님, 독립 모델 폐기 |
| 초기화·배치 순서 | random seed42; 모델 state 전부 epoch0 참조와 동일. 실제 학습 195개 epoch의 기록된 batch 순서도 Baseline과 일치 |

실제 Val 첫4장 동일 출력에서 기존/변경 total은 1.416403/1.485752, Mask 항은 둘 다 0.008233였다. 당시 predicted gate 12303픽셀, GT 유효 Front/Back gate 12696/12696픽셀이었다. 두 Loss는 정의가 다르므로 이 숫자의 크기 차이를 성능 개선으로 해석하지 않는다.

상세: [support_validation.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/support_validation.json), [mask_gate_validation.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/mask_gate_validation.json), [initialization.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/initialization.json), [gpu_smoke.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/gpu_smoke.json), [batch_order_verification.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/batch_order_verification.json).

## 4. 학습 조건과 checkpoint 선택

| 항목 | 실제 설정·결과 |
|---|---|
| 데이터 | 기존 Train80 / Val20, 생성 block [0, 1, 2, 3, 5, 6, 7, 8] / [4, 9], 실사0 |
| 모델·초기화 | ResNet34 random, seed42, pretrained 없음, 초기 state 동일 |
| 입력·optimizer | BGR 정규화, crop256/label128, batch4, Adam lr0.0002, AMP float16 |
| 증강·PnP | 증강 없음, 원본 Front/Back RANSAC EPNP 1.5px·150회·seed42+frame |
| 상한·중단 | 최대195epoch/3900 optimizer step, 동일 ValLoss patience8·min_delta0.0001 |
| 실제 학습 | **1회**, 195epoch, optimizer 3899회 / 시도3900회, AMP skip 1회, 종료 `max_epochs` |
| 자원·시간 | 학습 wall 321.44초, peak allocated 987.22MiB. 평가·저장 포함 wall 시간 |
| 공식 checkpoint | 변경 Loss의 PBR Val total Loss 기준 epoch **170**; [best_val_loss.pt](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checkpoints/best_val_loss.pt) |
| 보조 checkpoint | PBR Val GT 유효영역 `0.5*(Front XYZ L2+Back XYZ L2)` 기준 epoch **195**; [best_val_xyz.pt](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checkpoints/best_val_xyz.pt) |

두 기준은 서로 다른 epoch를 선택했다. 공식 결과는 Val Loss checkpoint이며 XYZ checkpoint는 PBR 보조 분석으로만 보존했다. 공식 선택은 실사 데이터를 읽기 전에 [selection_frozen.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/selection_frozen.json)에 epoch·SHA-256을 고정했다. 실사 결과를 이용한 재선택·추가 학습은 없다.

각 평가 시점의 Train/Val total·Mask·Front/Back loss, GT 전경 누락률, Front/Back XYZ·축별 MAE·Z0, Mask IoU, PnP·ADD는 [train_history.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/train_history.csv)와 `logs/evaluation.jsonl`에 있다. Loss history는 평가마다 새로 만들며, 교육용과 validation history를 섞지 않았다. [train_val_loss.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/train_val_loss.png).

## 5. PBR Validation — 실제 예측 비교

| 지표 | 기존 Baseline epoch155 | GT Mask Loss 공식 epoch170 |
|---|---:|---:|
| Mask IoU | 0.9358 | 0.9501 |
| Front XYZ L2 | 29.4046 | 29.9708 |
| Back XYZ L2 | 29.1845 | 30.0208 |
| Front Z0 오류율 | 44.57% | 47.51% |
| Back Z0 오류율 | 43.12% | 46.07% |
| Front PnP 유효 | 20/20 | 20/20 |
| Front ADD@0.1d | 1/20 | 3/20 |
| Front 평균 ADD (유효 반환) | 62.2474 | 67.7997 |
| Front 평균 RE ° (유효 반환) | 81.0261 | 80.7632 |
| Front 평균 TE (유효 반환) | 60.9094 | 67.1016 |

XYZ는 GT Mask∩실제 branch valid support 내부의 동일 128×128 픽셀에서 계산한 모델 좌표 L2다. Mask IoU는 crop 전체에서 측정했다. 정확도는 실제 네트워크가 예측한 HCCE·Mask와 원본 PnP 반환으로 계산했으며, 이전 GT XYZ Oracle 결과를 모델 성능에 섞지 않았다. 실패한 PnP는 ADD 실패로 포함해 분모를20으로 유지했다.

Front Z0 오류율은 44.57%→47.51%로 증가했다. 같은20프레임에서 Front XYZ 오차가 줄어든 프레임은 10개, 늘어난 프레임은 10개였다. ADD 통과 프레임은 기존 [40]에서 새 모델 [40, 48, 98]로 바뀌었고, 평균 Front PnP inlier 비율은 4.009%→4.642%였다. 평균 XYZ와 threshold 통과 수가 서로 다른 방향으로 바뀐 관찰이며, inlier 변화만이 원인이라는 뜻은 아니다. Front와 Back의 x/y/z MAE 및 유효 PnP만의 ADD·RE·TE는 [pbr_validation_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/pbr_validation_metrics.csv)에 있고 동일 프레임별 비교는 [pbr_validation_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/pbr_validation_per_frame.csv)에 있다. 유효 PnP 수가 다르면 조건부 평균의 분모가 다르며, 유효 Pose가 없을 때 평균은 N/A다. 위 길이는 CAD 원본 수치 단위이고 실제 mm는 독립 계측되지 않았다.

보조 XYZ checkpoint epoch195의 PBR Val Front/Back XYZ는 27.5000/27.0808, Front PnP 20/20, ADD 2/20으로 학습 중 보조 기록됐다. 공식 Loss checkpoint와 다르며, 보조 checkpoint를 실사에 적용하거나 사후 교체하지 않았다.

그림: [xyz_L2_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/xyz_L2_comparison.png), [Z0_error_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/Z0_error_comparison.png), [ADD_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/ADD_comparison.png), [validation_xyz_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/validation_xyz_comparison.png).

## 6. 실사 SMT 27프레임 — 공식 checkpoint 1회

기존 `agent_run_v1/stage1`의 동일한 K 명시 controlled-undistorted RGB, GT CAD 투영 Oracle ROI, GT Pose와 ADD threshold를 그대로 재사용했다. 전체27프레임에 공식 checkpoint를 **한 번만** 적용했고 누락·PnP 실패도 ADD 실패로 분모에 포함했다.

| 지표 | 기존 Baseline | GT Mask Loss |
|---|---:|---:|
| 유효 PnP | 27/27 | 27/27 |
| ADD@0.1d | 0/27 | 0/27 |
| 평균 inlier 비율, 전체27 | 0.4817% | 0.3946% |
| 평균 ADD, 유효 Pose만 | 129.0455 (n=27) | 127.3363 (n=27) |
| 중앙 ADD, 유효 Pose만 | 121.9500 | 107.9112 |
| 평균 RE, 유효 Pose만 | 151.5618° | 143.1742° |
| 중앙 RE, 유효 Pose만 | 157.6855° | 149.0185° |
| 평균 TE, 유효 Pose만 | 110.3013 | 108.7707 |
| 중앙 TE, 유효 Pose만 | 97.9409 | 85.2125 |

조건부 평균은 유효 수가 다르면 직접 비교할 수 없다. 양 모델 모두 유효했던 동일 27개 프레임의 평균 ADD는 기존 129.0455 / 새 모델 127.3363이다. 전체27프레임 모두 유효했다. 실사 Pose GT와 카메라의 물리 단위·정밀도는 완전히 검증되지 않았고, 5개 프레임은 이전 거시 검토에서 판단 불가였다. 제외 없이 보고했으며 이27장은 독립 최종 테스트가 아니다. 실제 visible Mask가 없어 실사 Mask IoU는 계산하지 않았다.

전체 결과와 실패 사유: [real27_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/real27_metrics.csv); 우선5장 비교: [real27_priority_overlay.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/real27_priority_overlay.png); 개별 overlay 27장 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures/real27_overlays`. GT는 녹색, 유효 PnP 예측은 자홍색이며 실패 시 GT만 보인다. CAD wireframe은 실제 도구 가림을 복원하지 않는다.

## 7. 질문별 해석과 한계

1. **PBR XYZ:** Front/Back 평균 L2 29.2946→29.9958, 증가. Front는 29.4046→29.9708, Back은 29.1845→30.0208다.
2. **Z0:** Front 오류는 44.57%→47.51%로 증가; Back은 43.12%→46.07%다. 한 plane의 개선을 전체 Pose 개선으로 간주하지 않는다.
3. **XYZ와 ADD:** Front ADD 1/20→3/20(+2개 증가). XYZ와 PnP·ADD가 다른 방향이면 correspondence 분포 및 PnP 민감도의 가능성을 따로 검증해야 하며, 이번 데이터만으로 원인을 확정하지 않는다.
4. **게이트의 기여:** 초기 상태·순서·주요 설정을 고정하고 공간 gate만 변경한 단일 Run의 관찰이다. 공식 checkpoint epoch는 Baseline155 대 새 모델170, AMP skip·early stopping 경로도 실측대로 달라 단일 seed의 최종 차이를 특정 원인에 대한 보편적인 인과 효과로 확정할 수 없다. 기존/새 total Loss의 절댓값은 정의가 달라 직접 비교하지 않는다.
5. **실사:** 전체27 ADD 0→0(변화 없음), 평균 ADD 129.0455→127.3363(유효 n=27/27)이며 동일 Oracle ROI 조건이다. 작은 개발 세트에서 평균의 수치 차이를 독립적인 정확도 개선으로 확정하지 않는다. 실사는 checkpoint 선택 전에 보지 않았으며 이후 재학습·재선택·조정하지 않았다.

PBR Val 2개 생성 블록과 연속적인 실사27은 작은 개발 자료다. GT 기반 ROI, 제한적인 가림 다양성, 정밀 GT·물리 단위·대칭·실사 visible Mask·검출기 미검증, 단일 seed가 한계다. 이 실험의 성능을 독립 최종 테스트나 공정 허용오차 수준으로 주장할 수 없다.

## 8. 후속 연구 — 제안만, 추가 학습 없음

이번 작업의 추가 개선 학습은 종료한다. 이후 독립 촬영 세션의 정확도 검증, 실측 CAD 단위·실사 카메라/GT 정밀도 확인, 실사 visible Mask 및 detector 검증, 사전에 고정된 독립 테스트와 반복 seed를 권장한다. source PnP Back/Mixed inlier 집계 문제는 별도 회귀검증 대상으로 남는다. 어떤 개선도 아직 확인된 사실로 가정하지 않는다.

## 9. 결과물과 보존

- 실험 설정: [experiment_config.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/configs/experiment_config.json).
- 사전 검증: [loss_equivalence.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/loss_equivalence.json), [mask_gate_validation.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/mask_gate_validation.json), [initialization.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/initialization.json).
- 학습: [train_history.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/train_history.csv), [training_summary.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/training_summary.json), 공식/보조 checkpoint는 4절 참조.
- 평가: [pbr_validation_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/pbr_validation_metrics.csv), [real27_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/tables/real27_metrics.csv), [real27_summary.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/real27_summary.json).
- 전체 그림: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/figures`. 모든 숫자는 저장된 실제 Run의 결과다.
- 원본 및 이전 실험 manifest 재검증: [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/gt_mask_loss_experiment_v1/checks/final_integrity.json). 모두 일치. 추가 설치·다운로드 없음. 종료 GPU compute 조회: `실행 중인 compute process 없음`.
