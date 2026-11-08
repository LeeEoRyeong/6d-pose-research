# HCCEPose 체크포인트 선택 및 XYZ Collapse 원인 진단

작성: 2026-11-03T23:41:23.553257+09:00  
결과: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1`

**결론: 회전 epoch10의 X축 붕괴는 epoch40·50에서 크게 완화됐지만, 세 checkpoint 모두 PBR Val ADD@0.1d가 0/20이다. Val Loss 기준은 붕괴된 epoch10을 선택했다. GT Mask로 좌표 Loss의 게이트만 바꾼 사후 대조에서 Loss 순서가 뒤집혀, 예측 Mask 게이팅이 선택 결과를 왜곡할 수 있다는 실측 근거가 있다.** 이 대조는 재학습이나 실제 개선 성능이 아니며 모든 학습 실패의 인과 원인을 확정하지 않는다.

- 지정된 두 run의 실제 저장 checkpoint 9개를 모두 평가했다. 손상·미평가 checkpoint는 없었다. 별도 진단 run에 실제 존재하는 untrained epoch0 checkpoint를 **초기 참고** 1개로 추가했다. 두 학습 run의 추가 epoch checkpoint를 복원하거나 새로 만들지 않았다.
- 같은 PBR Val20, epoch별 고정 checkpoint, GT visible BBox oracle crop, batch4, 기존 BGR 정규화, GPU AMP float16, 원본 HCCE 복호화·RANSAC EPNP 설정을 사용했다.
- Train optimizer·재학습·실사27 평가나 선택·코드/모델 수정은 수행하지 않았다. 회전 epoch40/50 평가는 사용자가 요청한 사후 진단이며 독립 테스트가 아니다.
- 전체 길이 단위는 PBR/CAD 원본 수치 단위다. mm를 확정하지 않는다. PnP 유효는 solver 성공·유한 오차·CAD 양의 깊이 기준이며 ADD 통과와 다르다.

## 1. 직접 답변

1. **epoch10과 last 중 XYZ가 좋은 쪽:** 회전 last epoch50이다. Front 평균 L2 61.8816→55.2427, Back 61.7875→56.3125. Front PnP 유효0→20이지만 ADD는 **둘 다0/20**이다. epoch40의 Front L2 52.9583는 세 저장 시점 중 더 낮지만 ADD도0/20이다.
2. **Loss 선택이 붕괴를 가렸는가:** 저장된 범위에서 그렇다. 선택된 epoch10의 Front X 표준편차 0.1759와 unique XYZ 75.1는 epoch50의 23.9476, 855.0보다 매우 작다. 이는 선택된 상태의 붕괴와 후속 상태의 복구를 보여주지만, 낮은 Loss가 모든 붕괴를 잡지 못한다는 범위의 결론이다.
3. **학습 부족·선택 오류·다른 문제:** epoch10 이후 X 분산과 XYZ가 개선되어 초기 상태를 고른 영향이 확인된다. 하지만 epoch40/50의 ADD도0이어서 선택 오류만으로 최종 정확도 실패가 설명되지는 않는다. 회전 증강·시점 다양성·손실 설계·추가 학습량의 개별 인과는 미확정이다.
4. **재학습 없이 해결 가능한가:** 실제 저장된 epoch40/50으로 PnP 반환과 XYZ의 일부를 회복할 수 있다. 이 Val에서는 ADD가 여전히0/20이므로 **정확도 문제가 해결됐다는 증거는 없다.** 본 작업에서 실사 checkpoint 선택·추론은 하지 않았다.
5. **다음에 바꿀 한 가지:** PBR 검증의 checkpoint 선택 점수만 기존 총 Val Loss에서 **GT 유효 Mask 내부 Front/Back XYZ L2의 평균**으로 바꾸는 사전 고정 대조를 제안한다. 다른 학습·증강·PnP 조건은 유지한다. 이 저장 checkpoint 집합에서는 epoch40이 선택되지만 ADD0/20이므로 Pose 개선 보장은 없고, 같은 Val을 보고 고른 사후 관찰은 독립 검증 결과가 아니다.

## 2. 실제 파일과 학습 기록

| run | 저장 epoch | 실제 checkpoint | 상태 |
|---|---:|---|---|
| baseline | 155 | `best_val_loss.pt` | 평가 완료 |
| baseline | 180 | `best_val_mask_iou.pt` | 평가 완료 |
| baseline | 50 | `epoch_0050.pt` | 평가 완료 |
| baseline | 100 | `epoch_0100.pt` | 평가 완료 |
| baseline | 150 | `epoch_0150.pt` | 평가 완료 |
| baseline | 195 | `last.pt` | 평가 완료 |
| rotation | 10 | `best_val_loss.pt` | 평가 완료 |
| rotation | 40 | `best_val_mask_iou.pt` | 평가 완료 |
| rotation | 50 | `last.pt` | 평가 완료 |
| diagnostic_initial_reference | 0 | `best_val_loss.pt` | 평가 완료 |

초기 참고 checkpoint는 `runs/baseline_seed42_20261103_214353/checkpoints/best_val_loss.pt`의 **별도 초기 진단 run epoch0**이다. 실제 저장된 파일을 읽었으며 학습 checkpoint로 포장하지 않았다. 기존 회전 run의 초기화 state가 이 파일과 같다는 기록은 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/agent_run_v1/checks/rotation_initialization.json`이다.

| 구분 | 학습 후 평가된 최저 Val Loss | 학습 종료 | optimizer 실제 적용 | AMP skip | 정지 사유 |
|---|---:|---:|---:|---:|---|
| Baseline | epoch155, 1.537052 | epoch195 | 3900 | 0 | Val Loss early stopping |
| 회전 증강 | epoch10, 1.800128 | epoch50 | 999 | 1 | Val Loss early stopping |

무학습 epoch0의 logged Val Loss 0.781321은 두 run의 trained checkpoint 선택 후보에서 의도적으로 제외됐다. 회전 run의 AMP skip은 epoch32에서1회다. Train step 로그는 모든 epoch의 총 optimizer Loss를 포함하고, Train 평가·Val Loss·Mask IoU는 epoch1과 매5epoch에 기록됐다. 각 epoch의 값과 기록되지 않은 평가는 [epoch_loss_history.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/epoch_loss_history.csv)에 있다.

기존 로그에는 epoch별 Mask·Front·Back 개별 Loss와 step별 learning rate가 없다. 해당 필드는 **N/A**이며 checkpoint에서 이번에 재계산한 성분과 구분한다. config와 코드에는 Adam lr=0.0002가 고정돼 있고 scheduler는 확인되지 않았지만, 누락된 과거 step LR 로그를 새로 측정했다고 주장하지 않는다. 저장되지 않은 평가 epoch 44개의 목록: [unsaved_evaluation_epochs.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/tables/unsaved_evaluation_epochs.csv).

## 3. 원본 Loss와 실제 게이트 반사실

원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`의 `HccePose_Loss.forward`는 다음을 계산한다.

- Mask Loss: 예측 sigmoid와 GT Mask의 L1을 crop 전체 픽셀에서 평균.
- Front/Back HCCE: **예측 sigmoid Mask를 round·detach해 각 코드와 GT 코드 양쪽에 곱한 뒤**, 축별 L1을 batch·height·width 전체에서 평균. GT Mask 또는 유효 라벨 Mask를 직접 좌표 Loss의 게이트로 쓰지 않는다. 예측 전경 밖 GT 전경의 좌표 오차는 이 항에서 0이 된다. 반대로 예측된 배경의 코드 오차는 포함될 수 있다.
- 축별8 plane의 동적 가중치는 예측 Mask 내부의 hard error를 `pred_mask.sum()+1`로 정규화하고 길이100 history를 갱신해 계산한다. Soft HCCE target은 `2*GT_code-1`이다. 총 Loss는 Mask + 3×Front 축별 합 + 3×Back 축별 합이다. 근거: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/support.py`의 `loss_value`.

각 checkpoint를 같은 순서의 Val20에서 **새 Loss history**로 평가한 총 Loss는 원래 기록과 최대 원소 차이 0.0였다. 개별 성분은 이때의 사후 재계산이며 과거 epoch별 train 로그가 아니다.

반사실에서는 원래 계산에서 얻은 **각 checkpoint의 예측 Mask history·동적 가중치와 원래 Mask Loss를 그대로 고정**하고, Front/Back 코드 항의 gate만 GT Mask로 바꿨다. 다시 학습하지 않았으며 가중치를 GT 기준으로 갱신하지도 않았다. 따라서 실제 GT-gated 재학습 Loss와는 다르다. 독립 예측-gate 수식을 두 checkpoint 첫 배치의 원본 값과 비교한 최대 차이는1.5e-8 이하다. [counterfactual_formula_check.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/checks/counterfactual_formula_check.json).

| 모델 | epoch | 실제 총 Val Loss | Mask Loss | 3×Front | 3×Back | GT 전경 누락률 | GT gate 반사실 총합 |
|---|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_initial_reference | 0 | 0.781321 | 0.493895 | 0.141500 | 0.145926 | 99.77% | 2.930815 |
| baseline | 155 | 1.537052 | 0.013663 | 0.760383 | 0.763006 | 5.60% | 1.649276 |
| rotation | 10 | 1.800128 | 0.050455 | 0.877819 | 0.871854 | 15.27% | 2.185984 |
| rotation | 40 | 1.950554 | 0.014032 | 0.962833 | 0.973689 | 1.97% | 1.942640 |
| rotation | 50 | 2.036923 | 0.012726 | 1.001461 | 1.022736 | 2.11% | 2.024131 |

**실제 기준:** 회전 epoch10 1.800128 < epoch50 2.036923. **고정 가중치 GT gate 반사실:** epoch10 2.185984 > epoch50 2.024131. epoch10은 GT 전경의 평균 15.27%를 예측 gate에서 누락했고 epoch50은 2.11%였다. 같은 라벨 배열·가중치를 고정한 게이트 교체가 저장된 두 시점의 Loss 순서를 뒤집는다는 직접 증거다. 두 모델 각각의 history 가중치는 기존 예측 Mask로 형성됐다. 원본 Loss와 동일하게 GT Mask 안의 별도 무효 라벨도 추가 필터링하지 않았으므로 이 반사실은 GT 유효 Mask 기반 선택 지표와 구분한다.

별도 untrained epoch0 참고에서는 pred gate가 전체 crop의 1.65%, GT Mask가 19.83%였고 GT 전경의 약 99.77%가 누락됐다. 총 Loss가 0.781321로 낮지만 Front XYZ는 84.1181였다. 초기 reference Mask IoU는 이번 원본 `inference_batch` FP16 sigmoid threshold에서 0.002244이며 과거 로그의 `raw_mask>0` IoU와 임계값 반올림 때문에 소폭 다를 수 있다. 참고 상태는 선택 후보에서 제외됐다.

## 4. 실제 checkpoint별 Val20 평가

| run | epoch | 재계산 Val Loss | Mask IoU | Front/Back XYZ L2 | Front/Back Z0 오류 | Front 유효 PnP | Front ADD 통과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 155 | 1.5371 | 0.9358 | 29.40/29.18 | 44.57%/43.12% | 20/20 | 1/20 |
| baseline | 180 | 1.5942 | 0.9535 | 31.37/31.49 | 47.03%/45.52% | 20/20 | 1/20 |
| baseline | 50 | 1.8147 | 0.9501 | 46.85/47.34 | 59.11%/58.37% | 19/20 | 0/20 |
| baseline | 100 | 1.6957 | 0.9526 | 37.26/37.41 | 58.04%/56.15% | 20/20 | 1/20 |
| baseline | 150 | 1.5715 | 0.9484 | 31.71/31.67 | 48.51%/47.14% | 20/20 | 1/20 |
| baseline | 195 | 1.5956 | 0.9511 | 31.88/32.24 | 45.86%/44.19% | 20/20 | 3/20 |
| rotation | 10 | 1.8001 | 0.8400 | 61.88/61.79 | 39.25%/47.37% | 0/20 | 0/20 |
| rotation | 40 | 1.9506 | 0.9533 | 52.96/52.39 | 62.61%/60.84% | 20/20 | 0/20 |
| rotation | 50 | 2.0369 | 0.9525 | 55.24/56.31 | 55.98%/55.18% | 20/20 | 0/20 |
| diagnostic_initial_reference | 0 | 0.7813 | 0.0022 | 84.12/63.22 | 49.86%/50.98% | 1/20 | 0/20 |

XYZ는 GT Mask와 유효 Front/Back label이 공통으로 있는 crop 픽셀에서 측정했다. Mask IoU는 전체 crop 기준이다. HCCE hard bit는 원본의 FP16 `sigmoid(raw)>0.5` 기준으로 평가했다. PBR `min_xyz/size_xyz`, channel0=X,1=Y,2=Z, 원본 encoder/decoder·crop/정규화는 두 run에서 동일하며 checkpoint는 strict loading을 통과했다. 이전 256값 HCCE round-trip과 GT→PnP Oracle 결과도 이 경로에 대한 근거지만, 이것만으로 회전 데이터의 통계적 적합성을 증명하지는 않는다.

| run | epoch | Front 유효 n | Front 평균 ADD | Front 평균 RE(°) | Front 평균 TE | Back 유효 n | Back 평균 ADD | Back 평균 RE(°) | Back 평균 TE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 155 | 20 | 62.25 | 81.03 | 60.91 | 20 | 74.21 | 83.30 | 75.89 |
| baseline | 180 | 20 | 68.27 | 98.33 | 67.95 | 20 | 65.76 | 86.19 | 70.59 |
| baseline | 50 | 19 | 102.55 | 146.68 | 102.60 | 20 | 109.29 | 129.66 | 111.13 |
| baseline | 100 | 20 | 84.50 | 110.93 | 90.07 | 20 | 74.34 | 105.19 | 76.24 |
| baseline | 150 | 20 | 68.84 | 92.64 | 63.58 | 20 | 77.78 | 79.93 | 73.95 |
| baseline | 195 | 20 | 62.49 | 80.84 | 58.56 | 20 | 54.75 | 81.32 | 53.53 |
| rotation | 10 | 0 | N/A | N/A | N/A | 10 | 236.61 | 113.76 | 215.53 |
| rotation | 40 | 20 | 81.25 | 137.04 | 81.74 | 20 | 3.561e+07 | 144.54 | 3.561e+07 |
| rotation | 50 | 20 | 631.23 | 141.80 | 627.17 | 20 | 108.38 | 126.34 | 109.42 |
| diagnostic_initial_reference | 0 | 1 | 168.74 | 30.74 | 169.24 | 1 | 174.28 | 174.77 | 174.26 |

평균 오차는 **유효 Pose만** 대상으로 하며, 유효 반환이0이면 N/A다. ADD 통과율은 실패를 포함해 항상20으로 나눈다. 회전 epoch50 Front 평균 ADD 631.2312는 최대 10520.8727인 극단적 유한 값에 민감하며 중앙값은 100.4713다. 유효 반환만을 정확한 Pose로 해석하지 않는다. ADD·RE·TE 평균은 유효 n이 다른 checkpoint 사이에서 직접적인 전체 샘플 평균 비교가 아니다.

Baseline의 epoch155는 ADD1/20, epoch195는3/20이지만 Val Loss는 epoch155가 더 낮았다. 이 작은 차이는 같은 Val20에 대한 사후 관찰로, 독립 Pose 성능이나 우월한 checkpoint 정책을 입증하지 않는다.

## 5. 3D Correspondence 붕괴

예측 Mask 내부 Front 3D 점의 축별 표준편차와 unique XYZ 수를 프레임마다 구했다. 공분산은 각 프레임의 `(X-mean)ᵀ(X-mean)/N`이고, 고유값을큰 순서로 표기했다. effective rank는 `exp(-sum(p*log(p)))`, `p=λ/sum(λ)`이다. 아래 값은 프레임별 통계의20장 평균이며 전체 점을 한데 합친 공분산이 아니다.

| run | epoch | X/Y/Z std 평균 | unique XYZ 평균 | covariance 고유값 평균 λ1/λ2/λ3 | effective rank 평균 |
|---|---:|---:|---:|---:|---:|
| baseline | 155 | 25.1331/28.8667/18.4225 | 1845.00 | 1167.192/402.427/271.18849 | 2.4710 |
| baseline | 195 | 23.7644/27.9136/18.5350 | 2219.50 | 1118.491/377.650/246.57380 | 2.4309 |
| rotation | 10 | 0.1759/24.3604/19.1914 | 75.10 | 673.722/297.692/0.05535 | 1.8528 |
| rotation | 40 | 22.2506/27.3776/19.4921 | 626.25 | 1144.234/431.535/217.76436 | 2.3777 |
| rotation | 50 | 23.9476/26.8990/19.4255 | 854.95 | 1156.394/403.732/238.67591 | 2.3841 |

회전 epoch10의 X std 0.1759, λ3 0.05535와 상대적으로 낮은 rank는 예측 3D 점이 한 축에 거의 몰렸음을 보여준다. 같은 epoch의 Y/Z 표준편차는 24.36/19.19이므로 모든 좌표가 같은 상수라는 뜻은 아니다. epoch40과50에서는 X std가 22.25/23.95로 회복되고 unique 점 수도 증가했다. 그러나 40/50 모두 ADD0/20이라 분산 회복만으로 Pose가 정확하다고 판단하면 안 된다.

전체 Front/Back 및 프레임별 고유값, rank, PnP 반환·실패: [correspondence_geometry.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/correspondence_geometry.csv). 시각화: [loss_vs_xyz.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/figures/loss_vs_xyz.png), [checkpoint_pose_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/figures/checkpoint_pose_comparison.png).

## 6. 가설별 판정

| 가설 | 지지하는 관측 | 반박·제한 | 판정 |
|---|---|---|---|
| A. 회전 학습 자체가 충분히 진행되지 않음 | epoch10→40/50에서 X 다양성, XYZ L2와 PnP 반환이 개선 | epoch40/50도 ADD0/20; 더 긴 학습의 효과는 미실행 | 초기 단계 부족은 가능. 추가 학습만으로 해결되는지는 미확정 |
| B. 더 나은 좌표를 학습했지만 Val Loss가 잘못 선택 | 낮은 Val Loss의 epoch10은 X 붕괴·PnP0; epoch40/50은 더 낮은 XYZ와 PnP20 | 나중 모델 ADD도0/20, 선택 epoch가 다르면 학습 단계도 달라짐 | checkpoint 선택이 붕괴를 가린 증거는 강함. 선택 변경만의 Pose 개선은 미입증 |
| C. Predicted Mask Loss 게이팅으로 품질과 Loss가 분리 | epoch10 GT 전경 누락15.27%; 동일 가중치 GT gate 대조에서 epoch10/50 Loss 순서가 역전 | GT gate 대조는 학습 없이 고정된 history에서 계산. Mask 이외 코드 값·가중치도 달라짐 | 현재 checkpoint 선택 왜곡의 직접적인 실측 근거. 전체 학습 실패의 단독 인과는 미확정 |
| D. 회전 구현 또는 데이터 분포 문제 | 같은 Val에서 회전 epoch50 XYZ 55.24, Baseline epoch50 46.85; augmented 학습 분포가 변경됨 | 이전 0° 동일성, 382,368개 독립 최근접 표본의 불일치0, affine/HCCE CPU 검증 통과. 한 seed, optimizer skip도 1회 차이 | 단순 좌표 불일치 근거는 약함. 증강 분포의 실제 영향은 별도 통제 실험 필요 |

기존 Val20을 이미 학습 checkpoint 선택과 이번 진단에 사용했다. 새로운 선택 규칙을 이 결과에 맞춰 반복 설계하거나 이 수치를 독립 최종 테스트로 보고하지 않는다.

## 7. 다음 한 가지 제안과 범위

다음 대조에서 **checkpoint 선택 점수만 `0.5 × (Front XYZ L2 + Back XYZ L2)`로 변경**하고, 각 L2는 PBR Val의 GT Mask∩유효 label에서 계산한다. 기존 seed·데이터 분할·모델·optimizer·증강·학습 budget·PnP를 유지하고 변경 전 점수·선택 epoch를 사전 기록한다. GT는 PBR validation의 모델 선택에만 사용하며 실사27을 선택에 쓰지 않는다. 이 분석의 세 저장 회전 checkpoint에서 사후 최소 XYZ는 epoch40이지만 그 ADD가0/20이므로, 이 제안은 **개선 보장이 없는 가설**이다. 새 실험이나 코드 수정은 실행하지 않았다.

## 8. 산출물과 보존

| 산출물 | 내용 |
|---|---|
| [checkpoint_inventory.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/checkpoint_inventory.csv) | 실존9개와 별도 epoch0 참고1개, 파일 크기·해시·strict loading 상태 |
| [epoch_loss_history.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/epoch_loss_history.csv) | Train 전 epoch, 검증 기록 시점, AMP skip, 미기록 성분 N/A |
| [checkpoint_comparison.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/checkpoint_comparison.csv) | Val Loss 재계산·Mask·XYZ·Z0·PnP·ADD/RE/TE·GT gate 반사실 |
| [correspondence_geometry.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/correspondence_geometry.csv) | Front/Back 프레임별 std·unique·고유값·rank·PnP |
| [checkpoint_frame_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/tables/checkpoint_frame_metrics.csv) | 주9개 checkpoint의 프레임별 정확도 |
| [loss_vs_xyz.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/figures/loss_vs_xyz.png) / [checkpoint_pose_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/figures/checkpoint_pose_comparison.png) | 실제 저장 시점에 한정한 발표용 그림 |
| [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/checkpoint_diagnostic_v1/checks/final_integrity.json) | 기존 자산·checkpoint 보존 SHA-256 검사 |

기존 원본740파일, 세미나366파일, Baseline61파일, Pose진단102파일, Train/Val진단63파일, 자율실험146파일 manifest를 모두 재검증해 일치했다. 10개 checkpoint의 평가 전후 해시도 일치했다. 기존 전용 venv를 재사용했고 패키지 설치·외부 다운로드·실사27 사용은 없었다. 종료 GPU compute 조회: `실행 중인 compute process 없음`.

## 9. 5줄 핵심 결론

1. 회전 last epoch50의 XYZ는 best epoch10보다 낮은 오차를 보이지만 ADD는 둘 다0/20이다.
2. 선택된 epoch10의 X축 붕괴는 epoch40·50에서 완화됐으므로 Val Loss는 좌표 붕괴를 가렸다.
3. GT Mask 게이트를 적용한 고정 가중치 반사실에서 epoch10·50의 Loss 순서가 뒤집혔다.
4. 재학습 없이 저장 checkpoint로 PnP 반환은 회복할 수 있지만 Pose 정확도 해결은 확인되지 않았다.
5. 다음에는 PBR GT 유효 Mask의 Front/Back XYZ L2만 checkpoint 선택 점수로 바꾸는 대조를 사전 고정한다.
