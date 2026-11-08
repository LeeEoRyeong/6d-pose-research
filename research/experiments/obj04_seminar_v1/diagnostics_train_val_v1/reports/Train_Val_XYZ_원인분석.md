# Train80 / Val20 XYZ 학습 실패와 일반화 실패 원인 분석

작성: 2026-11-03T22:49:21.854026+09:00

**결론: 현재 checkpoint는 Train 좌표를 상당히 학습했지만, 분리된 Val 블록에서 좌표 오차가 급증한다. 전반적 XYZ 학습 실패보다 과적합·합성 블록 일반화 실패의 증거가 강하며, Val Z plane0의 coarse 오류가 특히 크다.** 다만 작은 Train 잔차와 fine-plane 학습 문제도 남아 있다.

- 새 결과: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1`
- 모델: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt`, epoch155, 재학습 없음
- 주 조건: Train80/Val20 모두 batch4, CUDA AMP float16, model.eval/inference_mode
- GT BBox oracle crop·BGR 정규화·기존 XYZ 라벨·원본 HCCE 복호화 동일
- 계산2회 합계 약43.67초(첫 sign-threshold 대조 포함); 보고서 생성까지 전체 작업 약13.01분 /45분. 계산 wall time은 로딩·CPU 집계를 포함하며 GPU kernel 시간은 아니다.
- 실제 SMT27·YOLO·학습 optimizer는 사용하지 않았다. 기존 전용 venv 재사용, 설치·다운로드 없음.

## 1. 요청 질문에 대한 답

1. **어느 쪽 XYZ 오류가 큰가:** Val이 크다. Front L2 Train4.2057→Val29.4046(6.99배), Back4.5603→29.1845(6.40배)다.
2. **Train에서도 Z 오류가 심각한가:** Train Front/Back Z MAE1.3698/1.5070으로 Val16.3539/15.7365보다 훨씬 작다. coarse Z0 오류도 Train2.75%/3.57% 대 Val44.57%/43.12%다. Train fine Z planes5–7에는 약40–50% 오류가 남지만 영향 크기가 다르다. Train이 완벽하다는 뜻은 아니다.
3. **큰 좌표 오차에 기여하는 plane:** Val Z plane0이다. 이 한 plane만 GT로 치환한 Oracle에서 Front Z MAE reference 16.3542→3.2228, Back 15.7367→2.9552다. 실제 모델 출력의 개선 수치가 아니다.
4. **학습 부족과 과적합 중 증거가 강한 쪽:** 현재 조건에서는 과적합·블록 일반화 격차 쪽이다. Train Front ADD59/80 대 Val1/20, Train Z coarse 오류가 낮다. fine-plane 잔차가 있어 학습 부족 가능성 전체를 배제하지는 않는다. 한 checkpoint 비교로 epochs·Loss·초기화 중 인과 원인을 확정할 수 없다.
5. **다음 실험에서 딱 하나 바꿀 것:** **학습 crop의 동기화된 in-plane 회전 증강만 추가한다(고정 정책: uniform[-20°,20°]).** RGB·Mask·Front/Back XYZ·유효영역에 같은 affine를 적용하고 label은 nearest, HCCE는 변환된 XYZ에서 재생성한다. 나머지 초기화·Loss·분할·seed·step budget·평가·선택 규칙은 유지한다. 미실행 가설 검증이며, 3D viewpoint 일반화를 보장하지 않는다.
6. **내일 확정적으로 말할 결론:** “좌표는 Train에서는 상당히 맞지만 Val 블록에서 크게 틀리며, 특히 Z coarse plane 일반화가 나쁘다. Mask 양호만으로 Pose 일반화를 판단할 수 없다.” 독립 테스트 정확도·실사 정확도·개선 모델 확보는 주장하지 않는다.

## 2. 동일 조건 및 측정 정의

| 항목 | 실제 설정 |
|---|---|
| checkpoint | 기존 random-initialized ResNet34 학습의 epoch155, strict state loading |
| Train / Val | 80/20, Train블록0,1,2,3,5,6,7,8; Val블록4,9. 프레임·블록 교집합 없음 |
| crop | 원본 `pad_square_fp32` pad1.5, `crop_square_resize`, RGB256, Mask/XYZ128 |
| RGB | BGR, mean(0.406,0.456,0.485), std(0.225,0.224,0.229), 학습과 동일 |
| 좌표 | object model XYZ, min/size 동일. PBR 원본 단위, 절대 물리치수 독립 계측 없음 |
| 주 오차 영역 | `GTmask & Front/Back nonzero-label validity`, 양 branch 동일 |
| 보조 영역 | PredMask∩유효label, GT/Pred 교집합, GTmask3×3 erode; 동일 CSV에 보존 |
| Mask IoU | 전체 crop의 GTmask 대 predmask. 좌표 유효영역으로 IoU를 유리하게 제한하지 않음 |
| XYZ | 실제 원본 AMP decoder 출력 대 동일픽셀 GT 양자화 XYZ |
| raw target | 네트워크 raw code 출력 대 `2*hcce_encode-1` MAE, 총 Loss 아님 |
| hard bit | 원본과 동일 FP16 sigmoid>0.5 대 GT encode>0.5 |
| Pose | 실제 원본 RANSAC EPNP 반환,150iterations,1.5 원본px, seed42+frame, branch별 PnP |
| ADD | CAD1439점, threshold13.9378694855, 모든프레임 분모, GT행렬 복사 안 함 |
| 통계 | 주 표는 프레임별 평균의 split 평균. pixel-weighted 열도 별도 보존 |

GT BBox ROI는 모든 조건의 공통 Oracle 입력이다. 예측XYZ·예측Mask로 구한 Pose는 **oracle ROI 조건의 모델 결과**이며 E2E 검출 성능이 아니다. GT bit 치환은 추가로 GT를 쓴 **Oracle 민감도 진단**이다. Train 결과는 학습 적합도이며 일반화 성능으로 보고하지 않는다.

근거: [config.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/checks/config.json), [split.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/checks/split.json), `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/scripts/measure_source_threshold.py`, 이전 학습 `scripts/support.py`, 원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`, `bop_loader.py`, `PnP_solver.py`.

## 3. Train / Val XYZ·Mask·raw 비교

| split | branch | Mask IoU | XYZ L2 | X MAE | Y MAE | Z MAE | raw-target MAE | hard bit 오류 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | front | 0.973965 | 4.2057 | 2.2657 | 1.5855 | 1.3698 | 0.299721 | 18.60% |
| train | back | 0.973965 | 4.5603 | 2.1666 | 2.0036 | 1.5070 | 0.301414 | 18.65% |
| val | front | 0.935782 | 29.4046 | 12.3937 | 8.6497 | 16.3539 | 0.468275 | 29.19% |
| val | back | 0.935782 | 29.1845 | 11.1159 | 10.2033 | 15.7365 | 0.468172 | 28.74% |

GTmask 유효 픽셀은 Train 261,419개, Val 64,635개이다. Front Z MAE는 Val이 Train의11.94배, Back은 10.44배다. Mask IoU 차이에 비해 좌표 격차가 크다. raw MAE는 회귀 출력의 오차이고, hard bit 오류는24개 plane의 동일 가중치 평균이다. 저가중 fine plane 오류가 커도 전체 XYZ에 미치는 영향은 coarse 오류보다 작을 수 있다.

보조 Mask 영역에서도 같은 비교를 모두 저장했다. GT 없는 background 좌표를 임의로 정답화하지 않고 유효label과 교집합했다. GTmask의 명시적 무효 label은 좌표오차 계산에서 제외하고 프레임별 개수를 기록했다.

그림: [train_val_xyz_error.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/train_val_xyz_error.png), [mask_vs_xyz_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/mask_vs_xyz_comparison.png).

## 4. 실제 Train / Val Pose

| split | branch | 유효 PnP | ADD 통과 | ADD 평균/중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 비율 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | front | 80/80 | 59/80 (73.75%) | 11.9983/9.4924 | 3.4719 | 12.1574 | 13.42% |
| train | back | 80/80 | 65/80 (81.25%) | 7.9234/5.3821 | 2.3878 | 7.8404 | 13.92% |
| val | front | 20/20 | 1/20 (5.00%) | 62.2474/63.5035 | 81.0261 | 60.9094 | 4.01% |
| val | back | 20/20 | 1/20 (5.00%) | 74.2090/63.2314 | 83.2968 | 75.8861 | 4.19% |

100장×2branch 모두 실제 PnP를 실행했다. 유효 반환은 모두100%였지만 정확도는 크게 달랐다. Front Train ADD59/80=73.75%, Val1/20=5%다. Back Train65/80=81.25%, Val1/20=5%다. Back 모델을 GT로 프레임별 선택하거나 더 높은 Train 수치를 일반화 정확도로 보고하지 않는다.

원본 Back PnP의 reported inlier는 Front로 재집계되는 기존 문제를 유지하고 로그에 남겼다. 표는 원본solver에 실제 입력된 branch 3D점 기준으로 독립 재투영한 inlier 비율이다. 원본 파일은 수정하지 않았다. RANSAC raw inlier와 반환 후 재집계 inlier도 다를 수 있다.

Val20의 Front ADD·XYZ는 이전 진단과 프레임별 최대차이0으로 재현됐다. 이번 Train Mask 수치는 저장 checkpoint의 새 추론에서 직접 측정한 값이다.

## 5. HCCE plane 순서·가중치·민감도

원본 `hcce_decode`의 축별 순서는0→7이다. HCCE hard plane을g, 복원 binary를b라 하면:

```text
b0 = g0
bk = gk XOR b(k-1)
q = 128*b0 + 64*b1 + 32*b2 + 16*b3 + 8*b4 + 4*b5 + 2*b6 + b7
XYZ_axis = q/255 * size_axis + min_axis
```

따라서 gk 하나의 flip은 binary k..7에 전파될 수 있다. 복호화 자리 가중치128..1을 그 HCCE plane의 독립적인 오차 가중치로 간주하면 안 된다. 이론상 최대 code 변화는 `2^(8-k)-1`이다. Z에서 plane0 최대65.3153단위 대 plane7 최대0.2561단위이며, 실제 영향은 표본에 따라 달라진다.

### Z plane0~7 전체 비교

| branch | plane | binary 자리 / 최대cascade(code) | Train오류 | Val오류 | GT=1 비율 T/V | Pred=1 비율 T/V | Val GT-plane치환 MAE 감소 |
|---|---:|---:|---:|---:|---:|---:|---:|
| front | 0 | 128 / 255 | 2.75% | 44.57% | 51.8/50.1% | 50.3/54.3% | 13.1314 |
| front | 1 | 64 / 127 | 3.09% | 12.36% | 50.7/50.4% | 50.2/48.2% | 0.6621 |
| front | 2 | 32 / 63 | 5.69% | 17.12% | 50.8/50.4% | 49.1/45.4% | 0.2723 |
| front | 3 | 16 / 31 | 9.70% | 23.67% | 51.4/51.4% | 50.0/47.4% | 0.1839 |
| front | 4 | 8 / 15 | 18.50% | 34.86% | 50.7/50.8% | 48.9/47.2% | 0.1453 |
| front | 5 | 4 / 7 | 44.83% | 49.36% | 50.5/50.5% | 46.9/53.0% | 0.1445 |
| front | 6 | 2 / 3 | 48.16% | 49.49% | 49.7/49.8% | 43.1/37.6% | 0.0532 |
| front | 7 | 1 / 1 | 49.64% | 50.41% | 49.8/50.4% | 2.8/0.2% | 0.0219 |
| back | 0 | 128 / 255 | 3.57% | 43.12% | 49.6/50.9% | 47.4/47.3% | 12.7815 |
| back | 1 | 64 / 127 | 3.68% | 11.19% | 50.5/50.4% | 50.8/49.4% | 0.4525 |
| back | 2 | 32 / 63 | 6.15% | 15.84% | 50.6/50.4% | 49.7/45.5% | 0.2824 |
| back | 3 | 16 / 31 | 10.02% | 22.38% | 51.0/51.2% | 50.7/49.6% | 0.2007 |
| back | 4 | 8 / 15 | 17.62% | 34.47% | 50.7/50.6% | 47.8/45.1% | 0.1322 |
| back | 5 | 4 / 7 | 40.23% | 47.44% | 50.9/50.8% | 52.7/53.9% | 0.0913 |
| back | 6 | 2 / 3 | 48.43% | 49.30% | 49.7/49.7% | 41.3/34.5% | 0.0450 |
| back | 7 | 1 / 1 | 49.26% | 49.89% | 50.0/50.0% | 35.1/34.5% | 0.0123 |

감소량은 원본과 정수 복호화가 일치하는 FP32 reference에서 계산했다. 한 plane의 GT hard bit만 치환하고 나머지는 예측을 유지했다. 이 감소량을 실제 모델 성능이나 그 plane의 독립적 인과 기여율로 해석하지 않는다. plane별 감소량은 상호작용 때문에 단순 합산하지 않는다.

Front Val Z plane0 치환 reference MAE 감소는13.1314(80.3% 감소)이나 plane7은0.0219뿐이다. **오류율이 약50%인 fine plane과 약45%인 coarse plane을 동일한 Pose 병목으로 취급하면 안 된다.**

Train의 coarse Z0는 낮은 오류지만 Z5–7은 여전히 약40–50%이다. 이 fine-scale 학습 잔차는 존재하며, Train이 완전히 학습됐다는 주장을 제한한다. Val의 큰 Z 오차를 가장 강하게 설명하는 민감도는 plane0에서 보인다.

그림: [z_plane_error_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/z_plane_error_comparison.png).

### 같은 coarse plane의 X/Y 비교

| branch | axis plane0 | Train 오류 | Val 오류 |
|---|---|---:|---:|
| front | X | 2.88% | 15.28% |
| front | Y | 1.88% | 9.58% |
| front | Z | 2.75% | 44.57% |
| back | X | 2.67% | 12.07% |
| back | Y | 2.09% | 11.64% |
| back | Z | 3.57% | 43.12% |

Val의 coarse 오류는 Z에서 특히 크며 X/Y도Train보다 증가한다. Z만의 구현 오류가 확정됐다는 뜻은 아니다.

### GT/Pred 비율·margin·한쪽 출력

| split | branch Z0 | mean abs raw margin | mean sigmoid | heuristic confidence |
|---|---|---:|---:|---:|
| train | front | 0.4981 | 0.5004 | 0.6196 |
| train | back | 0.4975 | 0.4878 | 0.6194 |
| val | front | 0.4315 | 0.5068 | 0.6040 |
| val | back | 0.4269 | 0.4857 | 0.6028 |

Z0의 GT/Pred=1 비율은 대체로 50% 부근인데 Val 오류율은 약 44%이다. 즉 coarse plane의 **전역적인 한쪽 출력 붕괴보다 이미지별/픽셀별 대응의 일반화 실패**가 관측된다. sigmoid confidence는L1 code 회귀 출력에서 계산한 heuristic일 뿐 보정된 확률이 아니다. margin이 남아 있어도 정확한 bit라는 보장은 없다.

반면 Front Z7은 Val Pred=1이0.215%로 한쪽 출력이 강하다. Train은2.816%이다. 한쪽 출력 flag는 사전 정의한 pooled Pred≤1% 또는 ≥99% 기준이며, GT 비율도 함께 확인했다. 이 fine-plane 현상을 coarse Z0의 실패와 혼동하지 않는다. 모든 축·plane의 GT/Pred 비율, raw mean/std, margin, confidence, TP/TN, balanced accuracy, 프레임별 한쪽 출력 수는 CSV에 있다.

## 6. FP16 임계값 정밀도 대조

첫 측정은 기존 진단의 `raw>0` hard-bit 표현을 사용했다. 정밀도 점검에서 FP16 sigmoid가 0.5로 round되는 작은 양수는 원본의 `sigmoid(raw)>0.5`와 다를 수 있음을 확인했다. 원본 dtype/threshold로 재측정한 결과를 **현재 루트의 주 CSV**로 게시했다.

GTmask 유효영역의 총15,650,592bit 비교 중 13,454개(0.0860%)가 sign 대 실제 threshold에서 달랐다. 첫 결과는 `sign_threshold_control/`에 보존했다. 이 차이 때문에 앞선 진단의 Z0 약44.55%가이번 실제threshold에서44.57%로 정밀 보정되지만, 실제 AMP XYZ·Pose는 처음부터 원본 출력이므로 이전 진단 대비 최대차이0이다.

수정된 실제 hard bit와 원본 torch decoder의 정수 출력은 최대차이0이다. 이 reference의 XYZ와 원본 AMP XYZ 원소 최대차이는0.022008로, Val의29단위 오차를 설명하는 크기가 아니다. 첫 sign-threshold 대조의 큰 최대차이를 순수 AMP 좌표 연산 오차로 잘못 해석하지 않는다.

이 정밀도 대조는 새 증거를 보존하기 위한 것이며 소스·학습·checkpoint 수정이 아니다. 근거: [precision_control.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/tables/precision_control.csv), [publishing_policy.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/checks/publishing_policy.json).

## 7. 가설 판정

| 가설 | 판정 | 실제 증거 | 한계 |
|---|---|---|---|
| A Train/Val 모두 큰 XYZ 학습 실패 | 주 설명으로는 지지 약함, 세밀한 plane 학습 잔차는 확인 | Train Front L2=4.2057, Z MAE=1.3698; Val은 6.99/11.94배 | Train Front ADD도59/80으로 완전 적합은 아니며 fine plane 오류 남음 |
| B 과적합·블록 일반화 실패 | 현재 네 가설 중 가장 강한 지지 | Train Front ADD59/80 대 Val1/20, 두 Val 블록 모두 모든 Train블록보다 Front XYZ오차 큼 | 단일 seed/checkpoint·Val2블록. 원인이 초기화/증강/표본량 중 무엇인지는 미확정 |
| C Mask·XYZ 학습 불균형 | 현상은 지지, 최적화 불균형 원인은 미확정 | Mask IoU 0.9740/0.9358지만 XYZ gap 6.99배 | loss branch gradient/scale 실험 없음. Train coarse 좌표는 학습됨 |
| D Z coarse plane 민감도 | Val에서 강하게 지지; Train coarse는 상대적으로 양호 | Front Z0 hard error 2.75%/44.57%; GT plane0치환 Val Z MAE 약3.22 | Oracle 치환 진단이며 실제 모델 개선 아님; fine plane도 미학습 잔차 |

**현재 가장 강한 판정은 B와 D의 조합**이다. C는 출력 품질 격차의 현상으로 관측됐지만 gradient 경쟁이나 Loss 불균형이 원인인지 측정하지 않았다. A의 전체적인 학습 실패 설명은 약하지만, fine plane 잔차와 일부 Train Pose 실패는 남는다.

## 8. 가벼운 추가 진단

### 생성 블록별 Front 결과

| block | split | XYZ L2 | Z MAE | Mask IoU | ADD 통과 |
|---|---|---:|---:|---:|---:|
| 0 | train | 3.2318 | 1.0936 | 0.9793 | 8/10 |
| 1 | train | 4.1667 | 1.4481 | 0.9648 | 6/10 |
| 2 | train | 4.6811 | 1.1945 | 0.9815 | 8/10 |
| 3 | train | 3.1682 | 1.1861 | 0.9823 | 8/10 |
| 4 | val | 27.7782 | 20.5606 | 0.9613 | 1/10 |
| 5 | train | 3.2903 | 1.1038 | 0.9780 | 8/10 |
| 6 | train | 6.8858 | 1.9994 | 0.9793 | 8/10 |
| 7 | train | 4.0771 | 1.4882 | 0.9540 | 6/10 |
| 8 | train | 4.1449 | 1.4446 | 0.9724 | 7/10 |
| 9 | val | 31.0311 | 12.1472 | 0.9102 | 0/10 |

Train 블록의 Front 평균 오차 범위는 3.1682~6.8858인 반면 Val 블록4는 27.7782, 블록9는 31.0311이다. 한두 Val 프레임만의 문제로 설명되지 않는다. 블록4의 Mask IoU 0.9613도  XYZ 일반화를 보장하지 못했다. 같은 블록의 10프레임은 상관된 표본이므로, 100개 독립 장면으로 해석하지 않는다.

GT XYZ 분포는 같은 64개 model-axis bin을 사용해 분할별 합이 1이 되도록 정규화했다. Z0 GT positive 비율은 Train/Val 모두 대체로 50%여서 단순한 클래스 불균형으로 coarse 격차를 설명하기 어렵다. **marginal 분포가 비슷하더라도 RGB-좌표의 조건부 분포/시점 차이는 배제할 수 없다.** XYZ 축별 분포·TV·JS 요약은 CSV에 보존했고, 정식 두 분포 검정이나 인과 판정으로 사용하지 않았다.

GT object-Z와 camera-Z 각도 범위는  Train58.35~125.49deg, Val60.32~121.18deg로 겹친다. 이것만으로 전체 6D Pose 분포가 같다고 할 수 없다. Front XYZ 오차와 이 각도의 Pearson r은 Train -0.0391, Val 0.0833이다. 가시율은 모두 약 0.997~1.0으로 변동이 작아 가림 효과를 검증하기에 부족하다.

Front XYZ 오차와 PnP 실제 inlier 비율의 r은 Train -0.4362, Val -0.3354이며, ADD와는 Train 0.3292, Val 0.7858이다. 같은 생성 블록 안에서 상관된 표본의 기술통계다. 유효 PnP 반환 실패는 0이었으므로, 여기서는 ADD 기준 미달을 별도로 구분한다. ADD 통과/실패별 좌표·inlier 집계도 추가했다.

그림: [gt_z_histogram_comparison.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/gt_z_histogram_comparison.png), [block_xyz_error.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/block_xyz_error.png). 근거: `tables/GT_distribution_comparison.csv`, `tables/GT_distribution_per_frame.csv`, `tables/relationships.csv`, `tables/ADD_pass_failure_comparison.csv`.

## 9. 다음 실험에서 한 가지 변경

**선택: 학습 crop에 동기화된 in-plane 회전 증강만 추가한다.** 현재 Baseline에 증강이 없고 블록 일반화 격차가 크다는 관찰에 근거한 다음 대조 실험 제안이다. 개선 효과는 미확인이다.

- 고정 정책: Train crop의 회전각 uniform[-20°,20°], seed42. RGB는 bilinear, Mask·XYZ·유효 support는 nearest로 변환하고 HCCE는 변환 후 XYZ에서 생성한다. 모델 좌표값 자체를 회전하지 않고 픽셀 위치만 동기 변환한다.
- 유지할 설정: Train80/Val20 및 블록, random-initialized ResNet34, batch4/crop256, Adam lr0.0002, 최대195epoch/3900step, 기존 Loss 수식·타입 호환, epoch≥1의 Val Loss 체크포인트 선택 규칙. Validation에는 증강을 적용하지 않고 PnP 조건도 유지한다.
- 평가: 기존 Front 주지표와 Z0 bit/XYZ 오차를 함께 보고한다. Train 적합도 변화도 기록하되, 같은 Val을 재사용하므로 독립 테스트 성능이라고 부르지 않는다.
- 사전 확인: CPU에서 RGB·Mask·XYZ에 동일 affine가 적용되는지 검증한다. 증강한 영상에서 PnP를 평가한다면 K도 동일 affine로 변환해야 한다. 원본 GT는 수정하지 않는다.
- pretrained·Loss 가중치·epochs·PnP threshold를 동시에 바꾸지 않는다. 회전 증강은 3D out-of-plane 시점 변화를 완전히 만들지 못하므로, 개선이 없으면 그대로 보고한다.

이 제안은 이번에 실행하지 않았다. 기존 Back/Mixed PnP inlier 집계의 정식 수정은 별도 회귀 작업으로 분리한다. Front-only 주평가와 무관한 수정을 이번 한 변수 비교에 섞지 않는다.

## 10. 산출물·보존·재현

| 파일 | 내용 |
|---|---|
| [train_val_xyz_comparison.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/train_val_xyz_comparison.csv) | Train/Val·Front/Back·Mask영역별집계, 주표프레임평균과pixel가중치구분 |
| [train_val_pose_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/train_val_pose_metrics.csv) | 실제200회PnP의split·branch집계 |
| [hcce_plane_error.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/hcce_plane_error.csv) | 96개split/branch/axis/plane집계 |
| [xyz_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/tables/xyz_per_frame.csv) | 프레임별좌표·raw·bit오차 |
| [plane_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/tables/plane_per_frame.csv) | 프레임별plane비율·margin·Oracle민감도 |
| [pose_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/tables/pose_per_frame.csv) | 실제PnP오차·실패원인·inlier수 |
| [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_train_val_v1/checks/final_integrity.json) | 이전자산SHA-256보존확인 |

보존 대조: 원본 740파일, 이전 세미나 산출물 366파일, Baseline 61파일, 기존 `diagnostics_v1`의 102파일이 모두 기존 manifest의 SHA-256과 일치했다. 체크포인트 해시도 실행 전후 동일했다. 종료 시 GPU compute process는 없었다. 다른 사용자의 compute PID가 보이면 중단하는 guard를 배치 전과 프레임 처리 중간에 적용했다.

이전 `diagnostics_v1/scripts/diagnose.py`와 학습 `scripts/support.py`의 검증된 처리를 재사용했다. 이번 측정 진입점은 `scripts/measure_source_threshold.py`, 그래프·보고서 생성은 `scripts/build_reports.py`이며 실행 원문은 각 `logs/`에 보존했다. 재현 시 새 결과 루트로 절대 경로를 갱신한다. 현재 출력이 있는 경로에 그대로 재실행하지 않는다.

미수행: 재학습, YOLO, 실사27 평가·튜닝, 새로운 seed, Loss gradient 실험, 독립 테스트, 실제 회전 증강 학습. 새로운 모델의 성능 향상은 주장하지 않는다.
