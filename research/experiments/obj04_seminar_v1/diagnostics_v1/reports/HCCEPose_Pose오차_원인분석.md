# HCCEPose SMT obj04 Pose 오차 원인 분석

작성: 2026-11-03T22:28:44.886424+09:00

**결론: GT XYZ→PnP는 Val20 전체에서 통과했고, 큰 오차는 네트워크 예측 XYZ를 사용했을 때 발생했다. 현재 가장 강한 병목 증거는 좌표 예측이며, 특히 Z 축이다.** Mask만 GT로 치환해도 ADD는1/20에 머물렀고, XYZ를 GT로 치환하면 기존 예측 Mask로도20/20이었다. GT를 사용한 결과는 모두 **Oracle 진단**이며 실제 모델 성능이 아니다.

- 진단 루트: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1`
- 기존 모델: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt` (epoch155)
- 원본 소스: `/RESEARCH_DATA/HCCEPose` — 수정 없음
- 데이터: `/RESEARCH_DATA/HCCEPose_experiments/realistic_final100/train_pbr/000000`
- 기존 venv 재사용, 추가 설치·다운로드·재학습 없음. GPU에서는 제한된 checkpoint 추론만 수행했다.
- 수치·CSV·그림은 이번 신규 진단 결과다. 기존 Baseline 학습·실사27 검증 수치는 구분해 재사용했다.
- 보고서 생성 시 계산 시작 이후 약11.54분. 사용자 요청 접수부터 포함한 전체 작업도60분 이내이며 이후 신규 실험을 종료한다.

## 1. 질문별 답변

1. **오차가 처음 발생한 단계:** A0의 수치 오차 후 A1의8bit 양자화/라벨 표본화에서 작은 비영 오차가 처음 확인된다. A2까지는 ADD20/20을 통과했다. **실용적으로 큰 Pose 오차는 예측 HCCE를 복호화한 XYZ를 사용할 때 발생했다.**
2. **정답 XYZ에서도 실패하는가:** 이번 GT Front full-resolution·crop 및 GT Back crop은 모두 유효 PnP20/20, ADD20/20이었다. 모든 모델·카메라·형상에서 성공한다는 보장은 아니다.
3. **예측 XYZ 오차:** GTmask 내부 프레임별 평균 L2의20프레임 평균은 Front29.4046, Back29.1845(PBR 원본 단위). GT 양자화 좌표와 같은128×128 픽셀에서 비교했다.
4. **가장 큰 병목:** 예측 좌표/대응점 품질. Front Z MAE16.3539, 첫 Z HCCE plane 오류율44.55%. GT Z만 치환한 Oracle ADD10/20은 Z의 중요성을 지지한다. Y 치환도7/20으로 올라 복합 오류임을 함께 보여준다.
5. **다음 실험의 첫 조치:** 추가 학습 전 Train80에도 같은 좌표/bit 진단을 적용해 학습 자체의 실패와 Val 일반화 실패를 분리한다. 구현 수정 우선순위는 작업 복사본의 PnP inlier를 실제 사용한 `Points_3D`로 일관되게 계산하고 GT/Back/Mixed 회귀검증하는 것이다. 이 수정만으로 Front Baseline의 정확도가 개선된다고 주장하지 않는다.
6. **발표 가능한 결론:** “Mask IoU0.9358이어도 좌표 예측 오차가 크면 ADD5%에 머문다. GT 좌표 Oracle은100%여서 이번 Val20의 주요 오차를 좌표 예측 단계로 좁혔다.” 실사 정확도나 개선 모델 성능 확보 주장은 할 수 없다.

## 2. 원본 알고리즘과 좌표 규약

| 항목 | 실제 규약·근거 |
|---|---|
| Front / Back | 카메라 ray의 최소/최대 표면 깊이. 원본 렌더러 GL_LESS/GL_GREATER 의미, `HccePose/bop_loader.py:rendering_bop_dataset_back_front` |
| 3D 좌표 | object-to-camera `Xc=R Xm+t`; 역변환 행벡터 `(Xc-t)@R` |
| 정규화 | `(Xm-min_xyz)/size_xyz`, clip[0,1], round×255, uint8. min=(-45.25,-45.25,-32.65763855), size=(90.5,90.5,65.31527710) |
| 채널 | cv2 읽기/쓰기 배열의 channel0=X,1=Y,2=Z. PNG viewer의 일반 RGB 표시와 혼동하지 않음 |
| HCCE | 축당8개 soft plane,3축24개, Front/Back 총48개. 원본 `train_bop_dataset_back_front.hcce_encode` |
| 복호화 | soft GT를>0.5로 이진화 후 원본 `HccePose_BF_Net.hcce_decode`, 누적 절댓값 차분×2의 거듭제곱. `/255*size+min`으로 모델 좌표 복원 |
| 예측 경로 | 원본 `inference_batch` sigmoid threshold0.5. 모델의 raw 회귀 출력과 threshold 후 bit·3D는 서로 다른 평가 대상 |
| 영상·K | PBR1280×720, 무왜곡, 프레임별 `scene_camera.json` K. 원본 K를 사용하며 실사 K/newK 가정은 이번 진단에 사용하지 않음 |
| Crop | GT visible BBox pad1.5, RGB256, code/Mask128. 원본 affine crop과 nearest label 표본화 |
| 원본 좌표 복원 | `u=x0+j*w/128`, `v=y0+i*h/128`, 원본 `inference_batch`와 동일. 이 좌표에 원본 K 적용 |
| 단위·지표 | PBR 원본 좌표 단위; provenance에 mm→m0.001 기재, 실제 물리 치수 독립 계측 없음. diameter139.378694855, ADD 임계13.9378694855 |

`hcce_encode`의 soft target `2*code-1`과 네트워크 raw 출력의 MAE, threshold 후 hard-bit 오류, 복호화 XYZ의 L2/축 MAE를 분리했다. 총 학습 Loss와 혼동하지 않는다. HCCE plane은 독립적인 일반 binary digit이 아니며 앞 plane 오류는 뒤 복호화에도 영향을 줄 수 있다.

## 3. 실험 A — GT Oracle Round-trip

네트워크 예측을 사용하지 않았다. 모든 R/t는 실제 원본 `solve_PnP` 호출 반환값이다. GT 행렬을 반환값으로 복사하지 않았다.

- **A0:** CAD1439개 꼭짓점을 GT R/t/K로 정확히 투영하고 2D–3D를 PnP에 제공. 가려진 CAD 점도 포함한 기하 Oracle이다.
- **A1:** 기존 full-size Front PNG의 유효 GT visible 픽셀을 원본 HCCE로 encode→threshold→decode한 후 정수 원본 픽셀과 PnP. 라벨의 명시적 zero-invalid는 제외했다.
- **A2:** 기존 학습과 같은 crop·128출력·원본 좌표 복원. `GTmask`는 baseline과 같은 Mask support, `valid`는 추가로 라벨 유효 영역과 교집합한 대조다.
- **nearest control:** crop이 실제 선택한 원본 nearest 픽셀 좌표를 대응점으로 사용해 재표본화 위치의 작은 차이를 분리했다.

| 경로 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
| A0_exact_CAD | 20/20 | 20/20 (100%) | 1.96206e-05 / 1.12479e-05 | 0.0144 | 1.89461e-05 | 1439.00 / 100.00% |
| A1_full_HCCE | 20/20 | 20/20 (100%) | 0.164691 / 0.114441 | 0.0450 | 0.168892 | 22101.60 / 100.00% |
| A2_crop_GTmask | 20/20 | 20/20 (100%) | 0.288577 / 0.152482 | 0.0631 | 0.292099 | 3231.75 / 99.46% |
| A2_crop_valid | 20/20 | 20/20 (100%) | 0.269247 / 0.127755 | 0.0618 | 0.272972 | 3231.75 / 100.00% |
| A2_nearest_pixel_control | 20/20 | 20/20 (100%) | 0.148635 / 0.0996659 | 0.0444 | 0.15243 | 3231.75 / 100.00% |

A1에서 HCCE decode uint8와 원본 label의 차이는20장 모두0이다. 축별0..255 exhaustive 검사도 최대 오차0이다. float32/float64 나눗셈 차이로 직접 XYZ 복원과의 최대 차이는2.6865333e-6(PBR 단위)이며 정수 복호화는 정확했다.

평균 correspondence reprojection error는 A0=0.00001799px, A1=0.222280px, A2 GTmask=1.175377px, A2 valid=0.429055px, nearest control=0.222400px다. 이는 **PnP에 제공한 3D 대응점을 반환 Pose로 투영한 residual**이며 실제 이미지의 CAD contour 정합성 오차와 다르다.

### 라벨·crop 추가 대조

- GT Back crop도 ADD20/20, 평균ADD0.373703, 평균RE0.101927deg였다.
- GTmask crop 내 명시적 무효 라벨은20장 합계356pixel이다. 제거하면 A2 mean residual이1.175377→0.429055px로 줄지만 제거 전에도 ADD20/20이다. 작은 경계 잔차를 숨기지 않는다.
- crop-K 대조는 `Kcrop=A K`, `A=[[128/w,0,-x0*128/w],[0,128/h,-y0*128/h],[0,0,1]]`를 사용했다. RANSAC threshold도1.5 원본px→`1.5*128/w` crop px로 변환했다. ADD20/20, 평균ADD0.290105. 중간 보조 대조의1.5 crop-px 고정 결과는 임계값 단위가 달라 별도 폴더에 남겼고 주 비교에는 사용하지 않는다.
- 독립 Moller–Trumbore ray20개/프레임, 총400개를 CAD 삼각면과 교차시켰다. 같은+0.5 ray의 최소/최대 깊이와 양자화 Front/Back을 비교한 camera-Z 평균 오차는0.083362/0.090886, 전체 최대는0.238918/0.277384다. 내부 표본만 검사했으며 전체 표면 검증은 아니다.
- A0의 raw trace RE 약0.01439deg는 float32 GT R이 엄밀한 SO(3)가 아닌 영향이 포함된다. GT R을 SVD로 SO(3)에 투영한 **별도 수치 대조**의 평균 RE는0.000002127deg다. 원본 GT는 수정하지 않았다.
- CPU renderer와 원본 VisPy의 bitwise 비교, 실사 카메라·CAD 물리 정합성은 이번에도 미완료다. A0 자기일관성만으로 실물 GT 정밀도를 인증하지 않는다.

근거: [GT_XYZ_Oracle_PnP.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/GT_XYZ_Oracle_PnP.csv), [GT_HCCE_Roundtrip.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/GT_HCCE_Roundtrip.csv), [hcce_exhaustive.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/checks/hcce_exhaustive.json), [independent_CAD_ray_check.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/supplement/tables/independent_CAD_ray_check.csv), [cropK_equal_threshold_Oracle.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/cropK_equal_units/tables/cropK_equal_threshold_Oracle.csv). 프레임별 K·BBox·support는 [geometry.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/checks/geometry.json).

## 4. 실험 B — 예측 HCCE와 복호화 XYZ 오차

**주 결과는 기존 Baseline과 동일 batch4·CUDA AMP float16이다.** epoch155 가중치 로딩 후 eval/inference_mode만 사용했다. same affine·nearest crop·same min/size·same128×128 index에서 GT/예측을 비교했다. Front와 Back은 각자 GT branch와 비교했다.

GT XYZ는 기존 양자화 라벨의 모델 좌표이고 예측 XYZ는 원본 decoder 출력이다. 명시적 무효 GT label pixel은3D 오차 계산에서 제외하고 n을 CSV에 기록했다. `PredMask_validLabel`은 예측 Mask와 GT label 유효영역의 교집합이며, **GT가 없는 background의 좌표 오차를 만들어 계산한 것이 아니다.**

| branch | 평가 영역 | 전체 유효 pixel | 평균3D L2 | X MAE | Y MAE | Z MAE | hard bit 오류율 |
|---|---|---:|---:|---:|---:|---:|---:|
| front | GTmask | 64,635 | 29.4046 | 12.3937 | 8.6497 | 16.3539 | 29.19% |
| front | PredMask_validLabel | 61,334 | 28.6553 | 12.2905 | 8.0674 | 16.1204 | 29.07% |
| front | intersection | 61,253 | 28.6487 | 12.2875 | 8.0656 | 16.1150 | 29.07% |
| front | GTmask_eroded | 54,329 | 27.6082 | 12.0605 | 7.9151 | 15.2983 | 29.30% |
| back | GTmask | 64,635 | 29.1845 | 11.1159 | 10.2033 | 15.7365 | 28.74% |
| back | PredMask_validLabel | 61,334 | 28.4068 | 10.9950 | 9.7223 | 15.4106 | 28.65% |
| back | intersection | 61,253 | 28.4043 | 10.9930 | 9.7233 | 15.4080 | 28.66% |
| back | GTmask_eroded | 54,329 | 27.3928 | 10.6645 | 9.7003 | 14.6003 | 28.82% |

표는 **프레임별 평균의20장 평균**이다. Front GTmask의 pixel-weighted 평균L2는29.532293이고 표의29.404647과 구분한다. 프레임별 median/p95·raw soft target MAE도 CSV에 있다. 평균median을 전체pixel의 pooled median이라고 해석하지 않는다.

- Front GTmask raw 출력 대 `2*HCCE-1` 평균MAE0.468275, Back0.468172. 동적 가중치·Mask Loss를 포함한 총 Loss 값이 아니다.
- Front 첫 HCCE plane 오류율은 X15.27%, Y9.57%, **Z44.55%**; Back Z는43.11%였다. Z의 큰 coarse 오류가 확인된다.
- Front 예측을 Back GT에 잘못 대입해도 평균L2는29.7721로 더 작아지지 않았다(정상29.4046). Back도 교차 비교30.4567이 정상29.1845보다 크다. 단순 Front/Back 파일 뒤바뀜이 주원인이라는 근거를 얻지 못했다.
- Z 상반영역 점유는 Front GT약50.09%/예측54.32%, Back GT50.93%/예측47.34%다. 전체가 한쪽 부호로 붕괴했다고 단정할 수 없다.
- Front GTmask 평균좌표오차와 Front ADD의 Pearson r=0.785838(n=20). 기술적 양의 관계이며 인과나 독립 통계 검정을 의미하지 않는다.

### 성공40 / 실패90 비교

| 프레임 | Front 평균XYZ오차 | Back 평균XYZ오차 | ADD | RE(deg) | TE | 실제 inlier / 입력 | ADD 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| 000040 | 13.2525 | 13.2814 | 12.6888 | 6.0542 | 11.6562 | 101/3109 (3.25%) | 성공 |
| 000090 | 23.4305 | 17.4433 | 63.7365 | 13.9010 | 59.6079 | 71/2869 (2.47%) | 실패 |

성공40도 대부분의 좌표가 정확하다는 뜻이 아니다. 두 표본 모두 correspondence inlier가 소수이며 RANSAC 반환 성공만으로 품질을 보장하지 않는다.

그림: [XYZ_GT_prediction_000040.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/figures/XYZ_GT_prediction_000040.png), [XYZ_GT_prediction_000090.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/figures/XYZ_GT_prediction_000090.png), [HCCE_bit_errors.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/figures/HCCE_bit_errors.png), [XYZ_error_ADD_relation.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/figures/XYZ_error_ADD_relation.png).

CSV 주본: [Predicted_XYZ_error_batch4.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/Predicted_XYZ_error_batch4.csv), [HCCE_bit_channel_errors.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/supplement/tables/HCCE_bit_channel_errors.csv). 초기 `tables/Predicted_XYZ_error.csv`는 batch1 진단이다. 그 파일의 `soft_hcce_MAE`는 tanh 변환을 추가한 보조값이므로 학습 target의 raw MAE로 쓰지 않는다. **raw 오차의 확정치는 supplement의 `raw_output_target_MAE` 열**이다. 초기 파일도 삭제·덮어쓰지 않고 차이를 명시했다.

### Batch 재현성 확인

초기 단일 이미지 진단은 Front ADD2/20이었다. 기존과 같은 batch4에서는 **1/20, ADD평균62.2473843023, RE81.0261168561deg, TE60.9093847786**으로 기존 결과가 재현됐다. AMP·batch 변화가 일부 threshold/RANSAC 결과에 영향을 줄 수 있어 이를 정확도 개선으로 보고하지 않는다. 주 표는 batch4이며 batch1 CSV/JSON은 별도 보조 기록이다.

## 5. 실험 C — PnP 후보 비교

모든 후보는 동일 checkpoint·Val20이다. 원본 RANSAC EPNP,150iterations, 프레임별 RNG seed=42+frame, 기본1.5 원본pixel. Front/Back/Mixed 모두 GT를 이용해 Pose 후보를 생성하지 않았다. GT Mask·GT XYZ가 포함된 후보는 명시적으로 Oracle로 분리했다. 모든 조건의 GT BBox crop은 공통 oracle ROI이며, 여기서 GT 없는 후보선택은 Pose/Mask GT를 추가로 사용하지 않는다는 뜻이다. E2E 검출 평가가 아니다.

- `front`, `back`: 각 branch 좌표.
- `mixed_bf`: 원본처럼 선택된 대응점 순서의 번갈아 절반을 Back으로 치환.
- `mixed_bfu`: 원본의 Front/Back 및 ray상 보간 혼합.
- `topConfidence256/1024`: GT 없이 mask probability×양 branch 평균 bit margin 중 작은 값으로 순위를 매겨 일부 대응점 선택. 보정된 confidence는 아니다.
- threshold3/5는 원본pixel 단위다. 개선을 보장하는 권장 설정이 아니라 진단 후보다.

| 후보 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
| front | 20/20 | 1/20 (5%) | 62.2474 / 63.5035 | 81.0261 | 60.9094 | 120.75 / 4.01% |
| back | 20/20 | 1/20 (5%) | 74.209 / 63.2314 | 83.2968 | 75.8861 | 126.70 / 4.19% |
| mixed_bf | 20/20 | 1/20 (5%) | 85.5116 / 64.9909 | 93.0423 | 82.7705 | 118.30 / 3.89% |
| mixed_bfu | 20/20 | 3/20 (15%) | 70.7775 / 63.3374 | 91.9539 | 68.9581 | 135.70 / 4.49% |
| front_GTmask_Oracle | 20/20 | 1/20 (5%) | 61.7553 / 58.151 | 85.6968 | 62.6962 | 121.70 / 3.88% |
| front_intersection_Oracle | 20/20 | 1/20 (5%) | 68.0288 / 63.6131 | 90.0697 | 69.5148 | 121.20 / 4.04% |
| GTxyz_PredMask_Oracle | 20/20 | 20/20 (100%) | 0.270927 / 0.145242 | 0.0607 | 0.271798 | 3066.70 / 99.03% |
| front_threshold3 | 20/20 | 1/20 (5%) | 58.6327 / 59.7439 | 76.5136 | 55.3521 | 403.45 / 13.36% |
| front_threshold5 | 20/20 | 1/20 (5%) | 65.72 / 59.1839 | 77.2804 | 61.415 | 826.00 / 27.34% |
| front_topConfidence256 | 20/20 | 1/20 (5%) | 95.742 / 65.8189 | 94.1891 | 98.137 | 24.00 / 9.38% |
| front_topConfidence1024 | 20/20 | 2/20 (10%) | 76.2153 / 63.4691 | 92.6849 | 71.9694 | 61.25 / 5.98% |

**후보 선택 대조 — 고정4개(Front/Back/BF/BFU) 중 선택:**

| 선택 방식 | 유효 PnP | ADD 통과 | ADD 평균 / 중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 평균 / 비율 |
|---|---:|---:|---:|---:|---:|---:|
| source_maxReportedInliers | 20/20 | 1/20 (5%) | 70.9321 / 64.5197 | 87.5043 | 63.9269 | 124.70 / 4.12% |
| actual_maxUsedInliers | 20/20 | 2/20 (10%) | 81.57 / 65.0796 | 87.2084 | 78.2007 | 141.25 / 4.67% |
| GT_minADD_Oracle | 20/20 | 3/20 (15%) | 45.4544 / 52.129 | 79.9513 | 44.8794 | 122.70 / 4.07% |

`source_maxReportedInliers`는4후보 중 원본 reported count 최대를 고르는 진단 규칙이며 원본 keypoint subset prior까지 복원한 전체 Tester 재현은 아니다. `actual_maxUsedInliers`는 GT 없는 실제 대응점 inlier 최대 선택이다. `GT_minADD_Oracle`은 **프레임별 GT ADD가 최소인 후보를 고른 제한된4후보의 Oracle 상한**이며 실제 추론 규칙이 아니다. 그것조차3/20이다. 후보만 바꿔 좌표 품질 문제를 해소했다고 볼 수 없다.

### 확인된 원본 inlier 집계 문제

원본 `solve_PnP`는 Back/Mixed의 `Points_3D`로 Pose를 풀고, op2 반환 전 `cv2.projectPoints(pred_front, ...)`로 inlier를 재집계한다. 이번 Back/BF/BFU는 각각20/20프레임에서 실제 사용한 대응점 기준 count와 달랐다. 실제 count는 OpenCV에 넘긴 float32 `Points_3D`를 capture하여 반환 Pose로 재투영하고 동일 threshold로 다시 계산했다. RANSAC raw inlier count도 CSV에 별도 저장했다.

Front는 이 branch 불일치의 영향을 받지 않으므로 이 버그가 **Front Baseline ADD5%의 원인이라고 주장하지 않는다.** 일부 임계점에서는 float32/float64 재투영의 경계 판정으로1개 count 차이도 생길 수 있다. 집계 수정 후에도 실제 max-inlier 대조는2/20이고 평균ADD는81.5700으로 낮지 않았다.

근거: [PnP_candidate_comparison_batch4.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/PnP_candidate_comparison_batch4.csv), [PnP_candidates_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/tables/PnP_candidates_per_frame.csv), [PnP_candidate_selectors.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/supplement/tables/PnP_candidate_selectors.csv), 원본 `/RESEARCH_DATA/HCCEPose/HccePose/PnP_solver.py:solve_PnP`, `solve_PnP_comb`.

## 6. 좌표·Mask 치환을 통한 병목 분리

| 치환 조건 | ADD 통과 | 해석 |
|---|---:|---|
| 예측XYZ + 예측Mask | 1/20 | 실제 checkpoint의 기존 Baseline 조건 |
| 예측XYZ + GTmask | 1/20 | Mask만 정답으로 바꿔도 주 오차는 남음, Oracle |
| GTXYZ + 예측Mask | 20/20 | 같은 예측Mask로도 충분한 GT 대응점 확보 가능, Oracle |
| 예측Front + GTvalidMask | 1/20 | 아래 축 치환의 통제조건, Oracle Mask |
| GT X만 치환 | 2/20 | Oracle, 실제 성능 아님 |
| GT Y만 치환 | 7/20 | Oracle, 실제 성능 아님 |
| GT Z만 치환 | 10/20 | Oracle, 평균ADD21.6841, Z 좌표 중요성의 추가 증거 |

축 치환은 정답Mask·유효라벨 조건으로 통제하며 다른 두 축은 예측을 유지했다. 혼합 좌표는 물리적 단일 표면과 항상 일치하지 않아 정식 개선 알고리즘이 아니다. Z만으로 전체 원인을 설명할 수도 없다.

근거: [GT_channel_replacement_Oracle.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/channel_controls/tables/GT_channel_replacement_Oracle.csv).

## 7. 원인별 판정과 미확정 가설

| 원인 | 판정 | 확인된 증거 | 미확인·한계 | 다음 조치 |
|---|---|---|---|---|
| CAD·카메라 좌표계 | PBR 내부 거시 불일치 근거 없음 | A0 ADD20/20, 평균0.00001962; PBR GT 라벨 min/max 독립 ray400개 확인 | 실사 K/newK·절대 물리단위는 이번 검증 대상 아님 | 현 PBR K·CAD 유지; 실사 provenance 별도 확보 |
| XYZ 라벨 생성 | 정밀 잔차·경계 누락 확인, 주 오차 원인 근거 약함 | A1 ADD20/20 평균0.164691, A2도20/20; GTmask 경계의 무효 라벨356pixel | VisPy bitwise 비교·전체표면 point-to-mesh 미완료 | 경계 support 분리 및 원본 VisPy 소수 표본 비교 |
| HCCE encode/decode | 시험 범위에서 통과 | 각 축256값 원본 encode→threshold→decode 최대 차이0; A1 uint8 차이0; XYZ float 복원 최대차이2.6866e-6 | 예측 bit 오류는 존재, encoder 역함수 오류와 구분 | 원본 encoder/decoder 회귀테스트 유지 |
| Crop→원본 좌표 | 거시 오류 근거 없음, 작은 샘플링 잔차 | A2 GT20/20 평균ADD0.288577; nearest-pixel 대조0.148635; cropK 대조도20/20 | half-pixel/최근접·양자화 잔차는 남음 | 임의 offset 변경 금지; affine K·임계값 단위 명시 |
| 네트워크 좌표 예측 | 현재 가장 큰 병목, Z가 특히 중요 | GTmask Front/Back 평균3D오차29.4046/29.1845; GT Z치환 Oracle10/20 | 왜 Z 예측이 실패하는지(학습실패·일반화·손실/초기화)는 미확정 | 다음 run 전 Train80 동일 좌표/bit 측정으로 학습 대 일반화 분리 |
| Mask | 기여 가능하나 단독 주원인 근거 약함 | 예측XYZ+GTmask1/20, GTXYZ+예측mask20/20 | 다른 가림/검출 조건에서 중요하지 않다는 뜻 아님 | Mask 경계/무효 support를 명시하고 좌표 개선 우선 |
| PnP·대응점 품질 | 예측 대응점 품질 저하 및 원본 inlier 집계 문제 확인 | front 실제 평균inlier120.75/3095.30, 평균 비율4.01%; Back/Mixed는 Front로 inlier재계산 | 집계 수정만으로 정확도 개선된다고 확정 불가; 실제max-inlier ADD2/20 | 작업 복사본에서 inlier 계산을 실제 Points3D로 통일하고 회귀검증 |
| 미확정 원인 | 추가 근거 필요 | 단일 checkpoint·합성 Val20·같은 Val 다중 후보 비교 | 대칭 모호성·PBR 외관·증강·pretrained·일반화 효과 미검증 | 새 독립 테스트와 한 요인 대조 실험; 성능 향상 가설로 표시 |

**다음 변경 제안은 미실행이다.** 먼저 실제 대응점 inlier 집계를 복사본에서 바로잡아 비교 지표를 안정화한다. 학습 연구는 Z 중심의 raw/bit/XYZ 검증 지표를 추가하고 Train80 대 Val20을 비교하는 것부터 시작한다. 그 결과 없이 epochs·lr·PnP threshold를 무작정 바꾸지 않는다. 이후 pretrained/증강/좌표 손실 후보를 한 요소씩 비교하고 개선 효과는 새 결과로 검증한다. 대칭이나 단위를 성능이 좋아지는 방향으로 임의 변경하지 않는다.

## 8. 산출물과 재현

| 파일 | 내용 |
|---|---|
| [GT_XYZ_Oracle_PnP.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/GT_XYZ_Oracle_PnP.csv) | A0 각 프레임의 실제 PnP 결과·실패 원인 |
| [GT_HCCE_Roundtrip.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/GT_HCCE_Roundtrip.csv) | A1/A2 및 nearest 대조 |
| [Predicted_XYZ_error_batch4.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/Predicted_XYZ_error_batch4.csv) | 주 좌표·raw HCCE 오차 |
| [PnP_candidate_comparison_batch4.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/PnP_candidate_comparison_batch4.csv) | 주 후보별 mean/median ADD·RE·TE·inliers |
| [cause_diagnosis.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/tables/cause_diagnosis.csv) | 원인별 판정 표 |
| [returned_poses.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/checks/returned_poses.json) | A와 초기 batch1의 실제 반환R/t |
| [returned_poses.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/checks/returned_poses.json) | 주 batch4 후보의 실제 반환R/t |
| [GT_to_prediction_ADD.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/figures/GT_to_prediction_ADD.png) | GT 경로 대 예측 경로 평균ADD |
| [front_000040_overlay.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/figures/front_000040_overlay.png) | 성공40, GT녹색/예측자홍 CAD wireframe |
| [front_000090_overlay.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/matched_batch4/figures/front_000090_overlay.png) | 실패90, 같은 표시 |
| [A1_full_HCCE_000090_overlay.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/figures/A1_full_HCCE_000090_overlay.png) | 같은90의 GT 라벨 Oracle PnP overlay |
| [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/checks/final_integrity.json) | 원본·이전 산출물 SHA-256 보존 확인 |

주 평가 CSV에는 실패를0오차로 채우지 않고 명시적 failure를 남기는 정책을 적용했다. 이번 후보들은 모두 유효반환20/20이라 반환 실패 표본은 없었고, 이는 실패처리 경로 전체가 테스트됐다는 의미는 아니다. ADD 실패와 PnP 반환 실패를 구분한다. 성공률 분모는 항상20이다.

실행 순서와 코드(이미 생성된 결과를 덮어쓰지 않고 새 경로로만 재현):

```bash
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/scripts/diagnose.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/scripts/matched_batch4.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/scripts/supplement.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/scripts/channel_controls.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/diagnostics_v1/scripts/cropK_equal_units.py
```

위 스크립트는 절대 기존 학습 파일을 수정하지 않으며 같은 출력이 있으면 재사용 또는 실패할 수 있으므로 현재 폴더에 그대로 재실행하지 않는다. 새 진단 root의 경로 설정과 예산을 먼저 고정한다. `logs/`에 각 실행 원문을 보존했다. 코드의 원본 함수는 AST 추출로 재사용했고 네트워크는 이전 학습의 명시적 float 타입 호환 복사본을 사용했다. 이번 진단 중 CUDA busy 확인에서 다른 compute PID가 없었으며, 최종 query는 `실행 중인 compute process 없음`이다.

원본740파일, 이전 세미나366파일, 이전 Baseline61파일의 SHA-256이 모두 기존 manifest와 같았다. 기존 source·데이터·checkpoint·report·venv를 보존했다. 실사27은 이번 학습/평가/후보선택에 사용하지 않았다.

## 9. 세미나 발표 결론과 제한

발표 문장: **“정답 XYZ를 동일한 HCCE·crop·PnP 경로에 넣으면 Val20 전체가 통과한다. 예측 좌표는 평균 약29 PBR단위 오차이며, 정답Mask 치환만으로는 ADD5%가 개선되지 않았다. 주요 병목은 좌표 예측, 특히 Z 축으로 좁혀졌다.”**

GT Z 치환50%, Mixed15%, GT 후보선택15%는 서로 조건이 다른 Oracle/탐색 대조다. 기존 Baseline 정확도가5%에서50%로 개선됐다고 발표하면 안 된다. 독립 테스트·실사 추론·추가 학습·구조 개선·원본 VisPy 대조는 미수행이다.
