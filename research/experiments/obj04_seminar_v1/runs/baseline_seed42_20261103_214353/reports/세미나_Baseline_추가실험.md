# 세미나 HCCEPose Baseline 추가 실험

작성: 2026-11-03T22:05:28.185994+09:00

**PBR Train 80 / Val 20으로 첫 소규모 학습·Pose 평가를 완료했다. 주 결과(epoch 155)는 Val Mask IoU 0.935782, ADD@0.1d 1/20(5%)이다. PnP의 유효 Pose 반환 20/20은 정확한 Pose 20개를 뜻하지 않는다. 실사 SMT 정확도는 이번에 측정하지 않았다.**

- 신규 작업 루트: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353`
- 정식 학습·평가 실행: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only`
- 전체 추가 작업 경과: 보고서 생성 시 약 21.59분 / 최대 90분. 추가 학습은 중단했으며 이후에는 보존·보고서만 확인한다.
- 최종 학습 wall time: 105.457초. 진단 시도 20.113초를 포함해 두 학습 실행 약 125.571초. 검증·저장 포함 시간이며 CUDA kernel 시간이나 일반 처리량 benchmark가 아니다.

## 1. 기존 결과와 이번 신규 결과

| 구분 | 내용 |
|---|---|
| 기존 결과 재사용 | 격리 venv, CUDA 동작 확인, PBR 100장 검증, CPU Front/Back XYZ 100쌍, CAD 복사본, 블록 분할 |
| 기존 실사 결과 | GT 27장, 고유 행렬 15종, 거시 시각 성공22·판단불가5. 모델 정확도 수치 아님 |
| 이번 신규 결과 | Loss 호환성 수치·gradient 비교, 무작위 초기화 학습 195epoch/3900step, Val20 loss·Mask·PnP·ADD·RE·TE, 체크포인트와 그림 |
| 미수행 | 실사27 모델 추론, YOLO 학습/새 검출 평가, 독립 테스트, 다중 seed, pretrained 비교, 원본 VisPy와 XYZ bitwise 비교 |

기존 4-step 호환 smoke 체크포인트를 이어 학습하지 않았다. 이번은 별도의 무작위 초기화 실험이다. 기존 환경을 재사용했으며 이번 작업에서 패키지 설치·업데이트를 하지 않았다.

## 2. Loss 타입 수정 검증

원본 `HccePose/network_model.py`는 읽기만 했다. 신규 복사본 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/code/network_model_explicit.py`의 Front/Back 이동평균 대입 두 곳만 `np.mean(...)` → `float(np.mean(...))`로 변경했다. 수식·history 업데이트·가중치 계산은 유지했다.

| 비교 | 반복 | 최대 loss 차이 | 최대 gradient 차이 | 최대 weight/error 차이 | history |
|---|---:|---:|---:|---:|---|
| 기존 proxy 대 명시적 cast, CPU float32 | 120 | 0 | 0 | 0 | 완전 동일 |
| 같은 비교, CUDA float32 | 120 | 0 | 0 | 0 | 완전 동일 |
| 같은 비교, CUDA float16 | 120 | 0 | 0 | 0 | 완전 동일 |

window=100을 넘기는 반복과 zero/full/partial Mask를 포함했다. 원본 무수정 첫 forward·gradient도 일치했다. 원본은 둘째 forward에서 `can't assign a numpy.float32 to a torch.FloatTensor`로 중단돼 원본 무변환과 120회 직접 비교는 불가능했다. **확인 범위에서는 값·gradient 영향이 없었다. 모든 dtype/입력에 대한 수학적 증명은 아니다.**

근거: [loss_compatibility.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/loss_compatibility.json), [loss_type_fix.diff](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/loss_type_fix.diff), `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/scripts/verify_loss.py`.

## 3. 학습 설정·분할·누수 방지

| 설정 | 실제 값 |
|---|---|
| 물체·네트워크 | obj_id=4, 기존 HCCEPose/ResNet34, parameter 29,123,249 |
| 초기화 | random, pretrained 및 이전 smoke 가중치 사용 안 함 |
| 환경 | 기존 전용 venv, torch 2.8.0+cu128 / torchvision 0.23.0+cu128, RTX 5090 단일 GPU |
| seed / batch / crop | 42 / 4 / RGB256×256, Mask·좌표128×128 |
| optimizer | Adam, lr=0.0002, AMP float16+GradScaler |
| 입력 | 기존 BGR 정규화, GT visible BBox pad1.5 oracle crop |
| augmentation | 없음. 고정 crop, 색상 증강 없음 |
| Train | 0–39, 50–89: 80프레임, 생성 블록 0,1,2,3,5,6,7,8 |
| Validation | 40–49, 90–99: 20프레임, 생성 블록 4,9 |
| 실사27 사용 | 학습·모델 선택·이번 모델 추론 모두 0장 |
| 검증 | epoch1 및 매5epoch, eval/inference_mode; train80·val20 각각 새 Loss history |
| 종료 | 최대200epoch 중 195에서 Val Loss early stop, patience8 checks / min_delta0.0001 |
| 저장 | 새 디렉터리, epoch50/100/150 주기·best Loss·best Mask·last 별도 보존 |

프레임과 생성 블록 교집합은 없다. 단일 합성 자산의 Val20으로 모델 선택과 성능 집계를 모두 했으므로 독립 테스트 정확도가 아니다. 원본 스크립트의 random crop/색상 증강, pretrained 설정을 그대로 재현한 논문 Baseline이라고 주장하지 않는다. **증강 없는 소규모 실행 Baseline**이다. 라벨은 이전 CPU min/max-depth 구현을 재사용했으며 원본 VisPy 출력과 직접 동등성 비교는 아직 하지 않았다.

근거: [config.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/config.json), [split.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/split.json), `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/support.py`, `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/train_baseline.py`.

## 4. 초기 체크포인트 선택 문제와 재실행

첫 진단 실행은 35epoch/700step 후 종료됐다. epoch0 Val Loss=0.781321이 이후 학습 모델보다 낮아 무학습 모델이 best로 남는 문제가 있었다. 초기 Mask IoU는 0.002371이고 평균 예측 전경은 283.3pixel, GT는 3249.55pixel이었다. 원본 Loss에는 예측 Mask에 따른 좌표오차 게이팅이 있어 **낮은 총 Loss만으로 Pose 품질을 판단할 수 없다.**

첫 실행의 코드·로그·체크포인트는 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353`에 보존했다. 별도 `trained_only` 실행에서 epoch0을 reference로만 기록하고 후보는 epoch≥1로 제한했다. 같은 seed의 새 random 초기화로 다시 시작했으며 앞 700step을 이어 학습하지 않았다. 이 선택 정책 변경은 PnP 평가 전에 확정했다.

주 결과는 학습 epoch 중 최소 Val Loss(epoch155)이다. epoch180은 최대 Mask IoU 보조 결과, epoch195는 마지막 보조 결과이다. **마지막 ADD가 더 높다는 이유로 주 체크포인트를 사후 교체하지 않았다.**

근거: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/checks/training_summary.json`, [initial_mask_diagnostic.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/initial_mask_diagnostic.json), [training_summary.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/training_summary.json).

## 5. 실제 PBR Validation 결과

| 체크포인트 | epoch | step | Val Loss | Mask IoU 평균 | 유효 PnP | ADD@0.1d | RE 평균(deg) | TE 평균(원본 단위) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 주 결과: 최소 Val Loss | 155 | 3100 | 1.537052 | 0.935782 | 20/20 | 1/20 (5%) | 81.0261 | 60.9094 |
| 보조: 최대 Val Mask IoU | 180 | 3600 | 1.594160 | 0.953537 | 20/20 | 1/20 (5%) | 98.3313 | 67.9463 |
| 보조: 마지막 | 195 | 3900 | 1.595584 | 0.951124 | 20/20 | 3/20 (15%) | 80.8438 | 58.5579 |

- Mask IoU는 프레임별 IoU의 평균이다. GT BBox oracle 조건이며 YOLO 포함 E2E 성능이 아니다.
- PnP는 원본 `HccePose/PnP_solver.py:solve_PnP`의 front-only RANSAC EPNP, reprojectionError=1.5px, iterationsCount=150이다. Front/Back 후보 중 GT가 좋은 것을 선택하지 않았다.
- solver success·유한 R/t·모든 CAD 점 positive Z를 유효 반환으로 정의했다. 세 모델 모두 20/20 유효했으나 오차가 큰 Pose도 포함한다.
- ADD는 CAD 꼭짓점1439개의 동일 인덱스 대응 거리 평균이다. 대칭을 추가하지 않았다. diameter=139.3786948551, 임계값=13.9378694855, strict `<0.1d`. 실패도 전체20 분모에 포함한다.
- 주 모델 ADD 평균62.247384, 중앙값63.503533, p95=101.896021. RE 중앙값71.152861deg, TE 중앙값63.113380이다. 모두 유효 반환20개로 계산했다.
- 길이 오차는 **PBR 원본 좌표 단위**다. PBR provenance에는 mm→m scale=0.001이 있으나 이번에 실제 CAD 물리 치수를 독립 계측하지 않았다. 실사 단위 확정 결과로 확대 해석하지 않는다.

좌표는 원본 `inference_batch`로 복호화하고 crop bbox로 전체 영상 2D 좌표를 복원했다. GT는 object-to-camera로 비교했다. 이 결과는 원본 전체 Tester 경로의 재현이 아니라 원본 network·Loss·PnP를 연결한 별도 평가 어댑터 결과이다.

근거: [validation_pose_summary.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/validation_pose_summary.json), [validation_pose_per_frame.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/logs/validation_pose_per_frame.csv), `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/evaluate_pose.py`, 원본 `HccePose/network_model.py`, `HccePose/PnP_solver.py`.

## 6. 곡선·예측 그림과 해석

| 파일 | 내용 |
|---|---|
| [training_loss.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/training_loss.png) | 실제 step loss의 epoch 평균, smoothing 없음 |
| [train_vs_validation_loss.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/train_vs_validation_loss.png) | 동일 평가 절차의 Train80 대 Val20 Loss, epoch0 reference 제외 |
| [mask_iou_curve.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/mask_iou_curve.png) | 프레임별 Mask IoU 평균 변화 |
| [best_loss_mask_gt_prediction.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/best_loss_mask_gt_prediction.png) | 고정 Val40·45·90·95, RGB/GT/Prediction/TP·FN·FP |
| [best_loss_pnp_overlay_000040.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/best_loss_pnp_overlay_000040.png) | 고정 Val40, 녹색 GT contour / 자홍 predicted CAD wire |
| [best_loss_pnp_overlay_000090.png](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/figures/best_loss_pnp_overlay_000090.png) | 고정 Val90, 같은 표시. 큰 위치/크기 오차도 그대로 공개 |

주 모델의 000040은 ADD 약12.689로 임계값을 통과했으나 000090은 약63.736으로 통과하지 못했다. 그림은 성공 표본만 선택하지 않고 사전에 정한40·90을 사용했다. wireframe은 hidden-face 제거 렌더링이 아니며 실제 예측 R/t의 투영이다.

**[확인]** epoch155→195에서 train 평가 Loss는1.139226→1.118626으로 줄고 Val Loss는1.537052→1.595584로 증가했다. 마지막 train/val Mask IoU는0.981942/0.951124이다.

**[해석]** 후반 총 Loss에 과적합 징후가 있다. 동적 가중치와 예측 Mask 게이팅 때문에 Loss 단독으로 좌표 품질·Pose 개선을 판정할 수 없다. Mask IoU가 높아도 ADD는 낮았으며 이 불일치 자체가 이번에 확인한 연구 결과다. 작은 합성 Val2블록·단일 seed로 일반화나 정밀 공정 정확도를 확정하지 않는다.

**[미확인 가설]** 좌표 복호화·XYZ 양자화/라벨 렌더러 차이, 평면형상의 PnP 불안정, 학습량·증강·무작위 초기화·형상 대칭 등이 Pose 오차 후보이다. 이번 수치만으로 원인을 특정하지 않는다.

## 7. 자원·보존·재현 범위

- 성공 optimizer3900회, AMP skip0. peak allocated 1,035,334,656B (987.37MiB), reserved 1,140,850,688B (1088.00MiB). PyTorch allocator 수치이며 GPU 전체 점유와 다르다.
- 원본 manifest 740파일, 기존 실험 artifact 366파일 SHA-256 재비교 모두 일치했다. 기존 소스·원본85개·이전 체크포인트·보고서를 보존했다.
- GPU guard로 다른 compute PID를 감시했다. 종료 후 compute query: `실행 중인 compute process 없음`. 감시 시점의 정보이며 서버 독점 예약을 의미하지 않는다.
- 보존 근거: [final_integrity.json](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/final_integrity.json). 이전 환경/공용 Python 변경 없음, sudo 없음.
- 주 체크포인트: [best_val_loss.pt](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checkpoints/best_val_loss.pt). 마지막 체크포인트는 optimizer·Loss history를 포함한다. 이 실험의 메타데이터 형식은 원본 `load_checkpoint`/Tester에 바로 넣는 형식과 다르므로 추후 명시적 adapter가 필요하다.

실행한 학습·평가 진입점:

```bash
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/train_baseline.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B /RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/scripts/evaluate_pose.py
```

위 명령은 이미 생성된 출력을 덮어쓰는 재실행 안내가 아니다. 재현 시 새 run에 코드·config를 복사하고 `support.py`의 RUN/BASE, 절대 경로, 예산 시각을 갱신한다. 현 출력은 보존한다. seed 고정은 다른 환경에서 bitwise 동일함을 보장하지 않는다.

원문: `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/logs/training.log`, `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/logs/train_steps.jsonl`, [epoch_metrics.csv](/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/logs/epoch_metrics.csv). 실제 log 파일명은 결과 디렉터리 파일 목록을 함께 참조한다.

## 8. 세미나 슬라이드 교체 제안

1. **지난 연구 진행:** 기존1번 유지. 마지막에 “4-step 기능 검증에서 PBR80/20 소규모 학습·평가로 진행” 추가. 기존 GT 검증과 신규 모델 결과를 구분한다.
2. **실사27 GT 검증:** 기존2번 유지. 성공22/판단불가5는 시각 GT 점검 결과이며 이번 Pose 정확도가 아니라고 명시한다. 새 실사 추론 그림은 없다.
3. **PBR100 구성:** 기존3번 유지. 블록8/2와 train80/val20을 강조하고 Val이 독립 실사 테스트가 아님을 설명한다. 이전 `/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/figures/pbr_rgb_000010.jpg` 사용 가능.
4. **첫 Baseline 학습·PBR 검증:** 기존 “optimizer4회” 중심 슬라이드를 교체한다. 핵심 문장: “195epoch/3900step 학습 후 주 모델 Mask IoU0.9358, ADD1/20.” `train_vs_validation_loss.png`, `best_loss_mask_gt_prediction.png`를 삽입한다. random init·oracle BBox·실사 미평가를 함께 쓴다.
5. **Mask와 Pose의 차이·다음 실험:** `best_loss_pnp_overlay_000040.png`와 `...000090.png`를 나란히 넣는다. 핵심 문장: “PnP 반환100%는 ADD 성공5%와 다르다.” 총 Loss 선택 한계와 작은 Val 규모를 설명하고, 원인 분리 실험을 다음 단계로 제안한다. 마지막 checkpoint15%는 보조 관찰이며 주 결과로 바꾸지 않는다.

각 새 그림의 정확한 경로는 6절 표에 있다. 4번 실제 수치는 epoch155/3100step(선택 모델), 총 학습195epoch/3900step(전체 실행)을 구분해서 표시한다.

## 9. 다음 우선순위 — 제안, 미실행

1. **GT XYZ→원본 decoder→PnP 합성 round-trip**을 Val에서 확인해 모델 오차와 좌표·PnP 경로 오차를 분리한다. 원본 VisPy와 CPU 라벨도 소수 표본에서 비교한다.
2. 예측 좌표의 GT 대응 오차를 front/back·가시 영역별로 측정하고 PnP inlier·깊이 오차를 기록한다. 실패90번 등에서 실제 원인을 먼저 확인한다.
3. Val 선정 규칙을 다음 run 전에 고정한다. 총 Loss 외 좌표오차/ADD를 고려하되 같은 Val을 반복 튜닝하면 개발 평가로만 보고한다. 사후 유리한 체크포인트 선택을 피한다.
4. 증강·local pretrained·학습 budget 비교를 한 요소씩 수행한다. 실사 K/단위·ROI 준비 후 실사27 dev를 평가하고 새 촬영 세션의 독립 테스트를 확보한다.

추가 학습을 더 오래 하면 정확도가 오른다는 결론은 내리지 않는다. 이번 성과는 **재현 가능한 소규모 학습·정량 Pose 평가 결과와 Mask/Pose 성능 차이의 확인**이다.
