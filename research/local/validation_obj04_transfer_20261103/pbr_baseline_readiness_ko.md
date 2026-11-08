# RTX 5090 SMT obj04 HCCEPose Baseline 재현 준비 점검

- 점검일: 2026-11-03, Asia/Seoul
- 소스: `/RESEARCH_DATA/HCCEPose`
- PBR: `/RESEARCH_DATA/HCCEPose_experiments/realistic_final100`
- 실사·GT: `/RESEARCH_DATA/HCCEPose_obj04_transfer`
- 지정 환경: `/RESEARCH_DATA/USER/miniconda3/envs/hccepose`
- **종합 판단: PBR은 재사용 가능성이 높지만 현재 상태로 학습·실사 추론을 바로 실행할 수 없다. 필수 패키지, 학습용 CAD 배치, 앞·뒷면 XYZ 라벨, 학습·검증 분리 설정이 빠져 있다. YOLO 가중치는 정상 ZIP 구조이나 CPU 모델 로딩은 의존성 부재로 검증하지 못했고, 성능 기록은 smoke test 수준이다.**

## 0. 점검 범위와 수행하지 않은 작업

**[확인]** PBR 100프레임의 RGB·Depth·Mask·visible Mask를 시스템 `/usr/bin/python3`의 기존 Pillow로 CPU 디코딩했다. JSON·텍스트·모델 메타데이터·ZIP CRC·pickle opcode는 읽기 전용으로 검사했다. pickle 역직렬화나 체크포인트 모델 생성은 하지 않았다. `hccepose` 환경에서는 패키지 검색과 `torch`·`ultralytics` import 실패 확인까지만 수행했다.

GPU는 `nvidia-smi`로 장치·드라이버·메모리 상태만 조회했다. CUDA tensor 연산, 신경망 forward, 학습, 설치, 환경 변경, 데이터 변경, 기존 코드 수정은 수행하지 않았다. 공식 PyTorch·NVIDIA 웹 문서만 호환성 설명을 위해 조회했으며 모델·패키지·데이터를 다운로드하지 않았다. 본 보고서 하나만 신규 저장한다.

**표기:** [확인]은 실제 파일·검사값, [출처 기재]는 자산 메타데이터의 주장, [해석]은 구현·수치로부터의 판단, [제안]은 아직 수행하지 않은 작업이다.

## 1. PBR 데이터 존재 및 정합성

PBR 전체는 617파일, 81,322,399 B이다. 실제 BOP 장면 디렉터리는 `train_pbr/000000/` 하나이며 GT 프레임 키는 0~99이다.

| 항목 | 존재·디코딩 | 크기 합계 | 검사 결과 |
|---|---|---:|---|
| `rgb/*.jpg` | 100/100 | 22,677,522 B | 모두 1280×720, 파일 해시 중복 없음 |
| `depth/*.png` | 100/100 | 7,312,386 B | 모두 1280×720, Pillow `I` 모드 정수 깊이 |
| `mask/*.png` | 100/100 | 249,377 B | 모두 1280×720, 0/255 이진값 |
| `mask_visib/*.png` | 100/100 | 251,355 B | 모두 1280×720, 0/255 이진값, 전체 Mask의 부분집합 |
| `scene_gt.json` | 존재 | 46,549 B | 100프레임, 프레임당 obj_id=4 하나 |
| `scene_camera.json` | 존재 | 61,306 B | GT와 키 집합 일치, K는 루트 camera.json과 일치 |
| `scene_gt_info.json` | 존재 | 33,220 B | GT와 키·인스턴스 수 일치. 일부 픽셀 수 불일치 |

**[확인]** RGB·Depth·각 인스턴스 Mask의 대응 파일 누락은 없었다. `bbox_obj`, `bbox_visib`는 100프레임 모두 실제 Mask 경계와 일치했다. 규약은 BOP 로컬 코드의 `w=xmax-xmin`, `h=ymax-ymin`이며 inclusive `+1` 방식과 구분했다. 근거: `bop_toolkit.zip!bop_toolkit/bop_toolkit_lib/misc.py:calc_2d_bbox`.

### 발견한 경미한 GT-info 불일치

| 프레임 | 필드 | JSON | 실제 Mask/Depth에서 센 값 |
|---|---|---:|---:|
| 000024 | all / valid / visible | 각각 20,847 | 각각 20,848 |
| 000036 | all / valid / visible | 각각 33,592 | 각각 33,593 |
| 000077 | all / valid | 각각 17,602 | 각각 17,603 |
| 000077 | visible | 17,599 | 17,600 |

000077의 `visib_fract`도 재계산값과 엄밀히 일치하지 않는다. 총 10개 필드 불일치가 3프레임에서 발생했다. 나머지 97프레임의 해당 픽셀 통계는 일치했다. 원인은 미확인이다. **[해석]** 1픽셀 차이는 대규모 누락과는 다르며 기존 `visib_fract>=0.2` 필터 결과를 바꾸지 않지만, 정확한 데이터 버전을 위해 파생본에서 일관되게 재계산하는 것이 좋다. 원본을 고치지는 않았다.

| 수치 | 실제 범위 |
|---|---|
| `depth_scale` | 모두 1.0 |
| 프레임별 최소 raw depth | 193~529 |
| 프레임별 최대 raw depth | 553~3,307 |
| `cam_t_m2c.z` | 416.2843933105469~646.6917114257812 |
| `visib_fract` | 0.9968133666350875~1.0 |
| R 최대 직교 원소 오차 | 4.000356232936042e-7 |
| det(R) | 0.9999995402570205~1.0000003078801798 |

**[해석]** 회전 오차는 float32 계산 정밀도 수준으로 보이며 크게 잘못된 회전행렬 징후는 없다. 모든 표본이 거의 완전히 보이므로 실제 SMT의 도구 가림 조건을 학습하는 자료로는 부족할 수 있다. 실제 일반화 효과는 미측정이다.

**[출처 기재]** `sampling_and_provenance.json`은 10개 생성 장면 × 10프레임, 원래 realistic_100의 10~19를 neutral_replacement10으로 교체, `reference_poses_used_as_training=false`, `object_scale_m_per_mm=0.001`, `mesh_recentered=false`를 기록한다. 이 선언을 독립 생성 로그로 재현 검증하지는 않았다.

000000·000010·000040 RGB를 직접 확인했다. 검은색 고반사 표면과 목재 배경이 보인다. 실제 영상의 표면 외관·도구 가림과 도메인 차이가 있으며 정확도 병목이라는 가설은 별도 검증해야 한다. 법선 오류나 재질 설정 오류라고 단정하지 않는다.

## 2. CAD·models_info·카메라

**[차단 문제]** PBR의 `models/`에는 `models_info.json`만 있고 **`obj_000004.ply`가 없다.** `bop_dataset`은 모델 경로를 구성하므로 기존 경로 그대로는 CAD 로딩·XYZ 라벨 렌더링·ADD 평가를 할 수 없다.

`models_info.json`에는 물체 키 `"4"` 하나가 있고 다음 값이 있다.

```json
{
  "diameter": 139.37869485511956,
  "min_x": -45.25,
  "min_y": -45.25,
  "min_z": -32.65763854980469,
  "size_x": 90.5,
  "size_y": 90.5,
  "size_z": 65.31527709960938
}
```

이 필드는 현재 HCCEPose가 사용하는 min/size/diameter를 포함한다. max 필드가 없어도 min+size로 계산하는 현재 네트워크 경로에는 문제가 없다. 대칭 필드는 없으므로 현재 metric은 ADD를 선택한다. 실제 물체의 기능상 대칭을 확인하지 않고 필드를 추가하지 않는다.

### 실제 전송 CAD의 재사용 후보 검증

`HCCEPose_obj04_transfer/obj_000004.ply`는 중심이 (0,0,0)이며 PBR 모델 정보와 다음 정도로 차이가 작다.

| 항목 | PBR 모델 정보 − 실제 전송 CAD 계산값 |
|---|---:|
| diameter | -2.41439579440339e-7 |
| min_z | +1.4501953131684786e-6 |
| size_z | -2.900390626336957e-6 |
| min_x/min_y/size_x/size_y | 0 |

실제 전송 CAD에 PBR R·t·K를 적용한 CPU 꼭짓점 투영 bbox와 저장된 `bbox_obj` 경계의 최대 차이는 약 **1.912px**였다. 모든 꼭짓점의 카메라 Z 범위는 370.1421749193445~672.446800614146로 양수였다.

**[해석]** 같은 좌표계·스케일의 CAD 후보라는 근거는 강하다. 그러나 원래 PBR 생성 CAD가 누락되어 해시 동일성은 입증할 수 없고, bbox 근사만으로 전체 표면 일치를 보장하지 않는다. 승인 후 전송 CAD의 독립 복사본으로 PBR 재투영을 확인하고 사용해야 한다. 다시 중심 정렬하거나 임의 스케일 변환하지 않는다.

PBR K와 실제 전송 K의 최대 원소 차이는 4.392454854951211e-5로 작다. PBR은 PBR의 K를, 실사는 확정된 실제 출력 K를 사용한다. PBR 메타데이터는 mm→m 스케일을 명시하므로 PBR 단위 해석의 근거가 추가됐지만 실제 CAD 내보내기 단위·실사 GT 생성 이력까지 모두 확인됐다는 뜻은 아니다.

## 3. YOLO best.pt와 학습 기록

경로: `yolo11/train_obj_s/smoke_train/`.

| 파일 | 크기 | 읽기 전용 확인 |
|---|---:|---|
| `weights/best.pt` | 5,441,626 B | PyTorch ZIP 구조, CRC 전부 통과 |
| `weights/last.pt` | 5,441,626 B | ZIP CRC 통과 |
| `weights/epoch0.pt` | 16,052,641 B | ZIP CRC 통과 |
| `results.csv` | 275 B | 실제 평가 행 1개 |
| `args.yaml` | 1,910 B | smoke 설정·Windows 경로 포함 |

`best.pt` SHA-256:

```text
92d31f7a4bd418a06479ea68b3080ba145f025d684fa5970956dac76051e5aa2
```

pickle opcode를 실행 없이 읽었을 때 Ultralytics `DetectionModel` 관련 참조와 버전 `8.3.233`이 확인됐다. **ZIP CRC 통과는 의미론적 정상 모델 로딩과 다르다.** 지정 환경에서 `torch`와 `ultralytics` import가 모두 `ModuleNotFoundError`로 실패했으므로 `YOLO(best.pt)` CPU 로딩과 추론은 수행하지 못했다. 정상 로딩 검사는 **미완료/의존성으로 차단**이다.

### 실제 학습 설정과 결과

| 설정 | 기록값 |
|---|---|
| 모델 | `yolo11n.yaml` |
| 초기화 | `pretrained: false` |
| epochs | 1 |
| fraction | 0.1 |
| batch / imgsz / workers | 4 / 640 / 0 |
| AMP | false |
| 클래스 | nc=1, names=["4"], 라벨 클래스 인덱스 0 |
| train/val 파일 수 | 80 / 20 |

| 지표 — 유일한 epoch 행 | 값 |
|---|---:|
| epoch | 1 |
| 기록 time | 1.12763 |
| train box / cls / dfl loss | 3.21887 / 5.2971 / 5.04793 |
| val box / cls / dfl loss | 2.51416 / 4.3907 / 4.1589 |
| precision / recall / mAP50 / mAP50-95 | 모두 0 |

time은 CSV의 기재값이며 서버 실측 처리량이 아니다. `fraction=0.1`은 학습 자료 일부 사용 설정이며 로더가 실제 읽은 표본 수는 별도 로그 없이 확정하지 않는다. best/last의 epoch=-1 토큰은 저장 시 정리된 체크포인트 메타데이터이며 CSV의 1 epoch 실행 기록과 모순으로 해석하지 않는다.

**[판단]** best.pt는 구조·버전 확인과 추가 학습의 초기 후보로 보존할 수 있지만, 현재 기록만으로 유효한 obj04 검출기로 승인할 수 없다. Pose-only Baseline을 먼저 분리하고 검출기는 충분한 새 학습·검증이 필요하다. `resume=True`로 과거 smoke 설정과 경로를 그대로 이어가지 않는다.

### YOLO 경로와 라벨

- 100개 복사 이미지가 PBR RGB와 해시로 일치했다.
- 100개 라벨이 클래스 0과 `bbox_visib` 정규화 좌표에 맞았다.
- train 80, val 20, test 0. 같은 이미지 ID 중복·train/val 교집합 없음.
- train 블록: 0,1,2,3,5,6,7,8. val 블록: 4,9. 블록은 프레임 ID 정수 나누기 10이다.
- 각 블록의 `cam_R_w2c`, `cam_t_w2c`, 물체 Pose로 복원한 world 물체 자세는 거의 일정했다. 블록 내 최대 이동 차이는 약 0.000103, 회전 원소 차이는 약 4.33e-7 이하로 같은 생성 배치라는 기록과 부합한다.
- 상대경로 `autosplit_*.txt`의 100개 참조는 해당 데이터 루트 기준으로 존재한다.
- `data_objs.yaml`의 `path`와 `args.yaml`의 data/project/save_dir에는 **Windows 절대경로**가 남아 있다. Linux에서 그대로 사용하면 안 된다. 새 설정 파일에 Linux 경로를 기록해야 한다.
- 기존 `labels.cache`는 재사용 신뢰성을 확인하지 않았다. 원본을 삭제하지 말고 새 작업 루트에서 재생성한다.

근거: PBR `yolo11/train_obj_s/`의 labels·images·autosplit 파일 및 `smoke_train/{args.yaml,results.csv,weights/}`.

## 4. HCCEPose 학습 스크립트와 현재 구조

**[확인]** `s4_p2_train_bf_pbr.py`는 BOP `train_pbr/000000/` 구조와 obj_id=4를 표현할 수 있다. 모델 정보 필드도 호환된다. 하지만 현재 하드코딩 값과 자산 누락 때문에 그대로 실행할 수 없다.

| 현재 구현 | 필요한 수정/준비 |
|---|---|
| dataset_path=`/root/xxxxxx/demo-tex-objs` | 승인 후 새 실험 데이터 루트 명시 |
| 물체 1~5 순회 | obj_id=4만 처리 |
| train_pbr 학습·같은 폴더 앞 1% 검증 | 기존 80/20 그룹 분할을 HCCE에도 명시 적용 |
| 기본 50,001 iteration | 100장 규모에 맞는 검증 중심 step budget·조기 종료 설계 |
| 배치24·worker12 | 초기 batch4·worker0부터 별도 승인된 smoke 점검 후 조정 |
| 앞/뒤 XYZ 라벨 읽기 | `train_pbr_xyz_GT_front/back` 생성 필요 |
| 라벨 누락 시 0 이미지 대체 | 학습 전 완전성 검사 및 누락 즉시 중단 |
| ResNet `pretrained=True` | 로컬 가중치 또는 명시적 무작위 초기화, 자동 다운로드 차단 |
| 데이터 루트 안 HccePose 체크포인트 저장·이전 파일 삭제 | 새 run 전용 저장 경로와 fail-if-exists 정책 |

HCCE front/back 라벨과 HCCE 체크포인트는 지정 PBR 경로에서 발견되지 않았다. 기존 depth 한 장만으로 뒷면 XYZ를 복원할 수 없으므로 CAD와 렌더링 과정이 필요하다. `s4_p1_gen_bf_labels.py`는 VisPy 렌더러에 의존한다. 현재 프로젝트에는 `bop_toolkit.zip`, `blenderproc.zip`만 있고 펼쳐진 도구 디렉터리가 없다.

이미 PBR이 있으므로 RGB Baseline 학습에 BlenderProc/bpy를 다시 설치해 전체 합성 데이터를 재생성할 필요는 없다. 반면 BOP toolkit·KASAL·XYZ 라벨 생성 렌더러는 현재 import/처리 경로에서 따로 준비해야 한다. BlenderProc와 XYZ 라벨 렌더러를 같은 요구사항으로 취급하지 않는다.

근거: 소스 `s4_p2_train_bf_pbr.py`, `s4_p1_gen_bf_labels.py`, `HccePose/bop_loader.py`, `HccePose/network_model.py`.

## 5. Python·CUDA·RTX 5090

| 항목 | 실제 확인 |
|---|---|
| OS | Ubuntu 24.04.4 LTS, x86 서버 |
| 지정 Python | 3.10.22 |
| PyTorch / torchvision | 모듈 미발견, torch import 실패 |
| NumPy / OpenCV / Pillow / SciPy | 지정 환경에서 미발견 |
| Ultralytics / imgaug / KASAL / Kornia / VisPy / OpenGL / pandas / trimesh | 지정 환경에서 미발견 |
| GPU | NVIDIA GeForce RTX 5090 |
| compute capability | 12.0 |
| 드라이버 | 595.99.02 |
| nvidia-smi의 CUDA 표시 | 13.2 |
| GPU 메모리 스냅샷 | 32,607 MiB 전체, 462 MiB 사용 |
| `nvcc` | 현재 PATH에서 발견하지 못함 |

`nvidia-smi`의 CUDA 13.2는 드라이버가 지원하는 CUDA 수준이며 PyTorch CUDA runtime이나 전체 Toolkit이 설치됐다는 증거가 아니다. 다른 위치의 Toolkit 부재까지 단정하지 않는다.

**[공식 근거]** RTX 5090은 compute capability 12.0이다. [NVIDIA GPU 목록](https://developer.nvidia.com/cuda/gpus). PyTorch는 2.7에서 Blackwell 지원과 CUDA 12.8 wheel을 소개했다. [PyTorch 2.7 발표](https://pytorch.org/blog/pytorch-2-7/).

**[제안]** 이 저장소가 명시한 `torch==2.8.0`, `torchvision==0.23.0`, CUDA 12.8 조합을 재현성 우선 후보로 검토한다. 공식 이전 버전 설치 문서에도 해당 버전별 배포 조합이 있다. [PyTorch 이전 버전 문서](https://pytorch.org/get-started/previous-versions/). 최신 버전이라는 뜻이 아니다. 설치할 정확한 wheel과 의존성 충돌 검토·승인은 별도로 필요하다.

현재 드라이버·장치 정보만으로 CPU 모델 로딩, cuDNN, AMP, torchvision 연산, sm_120 커널의 실제 성공을 인증하지 않는다. 환경 준비 후 CPU import·CPU 체크포인트 로딩·장치 조회·작은 CUDA 연산·단일 배치 순서로 검증해야 한다. 이번에는 CUDA 연산을 하지 않았다.

## 6. 실행 차단 요소와 메모리 위험

| 우선순위 | 문제 | 계획 |
|---|---|---|
| 필수 | 지정 환경에 torch 등 필수 패키지 없음 | 승인된 고정 버전 환경 준비. 자동 설치 경로는 사용하지 않음 |
| 필수 | PBR CAD 파일 없음 | 실제 전송 CAD 후보를 새 작업 루트에 독립 복사하고 검증 |
| 필수 | front/back XYZ 라벨 없음 | renderer·카메라·단위 검증 후 새 루트에서 생성 |
| 필수 | HCCE 학습 데이터 경로·obj_id 하드코딩 | 새 실험 설정/진입점으로 obj04·분할·출력 명시 |
| 필수 | HCCE 검증이 학습 자료와 중복 | train80/val20 분리 적용, val에서만 모델·PnP 후보 선정 |
| 필수 | 정상 YOLO 모델 로딩 미검증 | 환경 준비 후 CPU에서 `best.pt` 버전·task·names 확인 |
| 필수(E2E) | YOLO 성능 기록 모두0, fraction0.1 smoke | 새 run에서 충분히 학습하거나 적합한 로컬 pretrained 초기화 선택 |
| 필수(E2E) | Tester의 YOLO 경로가 best.pt 위치와 다름 | 외부 checkpoint 인자를 명시하거나 파생 표준 배치 사용 |
| 필수(실사) | RGB-only 로더와 실제 K·ROI·Mask 조건 미정 | 별도 어댑터·카메라 provenance·Pose-only/E2E 분리 |
| 정합성 | 3프레임의 1픽셀 GT-info 차이 | 원본 보존, 파생본 통계 재계산과 차이 로그 |
| 안정성 | checkpoint 로더가 예외를 숨기고 기본값 반환 | 명시적 초기 학습/재개 분기, 로딩 실패 시 중단 |
| 안정성 | auto 환경 준비·pretrained 다운로드 경로 | PyTorch 경로 고정, ONNX/보정 비활성화, local-only 검사 |
| 자원 | 기본 batch24가 32GB에서 안전한지 미측정 | batch4·worker0·256 crop부터 peak VRAM 실측 후 증량 |

`Tester`는 `yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt`를 기대하지만 발견된 파일은 `smoke_train/weights/best.pt`이다. HCCE는 `<dataset>/HccePose/obj_04/best_score/`를 기대한다. 실사와 PBR이 다른 루트라서 이를 자동 공유하지 않는다. 가중치 경로를 설정으로 분리해야 한다. 근거: `HccePose/tester.py:Tester.__init__`.

메모리 위험은 OOM 재현 결과가 아니라 정적 분석이다. 학습 코드가 train/test 네트워크를 동시에 유지하고 Adam 상태·활성화·AMP 메모리를 사용한다. 검증 `test()`는 eval 모드이지만 gradient 비활성화가 없어 메모리 낭비 가능성이 있다. `torch.cuda.empty_cache()`는 매 step 호출해도 활성 tensor 메모리를 없애지 못한다. 순수 HCCE 학습에서 실제 VRAM을 측정하기 전 정확한 배치 크기를 보장할 수 없다.

호스트 메모리는 125GiB, 조회 시 available 약113GiB, `/dev/shm` 약63GiB, 프로젝트 파일시스템 여유 약1.3TB였다. 조회 시점의 상태일 뿐 자원 예약이나 단독 사용을 뜻하지 않는다. 기본 DDP는 sampler·예외 경로 문제도 있어 단일 GPU로 시작한다.

## 7. 실제 SMT 27프레임 연결

기존 재투영 검토 결과를 유지한다: 개발용 조건부 사용, 27프레임/15종 정확히 다른 GT 행렬, 시각 성공22·판단불가5. 이 숫자는 모델 정확도가 아니다. 원본85파일 SHA-256은 이전 manifest와 모두 동일했다.

1. **Pose-only 우선:** 실사 RGB·확정 K·CAD·GT·수동 ROI 또는 명시적 GT 투영 ROI로 네트워크+PnP를 평가한다. GT 투영 ROI는 oracle 조건이며 E2E 성능이 아니다.
2. **E2E 별도:** 검증된 YOLO로 실제 ROI를 검출하고 검출 누락도 실패에 포함한다. 모델 names의 클래스0을 obj_id4로 매핑한다.
3. **입력 카메라:** 현재 무왜곡 PnP에 RAW K+D 영상을 D 무시 상태로 넣지 않는다. 기존 UND의 new K/ROI 이력을 확인하거나 승인된 통제 UND를 생성한다. same-K 가설을 쓰는 임시 결과는 명시한다.
4. **GT 평가:** `ratio=1.0`으로 전체27개를 읽고 object-to-camera·단위·프레임 ID를 보존한다. visible Mask가 없다면 가짜 Mask나 `visib_fract=1`을 만들지 않는다. Pose-only 평가 로더는 depth와 Mask 없이도 동작하도록 분리한다.
5. **경로:** PBR 학습 가중치를 실사 데이터 경로와 별도로 설정한다. 네트워크 백본·CAD 경계·BGR 정규화를 학습과 일치시킨다.
6. **지표:** 현재 모델 정보에는 대칭 필드가 없어 ADD를 기본 사용한다. ADD@0.1d와 원시 오차·실패율·회전 오차를 보고하고, mm는 물리 단위 확인 후 사용한다. 필요하면 ADD-S를 보조 지표로 명시하되 성능이 좋아지는 방향으로 대칭 설정을 바꾸지 않는다.

근거: [기존 CPU 재투영 보고서](reprojection_report_ko.md), `HccePose/PnP_solver.py`, `HccePose/metric.py`, `HccePose/tester.py`.

## 8. 누수 방지와 실제 실행 순서 — 미실행 계획

### 분할

- 기존 YOLO train80/val20은 이미지가 겹치지 않고 10프레임 생성 블록도 분리되어 있다. 이 분할을 HCCE 학습·검증에도 명시적으로 적용한다. 현재 모든 파일이 `000000` 안에 있다는 이유로 장면 단위 분할이 자동 되는 것은 아니다.
- HCCE의 `train_pbr` 앞1% 검증은 폐기해야 한다. 현재100프레임이면 학습에 포함된 첫1프레임만 검증하게 된다.
- PBR과 전송 RAW/UND 이미지 간 SHA-256 완전 일치 중복은 0이다. 이것은 장면·분포 설계 과정의 간접 정보 사용까지 배제하는 증거가 아니다.
- `reference_poses_used_as_training=false`는 provenance 기재값으로 기록한다. 생성 스크립트·로그가 없으므로 포즈 선택·샘플링 설계의 독립성 전체를 인증하지 않는다.
- 현재 실사27은 dev 평가로 유지하며 가중치/PnP 선택은 PBR val20에서 수행한다. 실사 dev 결과를 보고 튜닝하면 후속 실험도 dev 결과로만 보고한다. 별도 실사 테스트를 새 촬영 세션으로 확보한다.

### 승인 후 실행 단계

| 순서 | 작업 | 다음 단계 진입 조건 |
|---|---|---|
| 1 | 원본 해시·파일 목록 고정, 새 run/data/config 경로 생성 | 기존 파일 무변경·출력 덮어쓰기 없음 |
| 2 | 환경 구성 계획 승인 후 PyTorch CUDA·필수 RGB 학습 패키지 준비 | torch/torchvision/cv2/NumPy/Ultralytics 및 소스 import 성공 |
| 3 | 작은 CUDA 호환성 검사와 CPU YOLO 로딩 | sm_120 지원, CUDA·AMP/필요 연산 확인, task=detect·names 매핑 확인 |
| 4 | 파생 데이터 루트에 CAD 배치, Linux 설정·80/20 목록 작성, GT-info 통계 정리 | 모델·카메라·마스크·분할 재검증 통과 |
| 5 | 앞·뒷면 XYZ 라벨 생성 및 100프레임 완전성 검증 | 앞뒤 라벨·Mask·crop 좌표 정합, 누락0 |
| 6 | 단일 배치 로딩/forward/backward 자원 smoke 검사 | 별도 GPU 실행 승인 후, 유한 loss·gradient·peak VRAM 확인 |
| 7 | HCCE Pose-only Baseline 학습 | obj4, train80/val20, 처음 batch4·worker0·256 crop·기본 ResNet, 로컬 초기화 명시 |
| 8 | val20으로 checkpoint와 PnP 후보 통계 결정 | 실사 dev를 모델 선택에 사용하지 않은 조건 기록 |
| 9 | 실사27 Pose-only 추론·GT 평가 | 카메라·ROI·단위·결과 경로 확정, 전27과 품질그룹별 보고 |
| 10 | YOLO 추가 학습/검증 후 실사 E2E 평가 | smoke 가중치만으로 검출 성능 충분하다고 가정하지 않음 |

50,001 step을 그대로 시작하지 않는다. 80장·batch4이면 drop_last 조건에서 20 step이 한 epoch에 해당한다. 초기 검증 간격·최대 step·조기 종료를 새 설정에 고정하고 과적합을 관찰한다. 100장의 소규모 실험이므로 높은 일반화 정확도를 기대한다는 주장을 하지 않는다.

현재 학습 스크립트는 데이터 경로·물체 ID용 CLI가 없어 **지금 실행할 유효한 학습 명령을 확정할 수 없다.** 설치와 최소 수정 승인 후 새 진입점과 고정 설정 파일을 작성하고, 정확한 명령·출력 경로·자원 예산을 제시한 뒤 학습을 시작해야 한다. 이번에는 학습 명령을 실행하거나 존재하지 않는 진입점의 명령을 실행 가능한 것처럼 제시하지 않는다.

## 9. 승인 전에 확정할 수정 목록

**필수 준비:** 로컬 학습 환경, 파생 CAD 배치, Linux 경로 설정, XYZ 라벨 생성, train/val 분리, local-only 초기화·checkpoint fail-fast, RGB-only 실사 평가 어댑터, YOLO 경로 설정.

예상 추가 경로는 `experiments/obj04/`의 설정·학습/평가 어댑터·검증 도구와 새 파생 데이터 및 `runs/obj04/<run_id>/`이다. 원본 PBR/실사, `smoke_train`, 기존 출력은 건드리지 않는다. 기존 `network_model.py`의 초기화 제어 등 수정이 필요하면 구체적 diff를 먼저 제시한다. `best.pt`·`labels.cache`·Windows 설정을 덮어쓰거나 삭제하지 않는다.

### 최종 준비 상태

| 구분 | 판단 |
|---|---|
| PBR RGB·Depth·Mask·카메라·Pose | 재사용 후보로 준비됨. 픽셀 통계3프레임 정리 필요 |
| PBR CAD | 해당 루트에서 누락. 실제 전송 CAD는 강한 호환 후보 |
| YOLO 파일 | 구조 무결성 확인. 모델 정상 로딩 미검증 |
| YOLO 정확도 | smoke 기록 모두0. 유효한 검출기로 준비 완료 아님 |
| RTX 5090 하드웨어·드라이버 | 인식 확인. PyTorch 실제 CUDA 동작 미검증 |
| hccepose 환경 | 필수 패키지 부재로 준비 미완료 |
| HCCE 학습 | XYZ 라벨·설정·검증 분리·출력 격리가 필요하여 즉시 실행 불가 |
| 실사27 추론·평가 | 개발 평가 후보 유지. RGB-only·카메라·ROI 연결 필요 |

**권고:** PBR을 보존·재사용하면서 먼저 환경·CAD·XYZ·분할을 준비하고 HCCE Pose-only Baseline을 구축한다. 기존 YOLO best.pt는 별도 검증·추가 학습 대상으로 두고, 완성된 검출기로 전제해 실사 E2E 정확도를 평가하지 않는다.
