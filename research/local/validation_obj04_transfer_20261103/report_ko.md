# SMT obj04 CAD·카메라·Pose GT 1차 정합성 검증

- 검증일: 2026-11-03 (Asia/Seoul)
- 입력: `/RESEARCH_DATA/HCCEPose_obj04_transfer`
- 코드 참조: `/RESEARCH_DATA/HCCEPose`
- 결과 폴더: `/RESEARCH_DATA/HCCEPose/validation_obj04_transfer_20261103`
- 상태: **구조·수치 검증 완료. CAD 2D 재투영은 실행 승인 대기.**
- 방법: Python 표준 라이브러리로 NPY·JSON·ASCII PLY·JPEG 헤더를 읽었다. NPY는 제한된 float32/64 4×4 행렬만 직접 파싱했으며 pickle을 사용하지 않았다. GPU·모델·패키지 설치·다운로드는 사용하지 않았다.
- 원본 85파일의 크기와 SHA-256을 `numeric_results.json`에 저장했으며 검증 후 다시 비교한 결과 모두 동일했다. 기존 HCCEPose 소스와 기존 보고서는 수정하지 않았다.

## 1. 전송 구조와 프레임 번호

**[확인]** 원본은 85파일, 합계 14,807,111바이트이다.

```text
HCCEPose_obj04_transfer/
├── image/                      27 JPEG
├── image_undistorted/           27 JPEG
├── poses_obj04/                 27 pose_XXXXXX.npy
├── obj_000004.ply               131,106 B
├── scene_gt_obj04.json          12,600 B
├── scene_camera_obj04.json       5,625 B
└── camera_params.json              371 B
```

Pose NPY, 원본 RGB, 왜곡 보정 RGB, GT JSON 키, Camera JSON 키의 프레임 집합이 모두 동일하다. 누락·추가 프레임은 없다. 각 RGB 폴더 안에서 SHA-256이 같은 중복 파일도 발견하지 않았다. JPEG 헤더에서 54장 모두 1280×720임을 확인했다. 전체 JPEG 디코딩 무결성 검사는 아직 하지 않았다.

```text
26, 32, 38, 39, 50, 51, 56, 61, 65, 66, 68, 70, 72, 73,
78, 83, 88, 91, 92, 94, 96, 98, 99, 108, 118, 119, 126
```

숫자형 JSON 키 `"26"`과 `pose_000026.npy`·`000026.jpg`를 정수 프레임 ID로 비교했다. 파일 번호 일치는 실제 촬영 시각과 Pose 대응 정확성까지 입증하지 않는다.

근거: 입력 `poses_obj04/`, `image/`, `image_undistorted/`, 두 scene JSON; [audit_numeric.py](audit_numeric.py), [numeric_results.json](numeric_results.json).

## 2. NPY Pose와 scene_gt 비교

**[확인]** 27개 모두 유한값의 4×4 float 행렬이다. 마지막 행은 `[0,0,0,1]`과 정확히 일치한다. 각 JSON 프레임은 `obj_id=4` 인스턴스 하나를 가진다.

| 항목 | 실제 검증값 |
|---|---:|
| NPY R 대 JSON `cam_R_m2c` 최대 원소 차이 | 0.0 |
| NPY t 대 JSON `cam_t_m2c` 최대 원소 차이 | 0.0 |
| 최대 `abs(RᵀR-I)` 원소 | 1.5543122344752192e-15 |
| det(R) 최솟값 | 0.9999999999999991 |
| det(R) 최댓값 | 1.0000000000000022 |
| 마지막 행 최대 오차 | 0.0 |

t에 1000 또는 0.001을 곱하는 비교보다 **스케일 1.0에서 정확히 일치**한다. 현재 NPY→JSON 사이에 추가 단위 변환은 필요하지 않다.

| JSON 이동 성분 | 최솟값 | 최댓값 |
|---|---:|---:|
| tx | -176.62349765198456 | 32.91070656235774 |
| ty | -98.8937785319268 | -15.076276738743168 |
| tz | 368.49733245793834 | 522.7356182029769 |

**[미확인]** NPY와 JSON이 동일하다는 것은 내보내기 정합성이다. 실제 물체 위치·자세가 맞는지, NPY 생성 도구가 실제로 object-to-camera를 사용했는지는 재투영과 생성 이력으로 확인해야 한다. JSON 필드명과 HCCEPose는 object-to-camera 규약 `Xc=R Xm+t`를 사용한다.

근거: 입력 `scene_gt_obj04.json`, 27 NPY; 코드 `HccePose/bop_loader.py`의 자세 로딩과 좌표 복원.

## 3. 카메라 파라미터

**[확인]** 27개 Camera JSON의 9원소 K는 `camera_params.json`의 `camera_matrix`와 최대 원소 차이 0.0으로 모두 동일하다.

```text
K = [[984.70009004,   0.0,        635.41416344],
     [  0.0,        983.75715611, 362.61850196],
     [  0.0,          0.0,          1.0       ]]

D = [-0.28801442, -0.07604108, -0.00069984, 0.00055886, 0.12258375]
```

명시된 이미지 크기 1280×720도 JPEG 헤더와 일치한다. `depth_scale`은 1.0이며 Depth 파일은 없다. 이 값만으로 CAD·이동량의 물리 단위를 정할 수 없다.

**[파일 기재값]** `camera_params.json`은 `reprojection_error_px=0.0487`을 포함한다. 이번 27장 또는 CAD를 이용해 재계산한 값이 아니며 캘리브레이션 표적·절차도 전송 자료에 없으므로 현재 GT 정합성 오차로 해석하지 않는다.

**[중요한 미확인 사항]** `image_undistorted/` 생성에 사용한 new K, alpha, ROI crop, resize, 좌표 원점 이동 이력이 없다. 같은 해상도라고 해서 K가 그대로라는 결론은 성립하지 않는다.

- 원본 영상에는 보통 K와 D를 함께 사용한다.
- 왜곡 보정 영상에는 생성 시 실제 사용한 출력 K와 필요시 crop/resize 보정을 사용해야 한다.
- BOP `cam_K`에 D가 없으므로 원본 왜곡 영상을 K만으로 처리하는 구성은 현재 HCCEPose의 무왜곡 PnP 가정과 일치하지 않는다.

근거: 입력 `camera_params.json`, `scene_camera_obj04.json`; 코드 `HccePose/PnP_solver.py`의 `distCoeffs=None`, `HccePose/tester.py`의 K resize/pad 처리.

## 4. CAD 좌표·크기·단위와 GT 호환성

**[확인]** ASCII PLY, 꼭짓점 1,439개, 삼각면 2,970개이며 XYZ와 법선 속성을 가진다. 텍스처·물리 단위·축의 의미는 헤더에 명시되어 있지 않다.

| 항목 | CAD 원본 좌표값 |
|---|---|
| min XYZ | (-45.25, -45.25, -32.65764) |
| max XYZ | (45.25, 45.25, 32.65764) |
| size XYZ | (90.5, 90.5, 65.31528) |
| AABB 중심 | (0, 0, 0) |
| 꼭짓점 쌍 최대 거리 | 139.37869509655914 |
| 면적 0에 가까운 삼각면 | 0 |
| 한 면에만 속하는 경계 edge | 0 |
| 세 면 이상에 속하는 edge | 0 |

위 edge 검사는 닫힌 표면에 부합하지만 자기 교차·일관된 winding·물리적 형상 정확성까지 검증하지 않는다.

**[확인]** CAD와 JSON t를 같은 단위로 해석하고 `Xc=R Xm+t`를 적용했을 때, 모든 프레임의 모든 CAD 꼭짓점이 양의 Z에 있다. 전체 Z 범위는 **320.5886201636834~581.7286400522748**이다. 이는 3D 카메라 전방 검증이며 2D 재투영은 아직 수행하지 않았다.

**[추정]** CAD 크기와 카메라 거리 수치는 mm 해석에 부합할 수 있다. 다만 원본에 물리 단위 메타데이터가 없으므로 mm라고 확정하지 않는다. 실제 obj04 치수 또는 내보내기 단위 확인이 필요하다. 축의 실제 의미와 GT와의 방향 일치도 재투영 전에는 확정할 수 없다.

현재 CAD는 이미 AABB 중심이 원점이다. 기존 `s1_p1_obj_rename_center.py`를 무조건 다시 적용하지 않는다. CAD 좌표를 변경하면 대응 Pose도 `T_new = T_old * A^{-1}`처럼 동일한 좌표 변환에 맞춰 갱신해야 한다 (`Xnew=A Xold`). 스케일·강체 변환은 구분해서 처리한다.

근거: 입력 `obj_000004.ply`; `audit_numeric.py`의 mesh·rotation·3D Z 검사; 기존 코드 `s1_p1_obj_rename_center.py`, `s1_p3_obj_infos.py`, `HccePose/network_model.py`의 min/size 복원.

## 5. BOP 변환에 필요한 사항 — 아직 변환하지 않음

| 항목 | 필요한 조치 |
|---|---|
| 디렉터리 | 별도 파생 루트에 `models/obj_000004.ply`, 예를 들어 `test/000000/rgb/000026.jpg` 구성 |
| JSON 이름 | 파생 scene 디렉터리에 `scene_gt.json`, `scene_camera.json`으로 저장. 기존 수치와 원본 프레임 ID 유지 가능 |
| `models_info.json` | ID `"4"`에 min/max/size와 직경 기록. 위 값은 CAD 원본 단위이며 물리 단위 확인 후 확정 |
| 대칭 | 실제 형상·극성·공정 방향을 확인한 뒤 설정. 대칭 정보를 임의로 추가하지 않음 |
| 영상·K 선택 | undistorted 생성 이력 확인 후 해당 영상과 실제 출력 K를 짝지음 |
| Mask·BBox | `mask/`, `mask_visib/`, `scene_gt_info.json`이 현재 없음. 렌더링 amodal Mask와 실제 visible Mask를 구분해서 준비 |
| 가림 | CAD 자체 렌더링만으로 다른 물체의 가림을 반영한 가시 Mask를 만들 수 없음. 가시율·픽셀 수를 임의로 생성하지 않음 |
| Depth | 파일이 없어 기존 `bop_loader.load_folder`의 Depth 확장자 처리와 충돌 가능. RGB-only 로더 지원은 향후 별도 코드 변경 승인 대상 |
| HCCE 라벨 | 학습용으로 사용할 경우 별도 front/back XYZ 라벨 필요. 현재 미생성 |
| 데이터 split | 이 27장을 GT 작성·정합성 조정에 쓰는 경우 최종 독립 평가 자료로 자동 취급하지 않음 |

현재 자료는 **Pose·카메라·CAD가 있는 부분 BOP 구성**이며, 기존 학습·GT BBox 평가 로더에 바로 넣을 완성 데이터셋은 아니다. 원본 변환·복사와 기존 HCCEPose 코드 수정은 이번 단계에서 하지 않았다.

근거: 코드 `HccePose/bop_loader.py`의 `load_folder`, `test_bop_dataset_back_front`; `s4_p1_gen_bf_labels.py`, `s1_p3_obj_infos.py`, `HccePose/metric.py`.

## 6. CPU 재투영 실행 계획과 승인 요청

### 설치 환경 확인

다음 두 기존 인터프리터에서 모듈 검색·설치 메타데이터를 확인했다. 환경 활성화·설치는 하지 않았다.

```text
/RESEARCH_DATA/USER/miniconda3/bin/python
/RESEARCH_DATA/USER/miniconda3/envs/hccepose/bin/python
```

두 환경 모두 NumPy, OpenCV, Pillow, SciPy, trimesh, Open3D, pyrender, torch 모듈을 찾지 못했다. 따라서 기존 VisPy/OpenGL 렌더링 경로를 이 환경에서 바로 실행할 수 있다고 보고하지 않는다. 현재 셸 PATH에서도 ImageMagick·ffmpeg·Blender 실행 파일을 찾지 못했다. 다른 모든 환경·컨테이너까지 조사한 것은 아니다.

### 설치 없는 대안

준비한 [render_cpu.py](render_cpu.py)는 표준 라이브러리만 사용한다. PLY 삼각면을 직접 투영하고 CPU z-buffer로 CAD의 자기 가림을 처리하며, PNG 투명 실루엣과 JPEG 원본을 내장한 SVG 오버레이를 저장한다. JPEG 픽셀 디코딩은 하지 않으므로 OpenCV/Pillow가 없어도 파일 생성이 가능하다. 파일을 표시하는 viewer의 JPEG 지원은 필요하다.

프레임별 두 경로를 비교한다.

1. `image/` + 원본 K + Brown-Conrady 5계수 D. 계수 순서는 OpenCV 관례 `(k1,k2,p1,p2,k3)`라는 가정이다.
2. `image_undistorted/` + 같은 K + 무왜곡. **같은 K로 생성됐다는 미확인 가설**을 이름에 표시한다.

녹색 CAD 채움·윤곽과 빨강/초록/파랑 XYZ 축을 겹쳐 위치·크기·방향을 육안 점검한다. Mask GT가 없어 IoU·평균 contour 오차 같은 정량 정합성 점수는 자동 산출하지 않는다. 원본에 대한 왜곡 투영은 삼각형 꼭짓점에서 계산하고 면 내부는 직선 근사하므로 강한 왜곡에서 곡면 경계의 근사 오차가 있을 수 있다. 방향의 미세한 차이·폐색·대칭 모호성은 시각 검토로 별도 기록한다.

실행 전 원본 SHA-256이 1차 검증 시점과 같은지 확인하며 다르면 중단한다. 스크립트는 구문 분석만 통과했으며 **렌더링 동작은 아직 실행·검증하지 않았다.**

### 승인 후 명령

작업 디렉터리 `/RESEARCH_DATA/HCCEPose`:

```bash
/RESEARCH_DATA/USER/miniconda3/bin/python -B validation_obj04_transfer_20261103/render_cpu.py --run
```

### 예상 부작용

- 원본 읽기와 별도 검증 폴더의 새 `reprojection_cpu/` 생성만 수행한다.
- 27프레임 × 2경로 = SVG 54개와 CAD 투명 PNG 54개, `index.html`, `projection_metadata.json` 총 110개를 생성한다.
- 기존 출력 폴더가 있으면 덮어쓰지 않고 중단한다. 중간 오류 시 생성된 부분 결과가 남을 수 있다.
- 수십 MB 수준의 출력이 예상되나 미측정이다. CPU 순차 실행이며 수분 이상 걸릴 가능성이 있고 실제 소요시간·최대 메모리는 미측정이다.
- CUDA·GPU·학습·모델 추론·OpenGL·외부 통신·자동 설치·환경 변경을 하지 않는다. Python bytecode도 생성하지 않도록 `-B`를 사용한다.
- GUI나 브라우저를 자동으로 실행하지 않는다. 승인 후 생성 결과를 읽고 한국어 시각 검토 결과를 별도로 작성한다.

**승인 대기:** 위 두 경로의 CPU CAD 재투영과 별도 이미지 출력, 결과 검토를 진행할지 확인이 필요하다. `image_undistorted/` 생성 코드 또는 실제 출력 K가 있으면 두 번째 경로의 가설을 확정할 수 있다.
