# HCCEPose CPU 사전 검증 계획

상태: **테스트 코드 준비, 미실행. 실행 승인 필요.**

## 범위와 실행 방식

실제 SMT 자료·CAD·체크포인트 없이 합성 입력만 사용한다. 프로젝트 원본을 변경하거나 전체 모듈을 import하지 않는다. `source_loader.py`는 원본 Python을 AST로 읽어 지정 함수만 원문 그대로 컴파일한다. 클래스 생성자·YOLO·렌더러·환경 준비 함수는 호출하지 않는다. BOP의 `add`, `adi`, `transform_pts_Rt`는 기존 ZIP 내부 소스를 읽으며 압축 해제하지 않는다. 함수별 원본 파일 SHA-256과 행 번호를 결과에 기록한다.

이 방식은 원본 함수의 수학·실패 처리를 검증하지만 패키지 전체 import, 모델 forward, 실제 이미지 추론, 설치 정상 여부를 검증하지 않는다. 독립 oracle와 원본 코드 검증을 구분해 보고한다. `Tester.predict`의 resize/pad/K 구간만 AST로 분리하며 GPU 이름이 포함되면 실행을 거부한다.

필요 패키지: NumPy, CPU tensor 연산이 가능한 PyTorch, OpenCV, SciPy. Kornia는 실제 추론 crop 샘플링 검사에 필요하다. 누락·로드 실패는 SKIP과 원인을 기록한다. 자동 pip/conda 설치, 다운로드, 환경 생성은 없다. `--check-deps`는 설치 메타데이터와 모듈 존재만 확인하므로 바이너리 import 성공까지 보장하지 않는다.

## 순서와 판정

| 단계 | 원본 대상 | 검증과 기준 |
|---|---|---|
| 1 | `bop_loader.py`: 학습/평가 `hcce_encode`; `network_model.py`: `hcce_decode` | 축별 0~255 전수, XYZ 채널 분리, 독립 Gray code 비교, 최대 정수 복원 오차 0, uint8/float 입력 일치 |
| 2 | `Tester.predict` 전처리 구간, `crop_trans_batch_hccepose`, `pad_square_fp32`, `crop_square_resize` | K'=TK, 주점 이동·원본 K 보존, crop 역변환 <=3e-5px, OpenCV ramp, Kornia `align_corners=False` ramp <=0.05px |
| 3 | `PnP_solver.py`: `solve_PnP`, `solve_PnP_comb` | 비평면 192점·3개 seed, pinhole 독립 투영. 무잡음 <=0.01deg/0.01mm, 0.25px 잡음+38 outlier <=1deg/2mm. 0/3/4점 실패·강제 OpenCV False 반환·앞뒤 조합 검사 |
| 4 | ZIP BOP `add`, `adi`, `misc.transform_pts_Rt`; `metric.py`: `add_s` | 독립 brute-force 최근접 거리, 3-4-0 이동 ADD=5mm, 대칭 회전 ADD-S=0, 엄격한 직경10% 경계, 입력 길이 불일치·빈 목록·NaN·음수 직경 거부 |

허용오차는 작은 합성 장면의 수치 정확성 기준이며 SMT 공정 허용오차가 아니다. 잡음 테스트는 확률적 성능 주장이 아닌 고정 seed 회귀 검사이다. 좌표 물체 범위 복원은 독립 계산이며 `inference_batch`나 네트워크를 실행하지 않는다. K resize 검사는 원본의 명목 배율을 검증하고 정수 영상 크기 배율과의 차이도 진단 수치로 남긴다. Kornia 실제 샘플링은 별도 테스트이므로 수학적 행렬 일치만으로 전체 crop 성공을 선언하지 않는다.

## 승인 후 사용할 명령

작업 디렉터리: `/RESEARCH_DATA/HCCEPose`. `python`은 사용할 기존 인터프리터를 의미한다. 임의로 환경을 활성화하거나 패키지를 설치하지 않는다.

먼저 의존성 확인만 수행:

```bash
python -B cpu_preflight/run_checks.py --check-deps
```

승인한 CPU 검증 전체 실행:

```bash
python -B cpu_preflight/run_checks.py --run --stage all
```

단계별 승인·실행도 가능:

```bash
python -B cpu_preflight/run_checks.py --run --stage 1
python -B cpu_preflight/run_checks.py --run --stage 2
python -B cpu_preflight/run_checks.py --run --stage 3
python -B cpu_preflight/run_checks.py --run --stage 4
```

`--run` 없이 실행하면 테스트를 수행하지 않는다. `--check-deps`와 `--run`은 동시에 허용하지 않는다. 종료코드: 0=전부 PASS, 1=FAIL 존재, 2=FAIL은 없지만 SKIP 존재. 실패를 예상하더라도 성공으로 바꾸거나 숨기지 않는다. 한 검사의 오류는 기록하고 이후 검사도 계속한다.

## 예상 부작용·자원

- `--check-deps`: stdout 출력만 하며 테스트 결과 디렉터리를 만들지 않는다.
- `--run`: `cpu_preflight/results/<UTC 시각>/results.json`, `report_ko.md`를 새로 생성한다. 기존 결과를 덮어쓰지 않는다. `--output-dir`도 이 하위 새 경로만 허용한다.
- 프로세스 내부에서만 `CUDA_VISIBLE_DEVICES`를 비우고 CPU thread 수를 1로 제한한다. OpenCV OpenCL을 비활성화한다. GPU 조회·CUDA 연산은 호출하지 않는다. 셸·서버 설정 파일은 변경하지 않는다.
- 최대 합성 영상은 1280×720이며 대규모 데이터나 모델을 로드하지 않는다. 첫 PyTorch/SciPy import 시간과 메모리 비용은 발생한다. 실행시간·peak RAM은 아직 측정하지 않았으며 timeout 없이 즉시 종료된다고 보장하지 않는다.
- OpenCV RNG seed 설정, 강제 solver 실패 모킹은 테스트 프로세스 안에서만 수행한다. 모킹은 컨텍스트 종료 시 복원한다.
- `-B`와 `sys.dont_write_bytecode`로 Python bytecode 쓰기를 막는다. 서드파티 라이브러리의 플랫폼별 내부 동작까지 OS 수준에서 격리하는 도구는 아니다. 엄격한 파일 접근 격리가 필요하면 별도 승인된 sandbox에서 실행한다.
- 외부 통신·pip·conda·체크포인트 역직렬화·다운로드 API를 호출하지 않는다.

## 사전 정적 검토에서 확인한 위험과 해석

다음은 실행 결과가 아니다.

1. 원본 PnP가 solver 실패 확인 전에 Rodrigues/재투영을 호출하는 경로가 있다. 강제 False 반환 테스트는 예외를 FAIL로 기록할 가능성이 있다.
2. 원본 `metric.add_s`는 `zip`으로 길이를 절단하고 입력 유효성 검사가 없다. 길이 불일치·빈 목록·NaN·음수 직경 거부 테스트는 현재 구현의 계약 공백을 드러내기 위한 것이다. 이 계약은 테스트 계획에서 요구하는 평가 신뢰성 기준이며 원본이 이미 보장하는 기능이라는 뜻이 아니다.
3. NumPy 버전에 따라 uint8에 `% 256`을 적용하는 코드의 호환성 차이가 있을 수 있다. 실제 실행 전에는 해당 서버의 오류 발생 여부를 확정하지 않는다.
4. 학습 float crop과 추론 경계 반올림, Kornia pixel-center 규약은 각각 분리해서 평가한다. 실제 오차와 수정 필요성은 승인 후 결과로 판단한다.

결과 보고서는 PASS/FAIL/SKIP, 허용오차, 수치, 원인과 traceback을 남긴다. 결과는 CPU 수학 구현 검증이지 SMT 정확도 또는 GPU 추론 성능 검증이 아니다. 원본 수정과 재검증은 별도 승인 대상이다.
