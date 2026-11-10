import csv,json,collections
from pathlib import Path
P=Path(__file__).resolve().parent

def read(path):
 with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(path,rs):
 with path.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
a=read(P/'bb100_frame_audit.csv');groups=read(P/'pose_groups.csv');s=json.loads((P/'audit_summary.json').read_text());ov=json.loads((P/'overlay_rasterization_check.json').read_text());
# These two visual notes are preliminary observations only, not independent GT certification.
for r in a:
 if r['frame']=='43':r['visual_review_status']='assistant preliminary visual review only; independent review pending';r['occlusion_status']='no obvious tool over projection in this image; not validated';r['boundary_status']='outer contour approximately aligned by visual inspection; not quantified'
 if r['frame']=='235':r['visual_review_status']='assistant preliminary visual review only; independent review pending';r['occlusion_status']='gripper overlaps projected support; visible portion not segmented';r['boundary_status']='partially occluded; independent inspection pending'
write(P/'bb100_frame_audit.csv',a)
repeat=[g for g in groups if int(g['n'])>1]
row='\n'.join(f"| {g['pose_group']} | {g['n']} | {g['frames']} | {g['states']} | {float(g['neighbor_thumbnail_mae_mean_0_255']):.2f} / {float(g['neighbor_thumbnail_mae_max_0_255']):.2f} |" for g in repeat)
(P/'중복_pose_보고서.md').write_text(f'''# BB100 동일 Pose 및 촬영 그룹

BOP pose R,t를 `pose_*.npy`에서 float64로 읽어 값이 정확히 같은 R,t끼리 묶었다(−0은 +0으로 정규화; 그 외 반올림·허용오차 없음). 전체 **36그룹, 그중 중복 11그룹 75장**, 나머지 25장은 단독이다. 같은 수치가 곧 라벨 오류라는 뜻은 아니다.

| 그룹 ID | 장수 | 촬영 순서 프레임 | 상태 | 연속 이미지 축소 MAE 평균 / 최대 |
|---:|---:|---|---|---:|
{row}

이미지 차이는 라벨 작업에 사용된 JPEG를 160×90으로 축소한 인접 프레임의 채널 평균 절대 차이(0–255)다. 움직임에는 그리퍼·반사·조명·물체가 모두 포함되므로 물체가 움직였다는 증거도, 동일 pose의 정확성을 보증하는 지표도 아니다. 촬영 시간과 전체 그룹은 `pose_groups.csv` 참조.

`continuous_BB_obj04/legacy_000060.json`에는 **060이 043 pose 복사본**이라는 사용자 제공 이력과 두 pose 파일 바이트 일치가 명시됐다. 같은 pose가 여러 프레임에서 발견된 다른 경우에는 복사 또는 라벨 재사용이 가능하지만 그 원인·오류 여부는 확정할 수 없다. `continuous_BB_obj04/frame_000090/provenance.json`은 신규 Vision6D export와 이전 export 내용 동일 여부를 기록하나 전체 100장의 개별 라벨링 이력으로 일반화할 수 없다.

기존 `bb100_audit.csv`의 시간 1초 이내 또는 평행이동 5 CAD 단위 이내·회전 5° 이내라는 탐색적 연결 기준을 보수적 누출 방지 그룹으로 재사용했다. 전이 연결 성분은 31 그룹 36장, 221 그룹 64장이고 모든 정확히 같은 pose 그룹은 하나의 연결 성분 안에 들어간다. 둘 다 **같은 촬영 세션**이므로 별도 그룹이라도 독립 촬영 테스트를 보장하지 않는다.
''',encoding='utf8')
(P/'카메라_보정_출처.md').write_text(f'''# BB100 카메라 보정 출처와 한계

## 확인된 사실

- 라벨 작업 영상은 `real_bb100_20261108/rgb_undistorted/`의 JPEG이며 라벨링 도구 `BB_obj04/pose_watcher_continuous_BB.py`는 `rgb_undistorted` 경로를 가리킨다. 이름과 도구 경로는 보정 처리의 출처 증명이 아니다.
- RAW 후보는 `qwen_smt/data/BB_rgb_obj04/run_20260813_164421/rgb/`이며 frame mapping과 파일명이 BB100에 대응한다. 두 영상군의 크기는 모두 1280×720이다.
- `BB_obj04/build_obj04_bop.py`는 `scene_camera_obj04.json`에 K를 하드코딩한다. JSON에는 D나 newK가 기록되지 않는다. 100장의 K는 개발27의 `HCCEPose_obj04_transfer/camera_params.json` K와 수치상 일치(최대 차이 {s['max_K_difference_dev27']})한다. 이는 같은 카메라나 같은 보정 처리를 독립적으로 입증하지 않는다.
- 개발27 K,D와 `newK=K`를 **가정한 실험적 비교**에서 기존 라벨 JPEG와 RAW/한 번 보정/두 번 보정 영상의 픽셀별 평균 MAE는 각각 {s['mean_mae_raw']:.3f}, {s['mean_mae_one_devKD']:.3f}, {s['mean_mae_two_devKD']:.3f}/255였다. 100장 모두 가정한 1회 영상이 가정한 2회 영상보다 가까웠다. JPEG 압축·보간·border와 기타 처리가 섞일 수 있다.

## 미확인

BB 카메라 고유 K,D의 측정 기록, 기존 JPEG의 정확한 보정 코드·횟수·newK·crop/resize 이력은 발견하지 못했다. 숫자와 픽셀 유사성은 개발27 D를 BB의 정답이라고 증명하지 않으며, 가능한 모든 후처리 조합을 배제하지 못하므로 **이중 보정이 없었다고 확정하지 않는다**. 해상도 동일성은 좌표계 동일성을 증명하지 않는다. 큰 K/D 불일치가 확인된 것은 아니지만 BB 전용 D를 확인할 수 없어 불일치가 없다고도 할 수 없다.

이 작업의 100개 그림은 라벨 작업에 사용된 기존 JPEG 위에 BOP K와 pose로 CAD를 1회 투영했으며, 이미지에 보정을 추가하지 않았다. 기존 JPEG의 보정 조건을 알 수 없어 영상과 투영이 일치하더라도 정밀 GT 인증으로 해석하지 않는다. 모든 프레임은 해당 문제가 해결될 때까지 HOLD다.

우선 확인할 자료: 해당 촬영 당시 원본 calibration 파일·체커보드/타깃 및 카메라 ID, 기존 `rgb_undistorted` 생성 스크립트와 파라미터, 해상도·newK·crop 이력. 이를 얻으면 RAW에서 별도 출력으로 1회 보정하고 알려진 타깃의 재투영 잔차를 평가한다. 기존 라벨/원본은 수정하지 않는다.
''',encoding='utf8')
(P/'학습_검증_분할_초안.md').write_text('''# 학습·검증 분할 초안 — 지금 실행 금지

`split_draft.csv`는 기존 연결 그룹 31(36장)을 후보 검증, 221(64장)을 후보 학습으로 임시 배정한 표다. 모든 정확히 같은 pose 그룹을 한쪽에만 두며, 시간·근접 pose 탐색적 연결 기준도 넘지 않도록 기존 두 성분을 유지한다. 데이터의 품질·성능을 보고 역할을 최적화한 분할이 아니다.

**두 성분 모두 같은 촬영 세션이며 개발27과의 독립성도 미확인이다.** 따라서 독립 테스트셋·최종 확정 split이 아니고, 프레임 모두 `usable_now=False` / HOLD다. 지금은 학습·평가에 이 초안을 사용하지 않는다. 분할 목표는 향후 누출 검토를 위한 구조 예시일 뿐 실제 독립성을 보장하지 않는다.

BB K,D/newK와 pose 정확도를 독립적으로 검증하고, 별도 날짜·장면의 촬영 세션을 확보한 뒤 세션 및 연속 구간·동일/근접 pose를 연결하여 새 split을 확정한다. 100장의 인접 프레임을 무작위로 나누지 않는다.
''',encoding='utf8')
(P/'연구_보고서.md').write_text(f'''# BB100 GT와 카메라 보정 신뢰성 감사

2026-11-09. **실사 fine-tuning NO-GO. 현재 100장 모두 HOLD**다. 파일 형식과 대응은 확인했지만 BB 촬영의 D·newK 및 pose 정확도는 독립적으로 검증되지 않았다. HOLD는 라벨이 틀렸다고 판정한 것이 아니라 학습 조건을 아직 입증하지 못했다는 뜻이다. PASS 0, REVIEW 0, HOLD 100이다.

## 확인한 경로와 출처

- RAW 후보: `/home/oylee/projects/qwen_smt/data/BB_rgb_obj04/run_20260813_164421/rgb/`; 촬영/로봇 매칭은 같은 run의 `frame_opc_mapping.csv`.
- 수동 pose: `/home/oylee/projects/HCCEPose_experiments/real_bb100_20261108/BB_obj04/poses_obj04/pose_*.npy`.
- 라벨 작업 영상: `real_bb100_20261108/rgb_undistorted/`. 라벨 watcher의 경로 기록과 연결된다. 명칭은 실제 왜곡 보정 이력의 증명이 아니다.
- BOP 변환: `BB_obj04/build_obj04_bop.py`, `scene_gt_obj04.json`, `scene_camera_obj04.json`. 100개 pose NPY의 R,t와 BOP JSON R,t는 모두 최대 차이 1e−8 미만이다. 변환은 **동일 값의 형식 변환 검증**이지 pose 정확성 검증이 아니다.
- CAD: 기존 공식 obj04 PLY. 렌더 투영은 모델→카메라 `R X+t`, BOP K를 사용한다. BOP K는 빌더에 하드코딩돼 있고 D는 BOP JSON에 없다.

## 카메라

BB 전용 D, 기존 영상 생성 코드와 newK·보정 횟수를 입증할 자료는 발견하지 못했다. K는 개발27 K와 수치상 같았다. 개발27 K,D를 **가정한** 실험 영상과 기존 라벨 JPEG의 평균 MAE는 1회 {s['mean_mae_one_devKD']:.3f}/255, 2회 {s['mean_mae_two_devKD']:.3f}/255였다. 동일 크기(1280×720)와 1회 가설에 대한 높은 유사성은 실제 BB 카메라 D·newK 또는 이중 보정 부재의 증명이 아니다. 상세 근거·미확인 항목은 `카메라_보정_출처.md`.

## 100장 검토 자료 및 분류

기존 라벨 JPEG를 추가 보정하지 않고 BOP K/pose로 CAD 삼각형의 투영 **합집합(amodal)** 윤곽을 겹친 오버레이 100장과 10개 contact sheet를 생성했다. 모든 그림에 K·undistortion 미검증과 amodal 표기를 넣었다. 영상 가림과 무관하게 그리퍼 뒤 영역도 포함할 수 있어 visible mask가 아니다. **투영은 독립적인 GT 정확성 검증이 아니다.**

기존 CPU CAD 렌더러의 대표 10장 amodal mask와 래스터 결과를 비교했을 때 IoU는 최소 {min(x['iou_to_existing_cpu_renderer'] for x in ov):.4f}, 평균 {sum(x['iou_to_existing_cpu_renderer'] for x in ov)/len(ov):.4f}였다. 이는 두 CAD 래스터화 방식의 차이를 보는 검사이며 실사 segmentation 정확도나 pose 인증이 아니다. 첫 시도는 여러 삼각형을 한 `fillPoly` 호출로 처리해 인공 내부 경계가 생겼고, 이를 `attempt_evenodd_invalid/`에 보존한 후 개별 삼각형 합집합으로 수정했다.

프레임별 `bb100_frame_audit.csv`는 이미지·pose 경로, 해시, 보정 상태, pose 그룹, 자동 검토 우선순위, GT 시각 검토·가림·반사·흐림·경계 상태, 학습 상태를 기록한다. 043은 예비 육안상 대략적인 경계 일치와 뚜렷한 도구 가림이 보이지 않음을, 235는 그리퍼와 투영 지지영역의 중첩을 관찰했다. 이는 **보조 관찰**이며 독립적인 사람의 픽셀·pose 검수는 전부 미완료다. 나머지 가림·반사·흐림·경계는 모른다고 기록했다. bbox 내 Laplacian 분산·밝은 픽셀 비율은 비보정 진단 수치로, 흐림·반사의 정답 판정에 쓰지 않았다.

우선순위는 P1 {s['priority_counts'].get('P1',0)}장, P2 {s['priority_counts'].get('P2',0)}장이다. P1은 명시된 043→060 pose 복사, 상태가 다른 동일 pose 그룹, 또는 이전 공식 모델 진단이 후보 GT와 불일치한 경우다. 모델 불일치는 **GT 오류 증거가 아니다**. P2는 나머지 반복 pose 또는 로봇 동작 상태로 지정했으며 가림으로 확정한 것이 아니다. 두 범주 모두 camera/pose 독립 검증 전 HOLD다.

## 중복·누출 및 분할

R,t 값이 정확히 같은 pose는 {s['pose_groups']}개 그룹, 이 중 반복 {s['repeated_pose_groups']}그룹 {s['repeated_pose_frames']}장이다. 원본 BB와 개발27 사이 바이트 SHA-256 완전 일치는 {s['raw_exact_match_dev27']}건이다. 재인코딩·유사 장면·촬영 세션 중복까지 없다고 단정하지 않는다. 043→060 복사 이력은 기록으로 확인됐지만 다른 반복 pose의 원인은 미확정이다. 그룹·촬영 순서·이미지 차이는 `중복_pose_보고서.md`, `pose_groups.csv`.

동일/근접 pose와 인접 시간으로 이전 감사가 만든 보수적 연결 그룹은 36장과 64장이다. `split_draft.csv`에 그룹을 나누는 초안을 제공하지만 두 그룹 모두 같은 세션이므로 독립 검증셋이 아니며 **사용 금지 초안**이다. 실제 split은 별도 촬영 세션과 GT 검증 후 다시 정한다. `학습_검증_분할_초안.md` 참조.

## 재현과 제한

`audit.py`를 새 빈 실험 디렉터리에서 실행하고 `verify_overlay.py`, `report.py`를 순서대로 실행한다. 코드는 CPU만 사용한다. 주요 원본 이미지·pose·BOP·카메라·모델 및 로그 {s['source_file_count']}개 해시는 분석 전후 일치했다(모든 연구 파일의 전수 해시가 아님). 첫 실행은 frame mapping CSV의 UTF-8 BOM을 처리하지 못해 투영 생성 전에 중단됐고, 로그를 보존한 뒤 인코딩을 수정했다. 원본 데이터·pose·모델을 수정하지 않았다. 모델 학습은 실행하지 않았다.

다음 단계: BB 촬영 카메라의 calibration ID/K,D/newK·보정 코드와 시간 정합을 확보하고 알려진 타깃으로 검증한다. 이후 그룹별로 물체 윤곽·홀·가림·조명과 pose를 사람이 독립 검수해 PASS/REVIEW/HOLD를 다시 평가하고, 별도 세션 분할을 확정한다. 현재는 실사 fine-tuning을 시작하지 않는다.
''',encoding='utf8')
(P/'README.md').write_text('''# BB100 GT 및 카메라 감사 재현

원본 데이터 경로가 있는 기존 연구 서버와 Python/OpenCV 환경이 필요하다. 새 빈 디렉터리를 만들고 코드를 복사해 실행하면 기존 결과를 덮어쓰지 않는다. 처리된 오버레이는 카메라 보정과 pose의 정답 인증이 아니다.

```bash
set -e
TASK_SRC=/home/oylee/projects/HCCEPose_experiments/bb100_gt_camera_verification_20261109
TASK_NEW=/home/oylee/projects/HCCEPose_experiments/bb100_audit_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir "$TASK_NEW"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
TASK_PY=/home/oylee/projects/HCCEPose_experiments/yolo_hcce_rgb_pipeline/.venv/bin/python
"$TASK_PY" -u -B "$TASK_NEW/audit.py" > "$TASK_NEW/audit.log" 2>&1
"$TASK_PY" -B "$TASK_NEW/verify_overlay.py"
"$TASK_PY" -B "$TASK_NEW/report.py"
```

프레임 검수표, 오버레이, 그룹과 임시 split 및 미확인 사유는 `연구_보고서.md`를 참조한다.
''',encoding='utf8');print('REPORTS CREATED')
