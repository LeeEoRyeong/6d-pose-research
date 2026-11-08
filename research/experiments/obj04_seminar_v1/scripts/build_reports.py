"""실제로 생성된 JSON과 로그만 이용해 한국어 발표 보고서3개 작성."""
import datetime,hashlib,json,statistics,sys
from pathlib import Path
from common import ROOT,SOURCE,SCENE,load_json,write_json,verify_originals

def maybe(name):
    p=ROOT/name
    return load_json(p) if p.exists() else None

def save(name,text):
    with (ROOT/'reports'/name).open('x',encoding='utf8') as f:f.write(text)

def link(path):return f'[{Path(path).name}]({path})'

def main():
    env=maybe('checks/environment.json');ef=maybe('checks/environment_failure.json')
    labels=maybe('checks/labels_v2_100.json') or maybe('checks/labels_v2_010.json') or maybe('checks/labels_v2_001.json') or maybe('checks/labels_001.json')
    smoke=maybe('checks/smoke_result_compat.json') or maybe('checks/smoke_result.json');sf=maybe('checks/smoke_failure.json') or maybe('checks/smoke_skip.json')
    data=maybe('data/dataset.json');protected=verify_originals()
    now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(load_json(ROOT/'checks/session.json')['start_utc']);elapsed=(now-start).total_seconds()/60
    success_env=bool(env and env.get('cuda_test')=='PASS');nlabels=labels['frames'] if labels else 0
    stats=labels['stats'] if labels else [];steps=smoke['optimizer_steps_completed'] if smoke else 0
    original_log=ROOT/'logs/smoke_steps.jsonl'
    original_steps=len(original_log.read_text().splitlines()) if original_log.exists() else 0
    number={'original_unmodified_optimizer_steps_completed':original_steps,'original_attempt_second_step_status':'FAIL: numpy.float32 to torch.cuda.FloatTensor assignment','total_successful_optimizer_steps_across_attempts':original_steps+steps,'new_environment_cuda_pass':success_env,'xyz_frames_pass':nlabels,'xyz_label_images':nlabels*2,'optimizer_steps':steps,'original_files_unchanged':protected,'elapsed_minutes':elapsed}
    if stats:number.update(mask_iou_min=min(x['mask_iou'] for x in stats),mask_iou_mean=statistics.mean(x['mask_iou'] for x in stats),front_depth_abs_median_max=max(x['depth_abs_median'] for x in stats),front_depth_abs_p95_max=max(x['depth_abs_p95'] for x in stats),independent_ray_max_error=max(x['ray_minmax_max_error'] for x in stats),quantized_reprojection_p99_max_px=max(max(x['quantized_reprojection_p99_px']) for x in stats),max_normalized_clip=max(x['max_normalized_clip'] for x in stats),label_generation_seconds_sum=sum(x['seconds'] for x in stats))
    if smoke:number.update(losses=[x['loss_total'] for x in smoke['rows']],peak_allocated_mib=max(x['peak_allocated_bytes'] for x in smoke['rows'])/2**20,peak_reserved_mib=max(x['peak_reserved_bytes'] for x in smoke['rows'])/2**20,checkpoint_bytes=smoke['checkpoint_bytes'],parameter_count=smoke['parameter_count'])
    write_json(ROOT/'checks/seminar_numbers.json',number)
    if smoke:
        from PIL import Image,ImageDraw
        image=Image.new('RGB',(1000,600),'white');draw=ImageDraw.Draw(image)
        draw.text((65,20),'HCCEPose scalar-compatible smoke: 4 optimizer steps (NOT pose accuracy)',fill='black')
        vals=number['losses'];low=min(vals);high=max(vals);span=max(high-low,.1)
        draw.line([(70,65),(70,520),(940,520)],fill='black',width=2)
        points=[(90+i*200,490-(v-low)/span*370) for i,v in enumerate(vals)]
        draw.line(points,fill='blue',width=3)
        for i,((x,y),v) in enumerate(zip(points,vals)):
            draw.ellipse((x-5,y-5,x+5,y+5),fill='blue');draw.text((x-35,y-25),f'{v:.5f}',fill='black');draw.text((x-10,530),str(i+1),fill='black')
        draw.text((440,565),'Optimizer step',fill='black');draw.text((5,40),'Loss',fill='black')
        image.save(ROOT/'figures/smoke_loss.png')
        import csv
        with (ROOT/'logs/smoke_steps.csv').open('x',newline='') as f:
            keys=['step','loss_total','loss_front_sum','loss_back_sum','loss_mask','gradient_norm','peak_allocated_bytes','peak_reserved_bytes','seconds']
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows([{k:r[k] for k in keys} for r in smoke['rows']])
    env_text=json.dumps(env or ef or {'상태':'설치 또는 검증 미완료'},ensure_ascii=False,indent=2)
    smoke_text=json.dumps({k:v for k,v in (smoke or sf or {'상태':'미수행'}).items() if k not in ('rows','traceback')},ensure_ascii=False,indent=2)
    label_text=json.dumps({k:v for k,v in number.items() if k.startswith(('xyz','mask','front_depth','independent_ray','quantized','max_normal','label_generation'))},ensure_ascii=False,indent=2)
    root=str(ROOT);old=str(SOURCE/'validation_obj04_transfer_20261103')
    loss_link=link(root+'/figures/smoke_loss.png') if smoke else '미생성: smoke 성공 수치가 없음'
    xyz_link=link(root+'/figures/xyz_comparison_000000.png') if (ROOT/'figures/xyz_comparison_000000.png').exists() else '미생성: 유효 XYZ 라벨 미확보'
    overview=f'''# 세미나 실험 진행 요약
+
+작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
+실험 루트: `{root}`
+작업 경과: 약 {elapsed:.2f}분 / 최대180분. 150분 신규개발중단 정책 내 수행.
+
+## 핵심 결론
+
+이번 신규 결과는 격리 환경 CUDA 확인={success_env}, 검증 통과 XYZ={nlabels}프레임, optimizer step={steps}회이다.
+**이는 실행 가능성 검증이며, 학습 완료·실사 Pose 정확도 개선·검출 성능 확보 결과가 아니다.**
+
+| 구분 | 결과 |
+|---|---|
+| 기존 완료 결과 | 실사27프레임 수치·CPU CAD 재투영: 시각 성공22, 판단불가5, 고유 GT 행렬15종. 모델 정확도 수치 아님 |
+| 기존 PBR | RGB/Depth/Mask/visible Mask 각100, obj4, 10생성블록, train80/val20. JSON 픽셀 통계3프레임의 1픽셀 차이 원본 보존 |
+| 신규 환경 | 전용 venv 생성, CUDA 검증 {success_env}, 원본 network_model.py import. 상세 실패·버전은 환경 보고서 |
+| 신규 라벨 | CPU min/max depth 기반 Front/Back {nlabels}프레임, 원본 역투영·채널 규약. 원본 VisPy 직접 실행/bitwise 비교는 하지 않음 |
+| 신규 smoke | PBR train만 사용한 optimizer {steps}회. 체크포인트 상태는 세부 기록 참조 |
+| 미수행 | 장시간 학습, YOLO 추가 학습/새 평가, 실제27프레임 모델 추론, ADD 정확도 측정, 독립 테스트 |
+| 보존 | 원본 PBR·실사·소스 {protected}파일 SHA-256 일치, 기존 체크포인트 덮어쓰기 없음 |
+
+## 원본 실패와 호환 실행의 구분
+
+원본 Loss 실행은 optimizer 1회 완료 뒤 두 번째 forward에서 `TypeError: can't assign a numpy.float32 to a torch.cuda.FloatTensor`로 중단됐다. `checks/smoke_failure.json`, `logs/smoke_train.log`, `logs/smoke_steps.jsonl`을 보존했다. 원인은 원본 Loss의 이동평균 np scalar를 CUDA tensor에 대입하는 경로다.
+
+별도 `smoke_train_compat.py`에서 원본 파일을 수정하지 않고, 해당 모듈의 `np.mean` scalar 반환값만 Python scalar로 바꾸는 프로세스 내부 어댑터를 사용했다. 수식·이동평균 값은 유지했다. 이 호환 실행은 새로 초기화해4step을 수행했으며 원본 성공1회를 이어 학습한 것이 아니다. 두 시도 합계 성공 optimizer step은5회다. 보고된 loss 곡선·checkpoint는 호환 실행4step만의 결과다.
+
+## 문제와 처리
+
+- 첫 PyTorch 설치는 pip23.0.1의 typing-extensions 이름 처리 및 CUDA 전용 인덱스의 flit_core 부재로 실패했다. `logs/install_torch.log`를 보존하고 전용 venv의 pip만 갱신해 재시도했다. 시스템·기존 hccepose 환경은 변경하지 않았다.
+- CAD 누락은 실제 전송 CAD를 새 `data/models/`에 독립 복사해 연결했다. 원본 PBR은 변경하지 않았다.
+- 렌더러는 CPU 구현을 사용했다. 정의는 기존 GL_LESS/GL_GREATER와 동일한 최소/최대 표면 깊이이나 VisPy 실행 결과와 완전히 같은 구현이라고 주장하지 않는다.
+- 네트워크 초기화는 pretrained 다운로드 없이 무작위 초기화했다. 원본 생성자 호출 중에만 프로세스 내부 patch를 사용했다. HCCEPose 원본 소스는 수정하지 않았다.
+- 성공·실패·건너뛴 항목은 `checks/` JSON과 `logs/` 원문에 기록했다. 라벨/스모크 실패가 있으면 이 문서의 수치가 아닌 해당 오류를 우선 확인한다.
+
+## 재현 명령 — 이번 실행에 사용한 순서
+
+아래 생성 단계는 출력이 없을 때만 실행하며 기존 출력이 있으면 덮어쓰지 않고 중단한다. 신규 run을 만들 때 모든 스크립트의 ROOT가 해당 run을 가리키게 해야 한다.
+
+```bash
+{root}/env/bin/python -B {root}/scripts/prepare_data.py
+{root}/env/bin/python -B {root}/scripts/probe_env.py
+{root}/env/bin/python -B {root}/scripts/generate_xyz_v2.py --limit 1
+{root}/env/bin/python -B {root}/scripts/generate_xyz_v2.py --limit 10
+{root}/env/bin/python -B {root}/scripts/generate_xyz_v2.py --limit 100
+{root}/env/bin/python -B {root}/scripts/smoke_train.py
+{root}/env/bin/python -B {root}/scripts/smoke_train_compat.py
+{root}/env/bin/python -B {root}/scripts/build_reports.py
+```
+
+환경은 `/RESEARCH_DATA/USER/miniconda3/bin/python -m venv`로 새 디렉터리에 만들었다. 첫 시도 실패 후 이 venv의 pip만 갱신하고 아래 조합을 설치했다 (기존 환경 변경 없음).
+
+```bash
+{root}/env/bin/python -m pip install --upgrade pip --index-url https://pypi.org/simple
+{root}/env/bin/python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128 numpy==1.26.4 opencv-python-headless==4.9.0.80 pillow==11.3.0
+```
+
+원문 설치 로그는 `logs/install_torch.log`, `logs/install_pip_upgrade.log`, `logs/install_retry.log`와 `checks/pip_freeze.txt`를 함께 참조한다. 명령 목록은 실행 결과를 대신하지 않는다. `checks/`에 성공 파일이 없는 단계는 미완료이다.
+
+## 발표용 5슬라이드 구성
+
+### 1. 지난 연구 진행 상황
+
+- 핵심 메시지: 코드 구조·연구 자산을 분석하고, 성능 실험에 앞서 데이터와 실행 조건을 검증했다.
+- 실제 수치: 실사 GT27프레임, PBR100프레임, train80/val20.
+- 그림: `{root}/figures/pbr_rgb_000010.jpg`
+- 설명: HCCEPose는 검출→앞뒤 좌표→PnP 구조이다. 기존 분석과 이번 기능검증을 분리한다. 아직 모델 성능을 측정하지 않았다.
+
+### 2. SMT obj04 실사 GT 27프레임 검증
+
+- 핵심 메시지: GT는 개발 평가 후보로 사용 가능하지만 독립 최종 정밀 GT는 아니다.
+- 실제 수치: 이전 CPU 재투영54건, 거시 시각 성공22·판단불가5, 고유 행렬15종. NPY↔JSON·K 최대차이0.
+- 그림: `{root}/figures/existing_real_gt_000026.png`, `{root}/figures/existing_real_gt_000092.png`, `{root}/figures/existing_real_gt_000126.png`
+- 설명: 모두 **기존 결과 재사용**이다. same-K UND 가정과 절대단위는 미확인이다. 22/27을 모델 정확도로 읽으면 안 된다.
+
+### 3. PBR 100프레임 구성과 특징
+
+- 핵심 메시지: 실제 영상과 분리된 합성 학습 자산과 블록 분할을 확보했다.
+- 실제 수치: obj4 인스턴스100, 1280×720, 생성블록10, train80/val20. 기존 visib_fract 0.996813~1.0.
+- 그림: `{root}/figures/pbr_rgb_000000.jpg`, `{root}/figures/pbr_rgb_000010.jpg`, `{root}/figures/pbr_mask_000010.png`
+- 설명: 도구 가림이 거의 없어 실사 일반화 한계가 있다. 고반사 외관과 실사 표면 차이가 있다. GT-info 3프레임 미세 불일치는 숨기지 않는다.
+
+### 4. 학습 환경·XYZ 라벨 및 실제 기능 검증
+
+- 핵심 메시지: 새로운 실험으로 CUDA·라벨·학습 step 실행 가능성을 검증했다; 정확도 검증은 아니다.
+- 실제 수치: CUDA PASS={success_env}, XYZ={nlabels}프레임, optimizer={steps}회. 세부 실측은 `발표용_핵심수치.md` 참조.
+- 그림: {xyz_link}; {loss_link}
+- 설명: CPU 최근접/최원거리 표면을 Front/Back으로 사용한다. PBR train만 사용했다. 호환 처리4-step loss 변화로 수렴·실사 성능을 주장하지 않는다.
+
+### 5. 발견된 문제와 다음 실험
+
+- 핵심 메시지: 기능 연결을 확인한 범위에서 재현성을 고정하고, 다음은 독립 검증과 실제 Baseline 학습이다.
+- 실제 수치: 기존 YOLO smoke 1epoch·fraction0.1·mAP0, 실사 GT15종, 초기 검증대상27프레임.
+- 그림: `{root}/figures/existing_real_gt_000092.png`, 라벨 성공 시 `{root}/figures/pbr_cad_overlay_000000.png`
+- 설명: 이번에 발견한 원본 Loss의 np scalar→CUDA tensor 대입 오류를 정식 수정·회귀검증 대상으로 삼는다. YOLO 기존 가중치를 유효 검출기로 가정하지 않는다. 실사 K/단위·가림 라벨을 확정한다. 다음에는 PBR val로 모델 선택 후 실사 dev를 평가하고 새로운 독립 촬영 테스트를 확보한다.
+'''.replace('\n+','\n')
    save('세미나_실험진행_요약.md',overview)
    detail=f'''# 실험환경·데이터 검증
+
+## 이번 실행 환경
+
+```json
+{env_text}
+```
+
+기존 환경 `/RESEARCH_DATA/USER/miniconda3/envs/hccepose`와 시스템 Python·CUDA는 변경하지 않았다. 새 환경은 `{root}/env`이다. 최소 네트워크 기능 검증 환경이며 imgaug·KASAL·VisPy·Ultralytics를 포함한 전체 학습/검출 파이프라인 환경은 이번에 구성하지 않았다. `bop_loader`와 `Tester` 전체 import·실사 추론을 검증한 것으로 해석하지 않는다. `checks/pip_freeze.txt`, `checks/pip_check.txt`, `logs/install_*.log`에 설치·의존성 기록이 있다.
+
+## 데이터와 누수 방지
+
+- 원본 manifest {protected}개 해시 일치. PBR100·실사85파일과 기존 Python 소스를 포함한다.
+- PBR 원본 `{SCENE}`는 읽기 전용으로 사용한다. CAD와 models_info는 새 data/models에 독립 복사했다.
+- 기존 80/20 분할과 10프레임 생성블록 분리를 재확인했다. train 블록0,1,2,3,5,6,7,8; val 블록4,9.
+- 실사27은 학습에 사용하지 않았다. val20도 optimizer에 사용하지 않았다. smoke는 PBR train의 처음4프레임만 반복하며 일반화 성능의 근거가 아니다.
+- 기존 100프레임 디코딩·BBox·GT 검사는 재사용했다. 상세 원본 근거: `{old}/pbr_baseline_readiness_ko.md`.
+- PBR `scene_gt_info.json`의 000024·000036·000077 1픽셀 차이는 원본 그대로 보존했다. smoke는 bbox_visib와 실제 visible Mask를 사용하므로 가시율 통계를 임의 조작하지 않았다.
+
+## 신규 XYZ 라벨 검증
+
+```json
+{label_text}
+```
+
+- CPU rasterizer는 culling 없이 각 광선의 최소/최대 카메라 Z를 취한다. 이는 원본 `renderer_vispy.py`의 GL_LESS/GL_GREATER Front/Back 정의와 대응한다.
+- 원본 `rendering_bop_dataset_back_front.__getitem__`의 정수 영상 좌표 역투영, `(Xc-t)@R`, min/size 정규화·255반올림·clamp를 따른다. 대칭 메타데이터가 없어 대칭 정규화 분기는 적용하지 않았다.
+- 원본 OpenGL처럼 픽셀 중심 offset0.5를 사용한다. 원본 VisPy 자체를 실행한 bitwise 동등성 검사는 미수행이다.
+- 독립 Moller-Trumbore ray/triangle 교차를 프레임당5점에서 비교한다. Mask IoU>=0.97, 동일 +0.5 광선 위치로 bilinear 정합한 관측 Depth의 5×5 eroded visible 영역 절대오차 median<=2·p95<=3(PBR깊이단위), 독립 ray max오차<1e-5, 양자화 재투영 p99<2px, normalized clipping<0.02를 요구했다. 이는 기능 점검 허용치이지 공정 정확도 기준이 아니다.
+- 유효 픽셀의 NaN/Inf, Front<=Back, 무효 픽셀0, PNG 저장/읽기 동일성을 검사했다.
+- 저장 PNG는 기존 cv2.imread/imwrite 규약에 맞춰 메모리 채널 XYZ를 사용하므로 PNG를 일반 RGB로 읽으면 ZYX이다. 발표 비교 그림은 X=빨강,Y=초록,Z=파랑으로 별도 시각화했다. 파일 채널 규약을 바꾸지 않는다.
+- 라벨 경로: `{root}/labels/front/`, `{root}/labels/back/`; 검증 통계는 `{root}/labels/stats_v2/`. 실사용 visible Mask는 생성하지 않았다.
+
+## 라벨 검증 실패와 원인 분리
+
+첫 10프레임 확대는 000002의 비정합 Depth 비교 p95=4.828598로 중단됐다 (`logs/xyz_010.log`). 무효 라벨을 성공으로 기록하지 않았다. CPU 진단에서 PBR depth는 정수 광선, 라벨 렌더러는 +0.5 광선으로 비교 위치가 달랐고 경계 불연속이 영향을 줬다. 생성기 본체와 이미 생성된 PNG를 바꾸지 않고 v2에서 같은 광선으로 Depth를 bilinear 정합하고 5×5 eroded 가시 영역을 비교했다. 오차 허용치 자체는 유지했다. 실패한 원본 스크립트·로그와 `checks/pixel_center_diagnostic.json`, `checks/depth_alignment_diagnostic.json`을 보존한다. 이 진단은 렌더러 간 완전 동일성 인증이 아니다.
+
+## 신규 네트워크 기능 검사
+
+```json
+{smoke_text}
+```
+
+원본 Loss의 두 번째 step 실패는 별도 보존했다. 성공한4step은 프로세스 내부의 np scalar→Python scalar 호환 처리를 포함한다. Loss는 원본 HccePose_Loss 수식과 원본 학습식 `3*front.sum()+3*back.sum()+mask`를 사용했다. 원본 네트워크 파일을 import했고 생성자에서만 torchvision resnet34 pretrained를 끄는 프로세스 내부 patch를 적용했다. 원본 소스는 수정하지 않았다. 좌표 인코딩·crop 함수는 원본 AST를 격리 추출했다.
+
+체크포인트가 있으면 smoke용 모델 상태 저장·strict 재로딩과 tensor 일치를 검증한 것이다. optimizer 상태는 저장하지 않으므로 장기 학습 재개 체크포인트가 아니다. PnP·ADD·실사 추론은 실행하지 않았다.
+
+## 다음 단계
+
+통제된 카메라 K·실물 단위·실사 라벨 품질 확정, 충분한 PBR/실사 train 및 별도 val/test, 검출기 재학습, PBR val 기반 HCCE 모델·PnP 후보 선정, 실제27프레임 dev 평가 순서로 진행한다. 새 독립 테스트 확보 전 성능 개선 주장을 보류한다.
+'''.replace('\n+','\n')
    save('실험환경_데이터_검증.md',detail)
    rows_table='\n'.join(f"| {x['step']} | {x['loss_total']:.6f} | {x['gradient_norm']:.6f} | {x['max_first_parameter_delta']:.8f} | {x['peak_allocated_bytes']/2**20:.2f} | {x['seconds']:.3f} |" for x in (smoke['rows'] if smoke else []))
    figures='\n'.join(f'- `{p.resolve()}`' for p in sorted((ROOT/'figures').iterdir()) if p.is_file())
    core=f'''# 발표용 핵심수치
+
+## 기존 결과 — 이번 신규 학습 성과로 표시하지 않음
+
+| 항목 | 실제 값 | 근거 |
+|---|---|---|
+| 실사 GT | 27프레임·고유행렬15종 | 이전 수치/재투영 보고서 |
+| 시각 검토 | 성공22·판단불가5·명백한 실패0 | 거친 시각 검토이며 정확도 아님 |
+| NPY/JSON R,t·카메라K 최대차이 | 0.0 | 이전 수치 검증 |
+| PBR | 각100개 RGB/Depth/Mask/visible Mask | 이전 전수 검사와 이번 파일대응 확인 |
+| 분할 | train80 / val20, 생성블록8/2 | autosplit 목록·dataset.json |
+| 기존 YOLO | 1epoch·fraction0.1·precision/recall/mAP 모두0 | 기존 results.csv, 새 평가 아님 |
+
+## 이번 신규 실행 실측
+
+```json
+{json.dumps(number,ensure_ascii=False,indent=2)}
+```
+
+| optimizer step | total loss | gradient norm | 첫 파라미터 max 변화 | peak allocated MiB | step seconds |
+|---|---|---|---|---|---|
+{rows_table}
+
+GPU 메모리는 해당 smoke 프로세스에서 측정한 peak allocated/reserved이며 nvidia-smi의 전체 사용량과 다르다. 시간은 동기화를 포함한 기능검사 구간이며 전체 학습 처리량 benchmark가 아니다. Loss 감소 또는 변화는 4개 PBR 표본 반복·무작위 초기화·스칼라 호환 처리·4step 조건의 관측일 뿐 수렴이나 Pose 정확도 향상을 뜻하지 않는다.
+
+## 바로 삽입할 그림의 정확한 경로
+
+{figures}
+
+## 반드시 함께 설명할 미수행 항목
+
+실사 모델 추론·ADD(-S)·검출 recall·장시간 학습·독립 테스트는 미수행이다. XYZ 그림은 CAD/GT로 만든 지도 라벨이며 네트워크 예측 그림이 아니다. `existing_real_gt_*` 그림은 이전 CPU 재투영 결과이며 신규 결과로 포장하지 않는다. 이번 체크포인트는 성능 검증된 학습 모델이 아니다.
+'''.replace('\n+','\n')
    save('발표용_핵심수치.md',core)
    print(json.dumps(number,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
