# YOLO + HCCEPose RGB-only 실행

Python 3.10, CUDA GPU, 기존 연구 경로 및 `dataset_manifest.json`의 환경 의존성이 필요하다. 이미지가 RAW면 카메라 JSON의 K,D로 보정하고 output K=입력 K를 사용한다. `--dist` 없이 실행할 때는 영상이 이미 해당 K로 undistort되어 있어야 한다. 물체 CAD의 단위는 mm다. pose.json의 t도 같은 CAD 좌표 단위이며 예측 ROI에 GT는 사용하지 않는다.

```bash
cd /RESEARCH_DATA/HCCEPose_experiments/yolo_hcce_rgb_pipeline
.venv/bin/python -B run_inference.py \
  --image /RESEARCH_DATA/HCCEPose_obj04_transfer/image/000026.jpg \
  --camera /RESEARCH_DATA/HCCEPose_obj04_transfer/camera_params.json \
  --dist --out inference_example_new
```

`--out`은 기존에 없는 경로여야 한다. confidence는 기본 0.25, `--conf`로 변경 가능하다. 성공 시 `pose.json`, `overlay.png`, 실패 시 R,t=null과 사유를 저장한다. `cli_smoke/`와 `cli_miss_test/`에 실행 검증 결과가 있다.

재현은 기존 결과를 보존하도록 새 디렉터리에서 진행한다. 아래는 기존 `.venv`와 COCO 가중치, 원본 데이터 경로를 재사용한다. GPU가 사용 중이면 학습·평가를 시작하지 않는다.

```bash
TASK_SRC=/RESEARCH_DATA/HCCEPose_experiments/yolo_hcce_rgb_pipeline
TASK_NEW=/RESEARCH_DATA/HCCEPose_experiments/yolo_hcce_reproduction_$(date +%Y%m%d_%H%M%S)
mkdir -p "$TASK_NEW/weights"
cp "$TASK_SRC"/*.py "$TASK_NEW/"
cp "$TASK_SRC/weights/yolo11n.pt" "$TASK_NEW/weights/"
TASK_PY="$TASK_SRC/.venv/bin/python"
"$TASK_PY" -B "$TASK_NEW/build_dataset.py"
"$TASK_PY" -B "$TASK_NEW/train_detector.py"
"$TASK_PY" -B "$TASK_NEW/evaluate_pipeline.py"
```

평가 결과는 개발27 조건부 결과다. `연구_보고서.md`의 독립성·amodal 라벨·보정 조건을 반드시 확인한다.
