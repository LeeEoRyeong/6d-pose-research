#!/usr/bin/env python3
"""명시적 --run 승인 후 사용하는 CPU 전용 합성 검증. 자동 설치 없음."""
import argparse
import datetime
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import types
from unittest.mock import patch

# 이 프로세스와 자식에만 적용. GPU 장치 탐색 API도 호출하지 않는다.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
from source_loader import ROOT, EVIDENCE, function, prediction_geometry, bop_metrics

RESULTS = []
DEPENDENCIES = {}


def require(*names):
    for name in names:
        if name not in DEPENDENCIES:
            try:
                module = importlib.import_module(name)
                if name == "torch":
                    module.set_num_threads(1)
                if name == "cv2":
                    module.setNumThreads(1)
                    module.ocl.setUseOpenCL(False)
                DEPENDENCIES[name] = module
            except Exception as exc:
                raise ImportError(f"필요 패키지 {name} 사용 불가; 자동 설치하지 않음: {exc}") from exc
    return [DEPENDENCIES[name] for name in names]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(np, actual, expected, atol=1e-6):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def record(stage, name, expected, callback):
    try:
        details = callback() or {}
        status, error, trace = "PASS", None, None
    except ImportError as exc:
        status, details, error, trace = "SKIP", {}, str(exc), None
    except Exception as exc:
        status, details, error = "FAIL", {}, f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
    RESULTS.append(dict(stage=stage, name=name, expected=expected, status=status,
                        details=details, error=error, traceback=trace))
    print(f"[{status}] {stage}: {name}" + (f" — {error}" if error else ""), flush=True)


def hcce():
    np, torch = require("numpy", "torch")
    enc = function("HccePose/bop_loader.py", "train_bop_dataset_back_front.hcce_encode", {"np": np})
    enc_test = function("HccePose/bop_loader.py", "test_bop_dataset_back_front.hcce_encode", {"np": np})
    dec = function("HccePose/network_model.py", "HccePose_BF_Net.hcce_decode", {"torch": torch})
    values = np.arange(256, dtype=np.float64).reshape(16, 16)
    image = np.stack([values, np.flip(values, 0), np.flip(values, 1)], axis=-1)
    codes = enc(None, image)
    check(codes.shape == (16, 16, 24), "XYZ 채널 구성 불일치")
    check(np.isfinite(codes).all() and codes.min() >= 0 and codes.max() <= 1,
          "연속 코드 범위 오류")
    close(np, enc_test(None, image), codes, 0)
    bits = torch.tensor((codes > 0.5)[None], dtype=torch.float32, device="cpu")
    decoded = dec(types.SimpleNamespace(powers=None), bits).numpy()[0]
    close(np, decoded, image, 0)
    # 독립 표준 binary-reflected Gray code와 임계값 코드 비교.
    quantized = image.astype(np.int64)
    gray = quantized ^ (quantized >> 1)
    oracle = np.concatenate([((gray[..., a, None] >> np.arange(7, -1, -1)) & 1)
                             for a in range(3)], axis=-1)
    close(np, bits.numpy()[0], oracle, 0)
    minimum = np.array([-10., -20., -2.])
    size = np.array([20., 40., 4.])
    xyz = decoded / 255 * size + minimum
    close(np, xyz.min(axis=(0, 1)), minimum)
    close(np, xyz.max(axis=(0, 1)), minimum + size)
    return {"values_per_axis": 256, "max_integer_error": float(abs(decoded-image).max()),
            "scope": "정수 표적/임계값/복호화 및 독립 물체 좌표 계산; 네트워크 실행 없음"}


def hcce_dtype():
    np, = require("numpy")
    enc = function("HccePose/bop_loader.py", "train_bop_dataset_back_front.hcce_encode", {"np": np})
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = np.stack([image]*3, axis=-1)
    close(np, enc(None, image), enc(None, image.astype(np.float64)), 0)
    return {"dtype": "uint8 (cv2.imread와 동일)", "note": "NumPy 버전별 modulo 동작도 검증"}


def camera_geometry(height, width):
    np, cv2 = require("numpy", "cv2")
    K = np.array([[830., 0, width/2+3], [0, 810., height/2-2], [0, 0, 1]])
    original = K.copy()
    img, Knew, ratio, dx, dy = prediction_geometry(np, cv2)(
        np.zeros((height, width, 3), dtype=np.uint8), K)
    scale = 1/ratio if ratio > 1 else 1.
    T = np.array([[scale, 0, dx], [0, scale, dy], [0, 0, 1.]])
    close(np, Knew, T @ original, 1e-9)
    close(np, K, original, 0)
    check(img.shape[0] % 32 == 0 and img.shape[1] % 32 == 0,
          "출력 영상이 32 배수가 아님")
    X = np.array([[10., 20., 400.], [-20., 10., 550.]])
    def project(k):
        q = X @ k.T
        return q[:, :2] / q[:, 2:]
    uv = project(original)
    close(np, project(Knew), uv*scale+[dx, dy], 1e-8)
    # 정수 resize 크기와 K의 명목 배율 차이는 PASS와 별도로 수치 보고.
    realized_x = int(width/ratio)/width if ratio > 1 else 1.
    realized_y = int(height/ratio)/height if ratio > 1 else 1.
    residual = np.max(abs(uv * [realized_x-scale, realized_y-scale]))
    return {"input_hw": [height, width], "output_hw": list(img.shape[:2]),
            "nominal_scale": scale, "realized_scale_xy": [realized_x, realized_y],
            "nominal_vs_integer_resize_max_px": float(residual),
            "note": "후자 수치는 정수 크기 반올림 영향 진단이며 보간 pixel-center 규약과 별개"}


def crop_geometry():
    np, cv2, torch = require("numpy", "cv2", "torch")
    pad = function("HccePose/bop_loader.py", "pad_square_fp32", {"np": np})
    crop = function("HccePose/bop_loader.py", "crop_square_resize", {"np": np, "cv2": cv2})
    batch = function("HccePose/tester.py", "crop_trans_batch_hccepose", {"torch": torch})
    errors = []
    for box in ([40., 50., 80., 40.], [-8., 5., 25., 50.], [1.3, 2.7, 33.2, 19.1]):
        b = np.array(box)
        expected = pad(b, 1.5)
        T, rounded = batch(torch.tensor([box], dtype=torch.float32), [128, 128], 1.5)
        rb = rounded.numpy()[0]
        left, top = np.round(expected[:2])
        right, bottom = np.round(expected[:2]+expected[2:])
        close(np, rb, [left, top, right-left, bottom-top], 1e-5)
        oracle = np.array([[128/rb[2], 0, -rb[0]*128/rb[2]],
                           [0, 128/rb[3], -rb[1]*128/rb[3]], [0, 0, 1.]])
        close(np, T.numpy()[0], oracle, 2e-5)
        # 출력 좌표 i/128을 원본으로 복원하는 inference_batch 규약과 비교.
        q = np.array([[0., 0., 1.], [63., 81., 1.], [127., 127., 1.]])
        restored = q @ np.linalg.inv(T.numpy()[0]).T
        close(np, restored[:, :2], q[:, :2]/128*rb[2:]+rb[:2], 3e-5)
        K = np.array([[500., 0, 160.], [0, 510., 120.], [0, 0, 1.]])
        X = np.array([[0.1, 0.2, 2.], [0.3, -0.1, 3.]])
        p = X @ K.T; p /= p[:, 2:]
        lhs = X @ (oracle @ K).T; lhs /= lhs[:, 2:]
        close(np, lhs, p @ oracle.T, 1e-8)
        errors.append(float(abs(rb-expected).max()))
    # 실제 원본 cv2 crop 함수의 샘플링을 좌표 ramp로 검증.
    y, x = np.mgrid[:240, :320].astype(np.float32)
    ramp = np.stack([x, y], axis=-1)
    box = np.array([40., 60., 128., 128.])
    out = crop(ramp, box, 128, cv2.INTER_LINEAR)
    close(np, out[20:100, 20:100], ramp[80:160, 60:140], 1e-5)
    return {"max_float_vs_rounded_bbox_delta": max(errors),
            "note": "학습 crop은 float, 추론 crop은 경계 반올림. 실제 Kornia warp는 별도 검사"}


def kornia_sampling():
    np, cv2, torch, kornia = require("numpy", "cv2", "torch", "kornia")
    batch = function("HccePose/tester.py", "crop_trans_batch_hccepose", {"torch": torch})
    y, x = np.mgrid[:240, :320].astype(np.float32)
    ramp = torch.tensor(np.stack([x, y])[None], device="cpu")
    T, boxes = batch(torch.tensor([[40., 60., 128., 128.]]), [128, 128], 1.)
    warped = kornia.geometry.transform.warp_perspective(
        ramp, T, dsize=(128, 128), mode="bilinear", align_corners=False)
    yy, xx = np.mgrid[:128, :128]
    expected = np.stack([xx+40., yy+60.])
    delta = abs(warped.numpy()[0, :, 8:-8, 8:-8] - expected[:, 8:-8, 8:-8])
    maximum = float(delta.max())
    check(maximum <= 0.05, f"Kornia 실제 샘플링과 inverse homography가 불일치: 최대 {maximum:.6f}px (허용 0.05px)")
    return {"max_sampling_error_px": maximum}


def pnp_functions():
    np, cv2 = require("numpy", "cv2")
    env = {"np": np, "cv2": cv2, "itertools": importlib.import_module("itertools")}
    solve = function("HccePose/PnP_solver.py", "solve_PnP", env)
    env["solve_PnP"] = solve
    comb = function("HccePose/PnP_solver.py", "solve_PnP_comb", env)
    return np, cv2, solve, comb


def fixture(np, cv2, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([-25, -15, -8], [25, 15, 8], (12, 16, 3)).astype(np.float32)
    rvec = np.array([0.2, -0.15, 0.1], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[12.], [-9.], [400.]])
    K = np.array([[800., 0, 320.], [0, 810., 240.], [0, 0, 1.]])
    # 독립 pinhole 투영으로 OpenCV 생성/풀이의 자기 검증을 피함.
    camera = xyz.reshape(-1, 3) @ R.T + t.T
    uv = camera @ K.T; uv = (uv[:, :2]/uv[:, 2:]).reshape(12, 16, 2)
    return (np.ones((12, 16), np.uint8), xyz, uv, K), R, t


def pose_errors(np, result, R, t):
    check(bool(result["success"]), "정상 합성 대응점에서 PnP 실패")
    check(np.isfinite(result["rot"]).all() and np.isfinite(result["tvecs"]).all(), "비유한 Pose 반환")
    angle = float(np.degrees(np.arccos(np.clip((np.trace(result["rot"] @ R.T)-1)/2, -1, 1))))
    distance = float(np.linalg.norm(result["tvecs"]-t))
    return angle, distance


def pnp_accuracy(mode, noisy=False):
    np, cv2, solve, _ = pnp_functions()
    worst_r = worst_t = 0.
    for seed in (0, 7, 31):
        data, R, t = fixture(np, cv2, seed)
        if noisy:
            mask, xyz, uv, K = data
            rng = np.random.default_rng(seed+100)
            uv = uv + rng.normal(0, 0.25, uv.shape)
            outliers = rng.choice(uv.shape[0]*uv.shape[1], 38, replace=False)
            uv.reshape(-1, 2)[outliers] += rng.uniform(40, 100, (38, 2))
            data = mask, xyz, uv, K
        cv2.setRNGSeed(seed)
        result = solve(data, pnp_op=mode, iterationsCount=1000)
        r, t_error = pose_errors(np, result, R, t)
        limit_r, limit_t = (1., 2.) if noisy else (0.01, 0.01)
        check(r <= limit_r and t_error <= limit_t,
              f"seed={seed}: 회전={r:.6f}deg, 이동={t_error:.6f}mm; 허용 {limit_r}deg/{limit_t}mm")
        worst_r, worst_t = max(worst_r, r), max(worst_t, t_error)
    return {"max_rotation_deg": worst_r, "max_translation_mm": worst_t,
            "seeds": [0, 7, 31], "noise_px": 0.25 if noisy else 0,
            "outlier_count": 38 if noisy else 0}


def pnp_sparse(count):
    np, cv2, solve, _ = pnp_functions()
    data, _, _ = fixture(np, cv2)
    mask = np.zeros_like(data[0]); mask.reshape(-1)[:count] = 1
    for mode in (0, 1, 2):
        result = solve((mask, *data[1:]), pnp_op=mode)
        check(not result["success"], "4점 이하에서 명시적 실패 필요")
        check(np.isfinite(result["rot"]).all() and np.isfinite(result["tvecs"]).all(), "실패 기본 Pose가 비유한 값")


def pnp_forced_failure(mode):
    np, cv2, solve, _ = pnp_functions()
    data, _, _ = fixture(np, cv2)
    # 실제 OpenCV 정상 계산과 별개로 solver의 False 반환 계약을 강제한다.
    with patch.object(cv2, "solvePnPRansac", return_value=(False, None, None, None)):
        result = solve(data, pnp_op=mode)
    check(not result["success"], "OpenCV 실패가 성공으로 처리됨")
    check(np.isfinite(result["rot"]).all() and np.isfinite(result["tvecs"]).all(), "실패 시 비유한 값")


def pnp_combined():
    np, cv2, _, comb = pnp_functions()
    data, R, t = fixture(np, cv2)
    mask, xyz, uv, K = data
    # 시선상 동일 투영을 갖는 실제 앞/뒤 합성 표면 쌍 생성.
    camera = xyz.reshape(-1, 3) @ R.T + t.T
    rear_camera = camera * 1.02
    rear = ((rear_camera-t.T) @ R).reshape(xyz.shape).astype(np.float32)
    prior = np.zeros(15); prior[0] = 1
    cv2.setRNGSeed(0)
    result = comb((mask, xyz, rear, uv, K), keypoints_=prior)
    r, te = pose_errors(np, result, R, t)
    check(r < .05 and te < .1, f"조합 선택 오차 초과: {r}deg/{te}mm")
    return {"rotation_deg": r, "translation_mm": te,
            "scope": "독립 광선상 앞/뒤 합성점과 15개 후보군 prior; 학습된 prior 성능 검증 아님"}


def metric_setup():
    np, spatial = require("numpy", "scipy.spatial")
    bop = bop_metrics(np, spatial)
    wrapper = function("HccePose/metric.py", "add_s", {"np": np, "pose_error": bop})
    return np, bop, wrapper


def metric_oracle():
    np, bop, _ = metric_setup()
    rng = np.random.default_rng(19)
    pts = rng.uniform(-10, 10, (31, 3))
    I = np.eye(3); z = np.zeros((3, 1))
    R = np.diag([-1., -1., 1.]); t = np.array([[3.], [4.], [0.]])
    for rot, trans in ((I, z), (I, t), (R, t)):
        a = pts @ rot.T + trans.T
        expected_add = np.linalg.norm(a-pts, axis=1).mean()
        expected_adi = np.linalg.norm(pts[:, None, :]-a[None, :, :], axis=2).min(axis=1).mean()
        close(np, bop.add(rot, trans, I, z, pts), expected_add, 1e-10)
        close(np, bop.adi(rot, trans, I, z, pts), expected_adi, 1e-10)
    close(np, bop.add(I, t, I, z, pts), 5., 1e-10)
    square = np.array([[-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.], [1., 1., 0.]])
    close(np, bop.adi(R, z, I, z, square), 0., 0)
    check(bop.add(R, z, I, z, square) > 0, "ADD가 대칭 점 순열을 구분하지 못함")
    return {"translation_3_4_0_ADD": 5., "symmetric_rotation_ADDS": 0.}


def metric_wrapper():
    np, bop, wrapper = metric_setup()
    I = np.eye(3); z = np.zeros((3, 1))
    pts = np.array([[-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.], [1., 1., 0.]])
    diameter = float(np.sqrt(8.))
    threshold = .1 * diameter
    distances = [0, threshold-1e-7, threshold, threshold+1e-7]
    gt = [[I, z] for _ in distances]
    pred = [[I, np.array([[d], [0.], [0.]])] for d in distances]
    _, passes, errors = wrapper({"pts": pts}, {"diameter": diameter}, gt, pred)
    close(np, errors, distances, 1e-12)
    # 실제 경계는 부동소수점 덧셈 영향이 있으므로 원점 점 하나로 엄밀 분리.
    _, exact, _ = wrapper({"pts": np.zeros((1, 3))}, {"diameter": 10.},
                          [[I, z]]*3, [[I, np.array([[d], [0.], [0.]])] for d in (.999, 1., 1.001)])
    close(np, exact, [1, 0, 0], 0)
    close(np, passes[[0, 1, 3]], [1, 1, 0], 0)
    R = np.diag([-1., -1., 1.])
    for key in ("symmetries_discrete", "symmetries_continuous"):
        info = {"diameter": diameter, key: [np.eye(4).reshape(-1).tolist()]}
        _, flags, errors = wrapper({"pts": pts}, info, [[I, z]], [[R, z]])
        close(np, flags, [1], 0); close(np, errors, [0], 0)
    return {"threshold_policy": "error < 0.1 * diameter; 등호는 실패"}


def metric_invalid(kind):
    np, _, wrapper = metric_setup()
    pose = [np.eye(3), np.zeros((3, 1))]
    gt, pred = [pose], [pose]
    info = {"diameter": 10.}
    if kind == "length":
        gt = [pose, pose]
    elif kind == "empty":
        gt, pred = [], []
    elif kind == "nan":
        pred = [[np.eye(3), np.full((3, 1), np.nan)]]
    elif kind == "diameter":
        info["diameter"] = -1.
    try:
        wrapper({"pts": np.zeros((3, 3))}, info, gt, pred)
    except (ValueError, AssertionError):
        return {"rejected": True}
    raise AssertionError(f"{kind}: 유효하지 않은 평가 입력을 거부하지 않음. 집계 왜곡 가능")


def cases(stage):
    if stage == "1":
        record(stage, "HCCE 256값·XYZ·Gray oracle·경계 복원", "최대 정수 오차 0, 연속 코드 범위 [0,1]", hcce)
        record(stage, "uint8 라벨의 NumPy 버전 호환", "float64 입력과 동일 코드", hcce_dtype)
    elif stage == "2":
        for h, w in ((480, 640), (720, 1280), (403, 639), (721, 1279)):
            record(stage, f"원본 resize/pad/K {w}x{h}", "K'=T K, 주점 이동 일치, 원본 K 보존", lambda h=h,w=w: camera_geometry(h,w))
        record(stage, "원본 crop 행렬·역변환·OpenCV 샘플링", "대응점 오차 <=3e-5px", crop_geometry)
        record(stage, "Kornia 실제 align_corners=False 샘플링", "좌표 ramp 최대 오차 <=0.05px", kornia_sampling)
    elif stage == "3":
        for mode in (0, 1, 2):
            record(stage, f"PnP 무잡음 mode={mode}", "회전 <=0.01deg, 이동 <=0.01mm, 3 seeds", lambda m=mode: pnp_accuracy(m))
        for mode in (1, 2):
            record(stage, f"PnP 잡음·outlier mode={mode}", "0.25px 잡음·약20% outlier; <=1deg/2mm", lambda m=mode: pnp_accuracy(m, True))
            record(stage, f"PnP OpenCV False 강제 반환 mode={mode}", "예외 없이 success=False, 유한 기본 Pose", lambda m=mode: pnp_forced_failure(m))
        for n in (0, 3, 4):
            record(stage, f"PnP 유효 대응점 {n}개", "모든 mode가 안전한 실패 반환", lambda n=n: pnp_sparse(n))
        record(stage, "앞·뒤 PnP 후보 조합", "동일 광선의 앞/뒤 대응점으로 정확한 Pose 선택", pnp_combined)
    else:
        record(stage, "BOP ADD·ADD-S 독립 수학 oracle", "이동 5mm, 대칭 회전 ADD-S=0, brute-force 일치", metric_oracle)
        record(stage, "HCCEPose 평가 선택·임계값 경계", "ADD/ADD-S 분기와 엄격한 직경10% 기준", metric_wrapper)
        for kind in ("length", "empty", "nan", "diameter"):
            record(stage, f"평가 입력 거부 계약: {kind}", "잘못된 입력을 명시적으로 거부", lambda k=kind: metric_invalid(k))


def dependency_report():
    report = {}
    packages = {"numpy": ["numpy"], "torch": ["torch"],
                "cv2": ["opencv-python", "opencv-contrib-python", "opencv-python-headless"],
                "scipy": ["scipy"], "kornia": ["kornia"]}
    for module, distributions in packages.items():
        versions = {}
        for dist in distributions:
            try:
                versions[dist] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                pass
        report[module] = {"module_found": importlib.util.find_spec(module) is not None,
                          "installed_distributions": versions}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="승인 후 실제 CPU 검증 실행")
    parser.add_argument("--check-deps", action="store_true", help="패키지 검색·설치 메타데이터만 출력; 패키지 import/검증/저장 없음")
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4"], default="all")
    parser.add_argument("--output-dir", type=Path, help="cpu_preflight/results 아래 존재하지 않는 새 디렉터리")
    args = parser.parse_args()
    if args.check_deps and args.run:
        parser.error("--check-deps와 --run은 동시에 사용할 수 없음")
    if args.check_deps:
        print(json.dumps(dependency_report(), ensure_ascii=False, indent=2)); return 0
    if not args.run:
        print("테스트 미실행. README.md의 명령·부작용을 확인하고 승인 후 --run을 사용하세요.")
        return 0
    base = (Path(__file__).resolve().parent / "results").resolve()
    out = (args.output_dir or base / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if base not in out.parents or out.exists():
        parser.error("출력은 cpu_preflight/results 하위의 존재하지 않는 디렉터리여야 합니다")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dependency_report()
    for stage in (["1", "2", "3", "4"] if args.stage == "all" else [args.stage]):
        cases(stage)
    counts = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    payload = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "python": sys.version, "executable": sys.executable,
               "root": str(ROOT), "dependencies": metadata, "evidence": EVIDENCE,
               "counts": counts, "results": RESULTS,
               "limitations": ["전체 프로젝트 import·GPU·신경망 forward·데이터셋 실행 없음",
                               "실제 모델 정확도와 SMT 성능을 검증하지 않음",
                               "SKIP은 성공이 아닌 미검증", "실패 계약 테스트의 FAIL도 숨기지 않음"]}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# HCCEPose CPU 사전 검증 결과", "", f"집계: {counts}", "",
             "합성 데이터·원본 함수 격리 검증이며 SMT 정확도 결과가 아닙니다.", "",
             "- PASS: 명시한 허용오차/계약 충족", "- FAIL: 오차 초과·실패 처리 결함 또는 환경/로더 오류",
             "- SKIP: 패키지 누락·로드 실패로 미검증. 자동 설치 없음", ""]
    for result in RESULTS:
        lines.extend([f"## [{result['status']}] {result['stage']}: {result['name']}", "",
                      f"기대: {result['expected']}", ""])
        if result["error"]:
            lines.extend([f"오류 원인: {result['error']}", "", "전체 traceback은 results.json 참조.", ""])
        if result["details"]:
            lines.extend(["```json", json.dumps(result["details"], ensure_ascii=False, indent=2), "```", ""])
    (out / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"결과 저장: {out}\n집계: {counts}")
    return 1 if counts["FAIL"] else (2 if counts["SKIP"] else 0)


if __name__ == "__main__":
    sys.exit(main())
