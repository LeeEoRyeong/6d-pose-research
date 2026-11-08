"""Continuous BB obj04 labeling. Each frame is an immutable, standalone BOP scene."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from pose_watcher import snapshot, validate, exclusive_write, run
from verify_bop_projection_cli import find_undistorted_image

BASE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\RESEARCH_DATA\Documents\Vision6D\Output\export_pose\obj_000004.npy")
IMAGE_ROOT = Path(r"C:\RESEARCH_DATA\Desktop\BB_rgb\BB_rgb_obj04\run_20260813_164421\rgb_undistorted")
MODEL = Path(r"C:\RESEARCH_DATA\Desktop\Digital Twin\state_dataset+label+Object 3D Model\obj_000004.ply")


def now():
    return datetime.now().astimezone().isoformat()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def legacy_record(work, image_root):
    """Preserve known provenance without asserting an unknown original save time."""
    record = work / "legacy_000060.json"
    if record.exists():
        return
    path = BASE / "poses_obj04" / "pose_000060.npy"
    if not path.is_file():
        return
    data = path.read_bytes()
    result = "passed"
    try:
        validate(data)
    except Exception as exc:
        result = f"failed: {type(exc).__name__}: {exc}"
    try:
        image = str(find_undistorted_image(60, image_root).resolve())
    except (FileNotFoundError, ValueError):
        image = None
    other = BASE / "poses_obj04" / "pose_000043.npy"
    same = hashlib.sha256(other.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest() if other.is_file() else None
    exclusive_write(record, json_bytes({"frame": 60, "method": "기존 43번 Pose 복사본 (사용자 제공 이력)",
        "image": image, "saved_at": None, "saved_at_note": "기존 저장 시각 불명",
        "file_modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(), "pose_validation": result,
        "bytes_equal_to_frame_43": same, "recorded_at": now(), "pose_path": str(path)}))


def check_duplicate(frame, work):
    if frame in (43, 60):
        raise FileExistsError("기존 BB 43번/60번 라벨은 보호됩니다")
    for path in (BASE / "poses_obj04" / f"pose_{frame:06d}.npy",
                 BASE / "overlays_obj04" / f"verify_{frame:06d}_overlay.jpg",
                 work / f"frame_{frame:06d}"):
        if path.exists():
            raise FileExistsError(f"기존 출력: {path}")
    for name in ("scene_gt_obj04.json", "scene_camera_obj04.json"):
        path = BASE / name
        if path.exists() and str(frame) in json.loads(path.read_text(encoding="utf-8")):
            raise FileExistsError(f"기존 JSON에 프레임이 있습니다: {path}")


def wait_export(source, baseline, poll, stable, timeout):
    start = time.monotonic()
    candidate = None
    stable_since = None
    print("새 Pose Export 대기 중 (대기 취소: q, 종료: Ctrl+C)", flush=True)
    while True:
        if os.name == "nt" and sys.stdin.isatty():
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                raise KeyboardInterrupt
        current = snapshot(source)
        if current and (baseline is None or current[0] != baseline[0]):
            if current[0] != candidate:
                candidate, stable_since = current[0], time.monotonic()
            elif time.monotonic() - stable_since >= stable:
                return current[1]
        else:
            candidate, stable_since = None, None
        if timeout and time.monotonic() - start >= timeout:
            raise TimeoutError("새 안정 Export를 기다리는 시간이 초과되었습니다")
        time.sleep(poll)


def process(frame, args, work):
    check_duplicate(frame, work)
    image = find_undistorted_image(frame, args.image_root).resolve()
    if not args.model.is_file():
        raise FileNotFoundError(f"모델 없음: {args.model}")
    baseline = snapshot(args.source)
    if args.source.exists() and baseline is None:
        raise OSError("기존 Export를 안전하게 읽지 못했습니다")
    data = wait_export(args.source, baseline, args.poll, args.stable, args.timeout)
    pose = validate(data)
    digest = hashlib.sha256(data).hexdigest()
    if baseline and digest == baseline[0][3]:
        print("경고: 새 Export의 내용이 이전 Export와 동일합니다. 이미지와 PnP를 확인하세요.")
        answer = input("그래도 GT로 저장하려면 YES 입력 (q: 종료): ").strip()
        if answer.lower() == "q":
            raise KeyboardInterrupt
        if answer != "YES":
            print("저장을 취소했습니다.")
            return
    check_duplicate(frame, work)
    folder = work / f"frame_{frame:06d}"
    folder.mkdir()  # exclusive reservation; a partial result also prevents retry/overwrite
    pose_path = folder / "poses_obj04" / f"pose_{frame:06d}.npy"
    stage = "pose_save"
    saved_at = None
    failure = None
    try:
        exclusive_write(pose_path, data)
        saved_at = now()
        if pose_path.read_bytes() != data:
            raise OSError("저장된 Pose의 바이트가 Export와 다릅니다")
        stage = "bop"
        # Builder resolves its directory, so copy only its script into the new frame folder.
        exclusive_write(folder / "build_obj04_bop.py", (BASE / "build_obj04_bop.py").read_bytes())
        run(folder / "build_obj04_bop.py")
        gt = json.loads((folder / "scene_gt_obj04.json").read_text(encoding="utf-8"))
        camera = json.loads((folder / "scene_camera_obj04.json").read_text(encoding="utf-8"))
        key = str(frame)
        if (set(gt) != {key} or set(camera) != {key} or gt[key][0]["obj_id"] != 4
                or not np.allclose(gt[key][0]["cam_t_m2c"], pose[:3, 3])
                or not np.allclose(np.array(gt[key][0]["cam_R_m2c"]).reshape(3, 3), pose[:3, :3])):
            raise ValueError("BOP JSON이 저장된 Pose와 다릅니다")
        stage = "overlay"
        run(BASE / "verify_bop_projection_cli.py", frame, "--project-dir", folder,
            "--image-root", args.image_root, "--model", args.model)
        stage = "success"
        print(f"완료: {folder}", flush=True)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"[실패:{stage}] {failure}; 저장된 Pose는 보존됩니다: {pose_path}", file=sys.stderr, flush=True)
        raise
    finally:
        record = {"frame": frame, "method": "Vision6D 신규 Pose Export", "image": str(image),
                  "saved_at": saved_at, "sha256": digest, "pose_validation": "passed",
                  "same_as_previous_export": bool(baseline and digest == baseline[0][3]),
                  "stage": stage, "result": "success" if stage == "success" else "failed",
                  "failure": failure, "recorded_at": now(), "pose_path": str(pose_path)}
        try:
            exclusive_write(folder / "provenance.json", json_bytes(record))
        except Exception as exc:
            print(f"[기록 실패] {type(exc).__name__}: {exc}; Pose는 보존됩니다", file=sys.stderr, flush=True)
            if failure is None:
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--work-dir", type=Path, default=BASE / "continuous_BB_obj04")
    p.add_argument("--poll", type=float, default=0.2)
    p.add_argument("--stable", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=0)
    args = p.parse_args()
    if args.poll <= 0 or args.stable <= 0 or args.timeout < 0:
        p.error("poll/stable > 0, timeout >= 0 이어야 합니다")
    work = args.work_dir.resolve()
    if work == BASE or BASE not in work.parents:
        p.error("작업 폴더는 BB_obj04 내부의 하위 폴더여야 합니다")
    work.mkdir(parents=True, exist_ok=True)
    lock = work / ".continuous.lock"
    exclusive_write(lock, f"pid={os.getpid()}\n".encode())
    try:
        legacy_record(work, args.image_root)
        while True:
            value = input("프레임 번호 (0..999999, 종료 q): ").strip()
            if value.lower() == "q":
                break
            if not value.isascii() or not value.isdecimal() or not 0 <= int(value) <= 999999:
                print("올바른 프레임 번호를 입력하세요.")
                continue
            try:
                process(int(value), args, work)
            except Exception as exc:
                print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n안전하게 종료합니다. 저장된 Pose는 보존됩니다.")
    except Exception as exc:
        print(f"[시작 실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
