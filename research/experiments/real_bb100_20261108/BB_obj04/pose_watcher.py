"""Wait for one new Vision6D export, archive it, rebuild BOP and render overlay."""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SOURCE = Path(r"C:\RESEARCH_DATA\Documents\Vision6D\Output\export_pose\obj_000004.npy")
IMAGE_ROOT = Path(r"C:\RESEARCH_DATA\Desktop\Digital Twin\state_dataset+label+Object 3D Model\C_rgb_obj04\image_undistorted")
MODEL = IMAGE_ROOT.parent.parent / "obj_000004.ply"
STAGE = "startup"


def report(stage, message):
    global STAGE
    STAGE = stage
    print(f"[{stage}] {message}", flush=True)


def snapshot(path):
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except (FileNotFoundError, PermissionError):
        return None
    signature = (after.st_mtime_ns, after.st_ctime_ns, after.st_size, hashlib.sha256(data).hexdigest())
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        return None
    return signature, data


def validate(data):
    pose = np.load(io.BytesIO(data), allow_pickle=False)
    if pose.shape != (4, 4) or pose.dtype.kind not in "fiu" or not np.isfinite(pose).all():
        raise ValueError("Expected a finite numeric 4x4 pose")
    R = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid homogeneous bottom row")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-4, rtol=0):
        raise ValueError("Invalid rotation matrix")
    return pose


def exclusive_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise


def run(script, *arguments):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=int)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--timeout", type=float, default=0, help="Seconds; 0 waits indefinitely")
    parser.add_argument("--poll", type=float, default=0.2)
    parser.add_argument("--stable", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.frame <= 999999 or args.poll <= 0 or args.stable <= 0 or args.timeout < 0:
        parser.error("frame must be 0..999999; poll/stable > 0; timeout >= 0")
    project = args.project_dir.resolve()
    destination = project / "poses_obj04" / f"pose_{args.frame:06d}.npy"
    overlay = project / "overlays_obj04" / f"verify_{args.frame:06d}_overlay.jpg"
    baseline = snapshot(args.source)
    report("baseline", f"Source: {args.source}; state: {baseline[0] if baseline else 'missing/unreadable'}")
    if args.source.exists() and baseline is None:
        raise OSError("Cannot safely record the existing export; retry after export finishes")
    report("preflight", f"Frame {args.frame:06d}; Python: {sys.executable}")
    for path in (destination, overlay):
        if path.exists():
            raise FileExistsError(f"Already exists; refusing overwrite: {path}")
    for path in (project / "build_obj04_bop.py", project / "verify_bop_projection_cli.py", args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    import cv2
    import trimesh
    from verify_bop_projection_cli import find_undistorted_image
    image = find_undistorted_image(args.frame, args.image_root)
    report("preflight", f"Image: {image}; packages OK")
    lock = project / ".pose_watcher.lock"
    exclusive_write(lock, f"pid={os.getpid()} frame={args.frame}\n".encode())
    try:
        report("waiting", "Waiting for a NEW Pose export. Click Pose after PnP. Ctrl+C cancels.")
        start = time.monotonic()
        candidate, stable_since = None, None
        while True:
            current = snapshot(args.source)
            if current and (baseline is None or current[0] != baseline[0]):
                if candidate != current[0]:
                    candidate, stable_since = current[0], time.monotonic()
                elif time.monotonic() - stable_since >= args.stable:
                    data = current[1]
                    break
            else:
                candidate, stable_since = None, None
            if args.timeout and time.monotonic() - start >= args.timeout:
                raise TimeoutError("No new stable export before timeout")
            time.sleep(args.poll)
        report("validate", "New export is stable; validating pose")
        pose = validate(data)
        if baseline and hashlib.sha256(data).hexdigest() == baseline[0][3]:
            print("[WARNING] Export timestamp changed but pose bytes are unchanged. Verify PnP/image selection.", flush=True)
        report("save", str(destination))
        exclusive_write(destination, data)
        if destination.read_bytes() != data:
            raise OSError("Saved pose differs from export")
        report("save", "OK; pose preserved even if later stages fail")
        with tempfile.TemporaryDirectory(prefix="pose_watcher_", dir=project) as tmp:
            staging = Path(tmp)
            shutil.copytree(project / "poses_obj04", staging / "poses_obj04")
            shutil.copy2(project / "build_obj04_bop.py", staging / "build_obj04_bop.py")
            report("bop", "Running existing builder against all archived poses in staging")
            run(staging / "build_obj04_bop.py")
            gt = json.loads((staging / "scene_gt_obj04.json").read_text(encoding="utf-8"))
            camera = json.loads((staging / "scene_camera_obj04.json").read_text(encoding="utf-8"))
            key = str(args.frame)
            if set(gt) != set(camera) or not np.allclose(gt[key][0]["cam_t_m2c"], pose[:3, 3]) or not np.allclose(np.array(gt[key][0]["cam_R_m2c"]).reshape(3, 3), pose[:3, :3]):
                raise ValueError("Generated JSON does not match saved pose")
            report("overlay", "Rendering Pc = R * Pm + t with calibrated K")
            run(project / "verify_bop_projection_cli.py", args.frame, "--project-dir", staging, "--image-root", args.image_root, "--model", args.model)
            report("publish", "Backing up current JSONs and publishing verified results")
            backup = project / "watcher_backups" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{args.frame:06d}")
            backup.mkdir(parents=True)
            names = ("scene_gt_obj04.json", "scene_camera_obj04.json")
            for name in names:
                if (project / name).exists():
                    shutil.copy2(project / name, backup / name)
            replaced = []
            try:
                for name in names:
                    os.replace(staging / name, project / name)
                    replaced.append(name)
                exclusive_write(overlay, (staging / "overlays_obj04" / overlay.name).read_bytes())
            except BaseException:
                for name in replaced:
                    if (backup / name).exists():
                        shutil.copy2(backup / name, project / name)
                    else:
                        (project / name).unlink(missing_ok=True)
                raise
        report("success", f"Frame {args.frame:06d}: pose -> BOP JSON ({len(gt)} frames) -> overlay OK\nPose: {destination}\nOverlay: {overlay}\nJSON backup: {backup}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[CANCELLED:{STAGE}] Ctrl+C received. Existing poses remain preserved.", flush=True)
        sys.exit(130)
    except Exception as error:
        print(f"[FAILED:{STAGE}] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
