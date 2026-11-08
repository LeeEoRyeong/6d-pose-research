"""Convert poses_obj04/pose_*.npy into object-4 BOP annotations.

Run with Python and NumPy. Each pose must be a 4x4 model-to-camera
transform; translation is preserved and is expected to be in millimetres.
Outputs are rebuilt from all matching pose files beside this script.
"""

import json
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
POSE_DIR = BASE_DIR / "poses_obj04"
OUTPUT_GT = BASE_DIR / "scene_gt_obj04.json"
OUTPUT_CAMERA = BASE_DIR / "scene_camera_obj04.json"
OBJ_ID = 4
K = [984.70009004, 0.0, 635.41416344,
     0.0, 983.75715611, 362.61850196,
     0.0, 0.0, 1.0]


def main():
    pose_files = sorted(POSE_DIR.glob("pose_*.npy"))
    if not pose_files:
        raise FileNotFoundError(f"No pose_*.npy files found in {POSE_DIR}")

    scene_gt = {}
    scene_camera = {}
    for pose_path in pose_files:
        match = re.fullmatch(r"pose_(\d+)\.npy", pose_path.name)
        if match is None:
            raise ValueError(f"Invalid pose filename: {pose_path.name}")
        key = str(int(match.group(1)))
        if key in scene_gt:
            raise ValueError(f"Duplicate frame number {key}: {pose_path.name}")

        pose = np.load(pose_path, allow_pickle=False)
        if pose.shape != (4, 4) or pose.dtype.kind not in "fiu":
            raise ValueError(f"{pose_path.name}: expected a numeric 4x4 matrix")
        pose = pose.astype(np.float64)
        if not np.isfinite(pose).all():
            raise ValueError(f"{pose_path.name}: matrix contains non-finite values")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
            raise ValueError(f"{pose_path.name}: invalid homogeneous bottom row")
        rotation = pose[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0)):
            raise ValueError(f"{pose_path.name}: rotation is not a proper rotation matrix")

        scene_gt[key] = [{
            "cam_R_m2c": rotation.reshape(-1).tolist(),
            "cam_t_m2c": pose[:3, 3].tolist(),
            "obj_id": OBJ_ID,
        }]
        scene_camera[key] = {"cam_K": K.copy(), "depth_scale": 1.0}

    # Validate all inputs before writing either output.
    for path, data in ((OUTPUT_GT, scene_gt), (OUTPUT_CAMERA, scene_camera)):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")

    gt_count = sum(len(instances) for instances in scene_gt.values())
    print(f"Frames: {len(scene_gt)}, GT instances: {gt_count}")
    print(f"Saved: {OUTPUT_GT}")
    print(f"Saved: {OUTPUT_CAMERA}")


if __name__ == "__main__":
    main()
