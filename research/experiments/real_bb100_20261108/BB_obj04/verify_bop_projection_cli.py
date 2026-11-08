"""Project model vertices using the established BOP model-to-camera convention."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

IMAGE_ROOT = Path(r"C:\RESEARCH_DATA\Desktop\Digital Twin\state_dataset+label+Object 3D Model\C_rgb_obj04\image_undistorted")
MODEL_PATH = IMAGE_ROOT.parent.parent / "obj_000004.ply"


def find_undistorted_image(frame_id, root=IMAGE_ROOT):
    for name in (f"{frame_id:06d}.jpg", f"{frame_id:06d}_undistorted.jpg"):
        matches = sorted(p for p in root.rglob(name) if p.is_file())
        if len(matches) > 1:
            raise ValueError("Ambiguous image matches: " + ", ".join(map(str, matches)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No image for {frame_id:06d} under {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", type=int)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    image_path = find_undistorted_image(args.frame, args.image_root)
    output = args.project_dir / "overlays_obj04" / f"verify_{args.frame:06d}_overlay.jpg"
    if output.exists():
        raise FileExistsError(f"Overlay already exists: {output}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Cannot read image: {image_path}")
    if image.shape[:2] != (720, 1280):
        raise ValueError(f"Expected 1280x720 image, got {image.shape[:2]}")
    gt = json.loads((args.project_dir / "scene_gt_obj04.json").read_text(encoding="utf-8"))[str(args.frame)]
    camera = json.loads((args.project_dir / "scene_camera_obj04.json").read_text(encoding="utf-8"))[str(args.frame)]
    objects = [item for item in gt if item["obj_id"] == 4]
    if len(objects) != 1:
        raise ValueError("Expected exactly one obj04 pose")
    R = np.asarray(objects[0]["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(objects[0]["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    expected_K = np.array([[984.70009004, 0, 635.41416344], [0, 983.75715611, 362.61850196], [0, 0, 1]])
    if not np.allclose(K, expected_K, atol=1e-8, rtol=0):
        raise ValueError("Camera K differs from the calibrated K")
    mesh = trimesh.load(str(args.model), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    points = ((R @ vertices.T) + t).T  # Pc = R * Pm + t; no inverse or axis flip.
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    u = K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]
    v = K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]
    uv = np.round(np.column_stack((u, v)))
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
    uv = uv[valid].astype(int)
    if not len(uv):
        raise ValueError("No projected vertices inside the image")
    step = max(1, len(uv) // 5000)
    for x, y in uv[::step]:
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise OSError("JPEG encoding failed")
    with output.open("xb") as handle:
        handle.write(encoded.tobytes())
    print(f"Image: {image_path}\nProjected vertices: {len(uv)}\nOverlay: {output}")


if __name__ == "__main__":
    main()
