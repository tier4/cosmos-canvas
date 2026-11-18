"""Write a ClipGT scene's camera as the OmniDreams renderer's f-theta camera, at the generated size.

NVIDIA's public OmniDreams checkpoint generates ``camera_front_wide_120fov``, an f-theta
camera whose calibration is in the rig of NVIDIA's ClipGT scenes (the ``.usdz`` archives
of https://huggingface.co/datasets/nvidia/omni-dreams-scenes). This reads it as
FlashDreams' apps do, with ludus' ClipGT loader scaled to the generated size, and
writes the renderer's ``camera`` file (``cosmos_canvas/omnidreams/rig.py``):

    {"width": 1280, "height": 704,
     "ftheta": {"principal_point": [...], "fw_poly": [...], "max_ray_angle": ..., "linear_cde": [...]},
     "cam_to_ego": {"translation": [...], "rotation_wxyz": [...]}}   # camera (OpenCV axes) -> rig

Run where FlashDreams (ludus) is installed:

    python scripts/ftheta_camera.py clipgt-<id>.usdz --out camera_front_wide_120fov.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_canvas.omnidreams.ludus import CV_FROM_FLU  # noqa: E402
from cosmos_canvas.omnidreams.rig import quat_wxyz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", type=Path, help="a ClipGT scene: .usdz archive or directory")
    ap.add_argument("--camera", default="camera:front:wide:120fov",
                    help="the rig's camera (ClipGT name; underscores also accepted)")
    ap.add_argument("--size", default="1280x704", help="the generated size, WxH (default 1280x704)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from ludus_renderer import load_scene

    width, height = (int(v) for v in a.size.lower().split("x"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene = load_scene(a.scene, device=device, target_resolution=(width, height), include_ego_trajectory=False,
                       include_dynamic_obstacles=False)
    names = {n.replace(":", "_"): n for n in scene.camera_name_to_id}
    name = a.camera if a.camera in scene.camera_name_to_id else names.get(a.camera.replace(":", "_"))
    if name is None:
        raise SystemExit(f"{a.scene} has no camera {a.camera!r}; it has {sorted(scene.camera_name_to_id)}")
    cam = scene.cameras[scene.camera_name_to_id[name]]
    ld = (cam.linear_distortion.cpu().numpy() if cam.linear_distortion is not None else np.eye(2)).astype(float)
    flu_to_rig = scene.sensor_to_rig[name].cpu().numpy().astype(float)
    cv_to_rig = flu_to_rig @ CV_FROM_FLU.T
    doc = {
        "width": width,
        "height": height,
        "ftheta": {"principal_point": cam.principal_point.cpu().tolist(),
                   "fw_poly": cam.fw_poly.cpu().tolist(),
                   "max_ray_angle": float(cam.max_ray_angle),
                   "linear_cde": [ld[0, 0], ld[0, 1], ld[1, 0]]},
        "cam_to_ego": {"translation": cv_to_rig[:3, 3].tolist(), "rotation_wxyz": quat_wxyz(cv_to_rig[:3, :3])},
        "source": f"{a.scene.name} {name}, scaled to {width}x{height} by ludus",
    }
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
