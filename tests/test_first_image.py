# SPDX-License-Identifier: Apache-2.0
"""The Transfer-Auto first-image path without a GPU: the f-theta lens inverted, one lens
resampled into another, and the WSM capture's frame times, poses, boxes and request
(the WSM renderer replaced by a stand-in)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_canvas.omnidreams.rig import FTheta, GeneratedCamera, resample  # noqa: E402

LENS = {"principal_point": [319.5, 175.5], "fw_poly": [0.0, 300.0, -10.0, 4.0], "max_ray_angle": 0.6,
        "linear_cde": [1.02, 0.01, 0.0]}
CAMERA = {"width": 640, "height": 352, "ftheta": LENS,
          "cam_to_ego": {"translation": [1.7, 0.0, 1.45], "rotation_wxyz": [0.5, -0.5, 0.5, -0.5]}}


def test_unproject_inverts_project_past_max_ray_angle():
    lens = FTheta.from_dict(LENS)
    theta = np.array([0.0, 0.2, 0.59, 0.9, 1.2])
    phi = np.array([0.3, 1.0, 2.0, -2.5, 3.0])
    rays = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    uv, _ = lens.project(rays)
    np.testing.assert_allclose(lens.unproject(uv), rays, atol=1e-5)


def test_a_lens_resampled_into_itself_is_unchanged(tmp_path):
    (tmp_path / "cam.json").write_text(json.dumps(CAMERA))
    cam = GeneratedCamera.from_json(tmp_path / "cam.json")
    rng = np.random.default_rng(0)
    frame = (rng.random((352, 640, 3)) * 255).astype(np.uint8)
    out, outside = resample(frame, FTheta.from_dict(LENS), cam)
    assert outside < 1e-3
    assert np.abs(out[10:-10, 10:-10].astype(int) - frame[10:-10, 10:-10]).max() <= 2


def test_a_wider_camera_sees_black_past_the_lens(tmp_path):
    wide = dict(CAMERA, ftheta=dict(LENS, fw_poly=[0.0, 150.0]))      # half the focal length: twice the view
    (tmp_path / "cam.json").write_text(json.dumps(wide))
    out, outside = resample(np.full((352, 640, 3), 200, np.uint8), FTheta.from_dict(LENS),
                            GeneratedCamera.from_json(tmp_path / "cam.json"))
    assert 0.3 < outside < 0.9 and out[176, 0].max() == 0 and out[176, 320].max() == 200


# -- the capture --------------------------------------------------------------------------

class FakeWsm:
    def __init__(self):
        self.calls = []

    def render(self, camera_pose, boxes):
        self.calls.append((np.asarray(camera_pose), boxes))
        return np.zeros((720, 1280, 3), np.uint8)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    from carla_canvas import WorldInfo

    from cosmos_canvas.omnidreams import first_image as F

    pytest.importorskip("scipy")
    (tmp_path / "cam.json").write_text(json.dumps(CAMERA))
    written = {}
    monkeypatch.setattr(F, "write_mp4", lambda path, frames, fps: written.update(path=path, n=len(frames), fps=fps))
    monkeypatch.setattr(F, "transfer_lens", lambda: {"width": 1280, "height": 720, "ftheta": LENS})
    r = F.WsmCapture({"camera": "cam.json", "out_dir": "fi", "caption": "A drive.", "frames": 9})
    fake = FakeWsm()
    monkeypatch.setattr(r, "_wsm_camera", lambda xodr: fake)
    r.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=tmp_path, opendrive="<OpenDRIVE/>"))
    return r, fake, written, tmp_path / "fi"


def test_the_capture_draws_30_hz_frames_from_the_cameras_mount_with_the_actors(capture):
    from test_omnidreams import _scene, _view

    r, fake, written, out = capture
    for i in range(6):                                     # 0.25 s at 20 Hz: frame 8 is at 0.267 s
        r.render_camera(_view(_scene(0.05 * i, 15.0), r))
    assert not fake.calls
    r.render_camera(_view(_scene(0.3, 15.0), r))
    assert len(fake.calls) == 9 and written["n"] == 9 and written["fps"] == 30.0
    poses = np.stack([p for p, _ in fake.calls])
    np.testing.assert_allclose(np.diff(poses[:, 0, 3]), 10.0 / 30, atol=1e-6)    # the ego at 10 m/s
    np.testing.assert_allclose(poses[0, 0, 3], -2.45 / 2 + 1.7)                   # base_link + the mount
    (car,) = fake.calls[4][1]
    assert car.object_type == "Car" and np.allclose(car.lwh, (4.6, 2.0, 1.5))
    assert car.box_to_world[0, 3] == pytest.approx(10.0 * 4 / 30 + 15.0)
    (req,) = [json.loads(line) for line in (out / "request.jsonl").read_text().splitlines()]
    assert req["model_mode"] == "text2video" and req["num_frames"] == 9 and req["num_conditional_frames"] == 0
    (view,) = req["multiview"]["views"]
    assert view["camera_key"] == "camera_front_wide_120fov" and view["control_path"].endswith("wsm.mp4")
    caption = json.loads(Path(view["caption_path"]).read_text())["caption_structured"]
    assert caption == {"chunk_0_9": {"caption": "A drive.", "start_frame": 0, "end_frame": 9}}
    r.render_camera(_view(_scene(0.35, 15.0), r))         # written once
    assert len(fake.calls) == 9


def test_the_capture_needs_frames_the_tokenizer_takes(tmp_path):
    from cosmos_canvas.omnidreams.first_image import WsmCapture

    with pytest.raises(ValueError, match="1 \\+ 4k"):
        WsmCapture({"camera": "c.json", "out_dir": "o", "caption": "x", "frames": 10})
