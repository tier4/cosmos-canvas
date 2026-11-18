# SPDX-License-Identifier: Apache-2.0
"""The ludus raster path without a GPU: the f-theta lens, a CARLA camera looking through
an f-theta frame, the ClipGT map ludus reads, the boxes it is handed, and the renderer
calling it per chunk (ludus itself replaced by a stand-in)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_canvas.omnidreams import mapping  # noqa: E402
from cosmos_canvas.omnidreams.ludus import box_arrays, prepare_clipgt  # noqa: E402
from cosmos_canvas.omnidreams.rig import FTheta, GeneratedCamera, look_through, quat_to_R  # noqa: E402

# an equidistant f-theta camera (300 px/rad, +-1 rad) on base_link, looking forward
FTHETA_CAMERA = {"width": 640, "height": 352,
                 "ftheta": {"principal_point": [320.0, 176.0], "fw_poly": [0.0, 300.0], "max_ray_angle": 1.0,
                            "linear_cde": [1.0, 0.0, 0.0]},
                 "cam_to_ego": {"translation": [1.3, 0.0, 1.9], "rotation_wxyz": [0.5, -0.5, 0.5, -0.5]}}


def _camera(tmp_path, doc=FTHETA_CAMERA) -> GeneratedCamera:
    (tmp_path / "cam.json").write_text(json.dumps(doc))
    return GeneratedCamera.from_json(tmp_path / "cam.json")


def test_an_ftheta_ray_lands_its_angle_times_the_polynomial_from_the_centre():
    lens = FTheta.from_dict({"principal_point": [320, 176], "fw_poly": [0, 300], "max_ray_angle": 1.0,
                             "linear_cde": [1.1, 0.05, 0.0]})
    theta = 0.3
    rays = np.array([[math.sin(theta), 0.0, math.cos(theta)],      # right
                     [0.0, math.sin(theta), math.cos(theta)],      # down
                     [0.0, 0.0, 1.0],                              # the axis
                     [0.0, 0.0, -1.0]])                            # behind
    uv, seen = lens.project(rays)
    r = 300 * theta
    np.testing.assert_allclose(uv[0], [320 + 1.1 * r, 176])
    np.testing.assert_allclose(uv[1], [320 + 0.05 * r, 176 + r])  # the linear term [[c, d], [e, 1]]
    np.testing.assert_allclose(uv[2], [320, 176])
    assert seen.tolist() == [True, True, True, False]


def test_past_max_ray_angle_the_polynomial_goes_on_along_its_tangent():
    """As ludus and NVIDIA's ray2pixel: max_ray_angle bounds the polynomial, not the view."""
    lens = FTheta.from_dict({"principal_point": [0, 0], "fw_poly": [0, 300, -50], "max_ray_angle": 0.5})
    uv, seen = lens.project(np.array([[math.sin(t), 0.0, math.cos(t)] for t in (0.4, 0.8)]))
    assert uv[0, 0] == pytest.approx(300 * 0.4 - 50 * 0.16)                       # the polynomial
    assert uv[1, 0] == pytest.approx(300 * 0.5 - 50 * 0.25 + 0.3 * (300 - 100 * 0.5))   # r(0.5) + 0.3 r'(0.5)
    assert seen.all()


def test_a_camera_file_has_one_lens(tmp_path):
    cam = _camera(tmp_path)
    assert cam.lens == "ftheta" and cam.K is None
    both = dict(FTHETA_CAMERA, intrinsic=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="either"):
        _camera(tmp_path, both)


def test_a_pinhole_looks_through_an_ftheta_frame(tmp_path):
    """A dot 0.3 rad right of the f-theta axis shows up at fx tan(0.3) right of a pinhole's
    centre when both cameras stand at the same pose; off the frame the pinhole gets black."""
    cam = _camera(tmp_path)
    frame = np.zeros((352, 640, 3), np.uint8)
    frame[:] = 40                                         # the generated view: grey
    u0 = round(320 + 300 * 0.3)
    frame[174:179, u0 - 2:u0 + 3] = 255                   # the dot
    pose = np.eye(4)
    pose[:3, :3] = quat_to_R(FTHETA_CAMERA["cam_to_ego"]["rotation_wxyz"])
    K = np.array([[200.0, 0, 256], [0, 200.0, 128], [0, 0, 1]])
    out = look_through(frame, cam, pose, K, pose, 512, 256)
    bright = np.argwhere(out.max(axis=2) > 200)
    assert len(bright)
    v, u = bright.mean(axis=0)
    assert u == pytest.approx(256 + 200 * math.tan(0.3), abs=1.0) and v == pytest.approx(128, abs=1.0)
    # 80 deg off the axis lands 300 * 1.40 = 419 px from the centre, off the 640 px frame:
    # a pinhole 160 deg wide sees black at its edge, grey inside
    wide = np.array([[256 / math.tan(math.radians(80)), 0, 256], [0, 256 / math.tan(math.radians(80)), 128],
                     [0, 0, 1]])
    out = look_through(frame, cam, pose, wide, pose, 512, 256)
    assert out[128, 0].max() == 0 and out[128, 256].max() > 0


def test_a_camera_looking_away_from_an_ftheta_frame_gets_black(tmp_path):
    cam = _camera(tmp_path)
    frame = np.full((352, 640, 3), 255, np.uint8)
    pose = np.eye(4)
    pose[:3, :3] = quat_to_R(FTHETA_CAMERA["cam_to_ego"]["rotation_wxyz"])
    back = pose.copy()
    back[:3, :3] = pose[:3, :3] @ np.diag([-1.0, 1.0, -1.0])          # turned 180 deg about its y axis
    K = np.array([[200.0, 0, 256], [0, 200.0, 128], [0, 0, 1]])
    assert look_through(frame, cam, pose, K, back, 512, 256).max() == 0


def _write(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def _pts(xyz):
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in xyz]


def test_prepare_clipgt_cuts_lines_in_intersections_and_replaces_the_lights(tmp_path):
    import pyarrow.parquet as pq

    square = _pts([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)])
    _write(tmp_path / "c.intersection_area.parquet", [{"intersection_area": {"location": square}}])
    _write(tmp_path / "c.lane_line.parquet", [{"lane_line": {
        "line_rail": _pts([(-10, 5, 0), (20, 5, 0)]), "colors": ["WHITE"], "styles": ["SOLID_DOUBLE"]}}])
    _write(tmp_path / "c.road_boundary.parquet", [{"road_boundary": {"location": _pts([(-10, -2, 0), (20, -2, 0)])}}])
    _write(tmp_path / "c.traffic_light.parquet", [{"traffic_light": {
        "center": {"x": 1.0, "y": 2.0, "z": 0.05}, "dimensions": {"x": 0.6, "y": 0.6, "z": 1.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}])
    head = mapping.StaticBox("traffic_light", np.array([1.0, 2.0, 5.0]), np.eye(3), np.array([0.4, 0.4, 1.0]))

    counts = prepare_clipgt(tmp_path, "c", [head])

    lines = pq.read_table(tmp_path / "c.lane_line.parquet").to_pylist()
    assert counts["lane_line"] == len(lines) == 2                    # the stretch inside the junction is gone
    xs = sorted((min(p["x"] for p in r["lane_line"]["line_rail"]), max(p["x"] for p in r["lane_line"]["line_rail"]))
                for r in lines)
    assert xs[0][0] == -10 and xs[0][1] <= 1 and xs[1][0] >= 9 and xs[1][1] == 20
    assert all(r["lane_line"]["styles"] == ["SOLID_GROUP"] for r in lines)
    assert counts["road_boundary"] == 1                              # outside the junction: kept whole
    (light,) = pq.read_table(tmp_path / "c.traffic_light.parquet").to_pylist()
    assert light["traffic_light"]["center"] == {"x": 1.0, "y": 2.0, "z": 5.0}
    assert light["traffic_light"]["dimensions"] == {"x": 0.4, "y": 0.4, "z": 1.0}
    assert light["traffic_light"]["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    prepare_clipgt(tmp_path, "c", [])                                # no heads: no lights at all
    assert not (tmp_path / "c.traffic_light.parquet").exists()
    # roadgen's empty rig (no route) gets a camera: ludus' loader needs one
    (row,) = pq.read_table(tmp_path / "c.calibration_estimate.parquet").to_pylist()
    (sensor,) = json.loads(row["calibration_estimate"]["rig_json"])["rig"]["sensors"]
    assert sensor["name"].startswith("camera:") and sensor["properties"]["polynomial"]


def test_box_arrays_keep_one_track_per_box_with_xyzw_quaternions():
    yaw = np.eye(4)
    yaw[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]        # 90 deg left
    poses = np.repeat(yaw[None], 3, axis=0)
    poses[:, 0, 3] = [0.0, 1.0, 2.0]
    a = box_arrays([("car", poses, (4.6, 2.0, 1.5)), ("pedestrian", np.repeat(np.eye(4)[None], 3, 0),
                                                       (0.5, 0.5, 1.8))], [0, 33333, 66667])
    assert a["track_timestamps_us"].tolist() == [0, 33333, 66667] * 2
    assert a["track_lengths"].tolist() == [3, 3]
    np.testing.assert_allclose(a["translations"][:3, 0], [0.0, 1.0, 2.0])
    s = math.sqrt(0.5)
    np.testing.assert_allclose(a["quaternions"][0], [0.0, 0.0, s, s], atol=1e-6)
    np.testing.assert_allclose(a["quaternions"][3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(a["scales"], [[4.6, 2.0, 1.5], [0.5, 0.5, 1.8]], rtol=1e-6)
    with pytest.raises(ValueError, match="poses"):
        box_arrays([("car", poses[:2], (1, 1, 1))], [0, 1, 2])


# -- the renderer with a stand-in for ludus --------------------------------------------------

class FakeLudus:
    """Stands in for LudusRaster: records the chunks it is asked to draw; draws flat grey."""

    lens = "ftheta"

    def __init__(self, camera):
        self.camera = camera
        self.calls = []

    def load_map(self, world):
        return {}

    def draw(self, chunk):
        self.calls.append((np.asarray(chunk.cameras), list(chunk.times_us), list(chunk.boxes)))
        return np.full((len(chunk.times_us), self.camera.height, self.camera.width, 3), 90, np.uint8)


@pytest.fixture
def ludus_renderer(tmp_path, monkeypatch):
    from PIL import Image
    from test_omnidreams import EchoStream

    from cosmos_canvas.omnidreams.renderer import OmniDreamsRenderer

    (tmp_path / "cam.json").write_text(json.dumps(FTHETA_CAMERA))
    Image.new("RGB", (640, 352)).save(tmp_path / "first.png")
    r = OmniDreamsRenderer({"camera": "cam.json", "first_image": "first.png", "raster": "ludus", "fps": 30})

    def make(camera, world):
        if camera.lens != "ftheta":               # as LudusRaster refuses it
            from cosmos_canvas.omnidreams.ludus import LudusRaster

            LudusRaster(camera)
        r.fake = FakeLudus(camera)
        return r.fake

    monkeypatch.setattr(r, "_make_raster", make)
    monkeypatch.setattr(r, "_load_stream", lambda world: EchoStream())
    r.test_dir = tmp_path
    return r


def _world(r):
    from carla_canvas import WorldInfo

    return WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=r.test_dir, opendrive="<OpenDRIVE/>")


def test_ludus_draws_each_chunk_with_the_actors_at_the_frames_times(ludus_renderer):
    from test_omnidreams import _scene, _view

    r = ludus_renderer
    r.load_world(_world(r))
    frames = [r.render_camera(_view(_scene(0.05 * i, 15.0), r)).rgb for i in range(5)]   # 0.2 s: frame 6
    (_, first, _), (poses, times_us, boxes) = r.fake.calls
    assert first == [0, 33333, 66667, 100000, 133333]                   # chunk 0 at 30 Hz
    assert times_us == [round(f / 30 * 1e6) for f in range(5, 13)]      # chunk 1: frames 5-12
    assert len(poses) == 8 and poses[1][0, 3] - poses[0][0, 3] == pytest.approx(10.0 / 30)   # the ego at 10 m/s
    (cls, box_poses, lwh), = boxes                                        # the car ahead, also at 10 m/s
    assert cls == "car" and np.allclose(lwh, (4.6, 2.0, 1.5)) and len(box_poses) == 8
    gap = box_poses[:, 0, 3] - poses[:, 0, 3]                             # car centre - camera
    np.testing.assert_allclose(gap, 15.0 - (1.3 - 2.45 / 2), atol=1e-6)
    centre = frames[-1][96:160, 192:320]
    assert centre.min() > 0                                               # the CARLA camera sees the f-theta frame


def test_ludus_needs_an_ftheta_camera(ludus_renderer):
    from test_omnidreams import CAMERA

    r = ludus_renderer
    (r.test_dir / "cam.json").write_text(json.dumps(CAMERA))
    with pytest.raises(ValueError, match="ftheta"):
        r.load_world(_world(r))


def test_a_checkpoint_needs_the_variable_its_configuration_reads(monkeypatch):
    """NVIDIA's configurations download their weights and read no variable: a checkpoint
    given for one is refused rather than written to an unnamed variable."""
    from cosmos_canvas.omnidreams.stream import OmniDreamsStream

    with pytest.raises(ValueError, match="checkpoint_env"):
        OmniDreamsStream(config="omnidreams", checkpoint="/x.pt", checkpoint_env="", device="cpu")
