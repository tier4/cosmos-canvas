# SPDX-License-Identifier: Apache-2.0
"""The OmniDreams backend without a GPU: camera geometry, chunk timing, the renderer's chain.

The model is replaced by one that returns its rasters as the generated frames, and
the raster by one that fills each road user's projected box, so a box in front of
the ego must show up in a CARLA camera through the whole chain (tracking, base_link,
the chunk handed to the raster, the homography).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_canvas.omnidreams.rig import homography, quat_to_R, quat_wxyz  # noqa: E402
from cosmos_canvas.omnidreams.stream import FIRST_CHUNK, chunk_frames, chunk_of_frame  # noqa: E402


def test_chunks_cover_every_frame_once():
    seen = []
    for k in range(6):
        frames = chunk_frames(k)
        assert all(chunk_of_frame(f) == k for f in frames)
        seen += list(frames)
    assert seen == list(range(len(seen)))
    assert len(chunk_frames(0)) == FIRST_CHUNK and len(chunk_frames(1)) == 8






def test_homography_of_a_camera_to_itself_is_identity():
    K = np.array([[500.0, 0, 640], [0, 550, 350], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = quat_to_R([0.5, -0.5, 0.5, -0.5])
    assert np.allclose(homography(K, pose, K, pose), np.eye(3))
    assert np.allclose(quat_to_R(quat_wxyz(pose[:3, :3])), pose[:3, :3])


# -- the renderer, end to end with a stand-in model --------------------------------

# a front-wide pinhole on base_link, looking forward (OpenCV axes -> FLU)
CAMERA = {"width": 640, "height": 352, "intrinsic": [[230.0, 0, 320], [0, 277.0, 176], [0, 0, 1]],
          "cam_to_ego": {"translation": [1.3, 0.0, 1.9], "rotation_wxyz": [0.5, -0.5, 0.5, -0.5]}}


class BoxRaster:
    """A pinhole raster that fills the image rectangle around each road user's box."""

    lens = "pinhole"

    def __init__(self, camera, options=None, world=None):
        self.camera = camera
        self.maps = 0

    def load_map(self, world):
        self.maps += 1
        return {}

    def draw(self, chunk):
        cam = self.camera
        out = np.zeros((len(chunk.times_us), cam.height, cam.width, 3), np.uint8)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        for i, pose in enumerate(chunk.cameras):
            world_to_cam = np.linalg.inv(pose)
            for _, poses, lwh in chunk.boxes:
                corners = (poses[i][:3, :3] @ (signs * np.asarray(lwh) / 2).T).T + poses[i][:3, 3]
                c = (world_to_cam[:3, :3] @ corners.T).T + world_to_cam[:3, 3]
                if (c[:, 2] <= 0.1).any():
                    continue
                uv = (cam.K @ c.T).T
                uv = uv[:, :2] / uv[:, 2:]
                u0, v0 = np.clip(uv.min(0), 0, [cam.width, cam.height]).astype(int)
                u1, v1 = np.clip(uv.max(0), 0, [cam.width, cam.height]).astype(int)
                out[i, v0:v1, u0:u1] = 255
        return out


class EchoStream:
    """Returns the rasters it is given as the generated frames."""

    def __init__(self):
        self.index = 0
        self.started = 0

    def start(self, caption, first_image, seed=0):
        self.index = 0
        self.started += 1

    @property
    def next_chunk_frames(self):
        return chunk_frames(self.index)

    def step(self, rasters):
        self.index += 1
        return rasters, 0.0


@pytest.fixture
def renderer(tmp_path, monkeypatch):
    from PIL import Image

    from cosmos_canvas.omnidreams.renderer import OmniDreamsRenderer

    (tmp_path / "cam.json").write_text(json.dumps(CAMERA))
    Image.new("RGB", (640, 352)).save(tmp_path / "first.png")
    r = OmniDreamsRenderer({"camera": "cam.json", "first_image": "first.png", "fps": 10})
    monkeypatch.setattr(r, "_make_raster", lambda camera, world: BoxRaster(camera))
    monkeypatch.setattr(r, "_load_stream", lambda world: EchoStream())
    r.test_dir = tmp_path
    return r


def _scene(t: float, car_ahead_m: float):
    from carla_canvas.geometry import Transform
    from carla_canvas.scene import ActorInfo, ActorState, BoundingBox, Scene

    s = Scene()
    s.reset("Carla/Maps/Town10HD_Opt", 1)
    s.add_actor(ActorInfo(1, "vehicle.lincoln.mkz", bounding_box=BoundingBox((0, 0, 0.75), (2.45, 1.0, 0.75))))
    s.add_actor(ActorInfo(2, "vehicle.tesla.model3", attributes={"base_type": "car"},
                          bounding_box=BoundingBox((0, 0, 0.75), (2.3, 1.0, 0.75))))
    s.add_actor(ActorInfo(3, "sensor.camera.rgb", parent_id=1))
    s.elapsed_seconds = t
    ego = Transform((10.0 * t, 0.0, 0.0), (0.0, 0.0, 0.0))          # 10 m/s along x
    s.states = {1: ActorState(ego),
                2: ActorState(Transform((10.0 * t + car_ahead_m, 0.0, 0.0), (0.0, 0.0, 0.0))),
                3: ActorState(Transform((10.0 * t + 0.1, 0.0, 1.9), (0.0, 0.0, 0.0)))}
    return s


def _view(scene, renderer):
    from carla_canvas import CameraView
    from carla_canvas.geometry import PinholeCamera

    return CameraView(camera=PinholeCamera(512, 256, 62.0, scene.states[3].transform), scene=scene,
                      outputs=frozenset({"rgb"}), sensor_id=3, frame=renderer.coordinate_frame)


def test_a_car_ahead_reaches_the_carla_camera(renderer):
    from carla_canvas import WorldInfo

    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    seen = []
    for i in range(12):                                   # 0.55 s at 20 Hz
        scene = _scene(0.05 * i, 15.0)
        seen.append(renderer.render_camera(_view(scene, renderer)).rgb)
    frame = seen[-1]
    assert frame.shape == (256, 512, 3)
    centre = frame[96:192, 192:320]
    assert (centre.max(axis=2) > 0).mean() > 0.05         # the car's box, in the middle of the view
    assert renderer._stream.index == 2                    # 0.55 s needs frame 6: chunks 0 and 1
    assert renderer._stream.started == 1


def test_frames_follow_the_checkpoints_rate(renderer):
    """At ``fps: 30`` (NVIDIA's public checkpoint) a chunk's 8 frames span 8/30 s: a tick
    at 0.55 s reads frame 16 (chunk 2), and each frame is drawn from the ego at f / 30 s."""
    from carla_canvas import WorldInfo

    renderer.fps = 30.0
    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    for i in range(12):                                   # 0.55 s at 20 Hz
        renderer.render_camera(_view(_scene(0.05 * i, 15.0), renderer))
    assert renderer._stream.index == 3                    # frame 16 is in chunk 2 (frames 13-20)
    drawn = renderer._rollout.drawn_at
    assert drawn[13][0, 3] - drawn[5][0, 3] == pytest.approx(10.0 * 8 / 30)   # 10 m/s over 8 frames


def test_a_jump_back_in_time_starts_a_new_rollout(renderer):
    from carla_canvas import WorldInfo

    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    for t in (0.0, 0.05, 0.1, 0.0):
        renderer.render_camera(_view(_scene(t, 15.0), renderer))
    assert renderer._stream.started == 2


def _turning_scene(t: float, yaw_deg: float):
    """The ego (and its camera) at *t*, yawed by *yaw_deg*; a car 15 m ahead of where it started."""
    from carla_canvas.geometry import Transform
    from carla_canvas.scene import ActorInfo, ActorState, BoundingBox, Scene

    s = Scene()
    s.reset("Carla/Maps/Town10HD_Opt", 1)
    s.add_actor(ActorInfo(1, "vehicle.lincoln.mkz", bounding_box=BoundingBox((0, 0, 0.75), (2.45, 1.0, 0.75))))
    s.add_actor(ActorInfo(2, "vehicle.tesla.model3", attributes={"base_type": "car"},
                          bounding_box=BoundingBox((0, 0, 0.75), (2.3, 1.0, 0.75))))
    s.add_actor(ActorInfo(3, "sensor.camera.rgb", parent_id=1))
    s.elapsed_seconds = t
    s.states = {1: ActorState(Transform((0.0, 0.0, 0.0), (0.0, yaw_deg, 0.0))),
                2: ActorState(Transform((15.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
                3: ActorState(Transform((0.1, 0.0, 1.9), (0.0, yaw_deg, 0.0)))}
    return s


def test_a_frame_is_turned_from_where_it_was_drawn_to_where_the_camera_is(renderer):
    """Frames 0-4 are drawn when frame 0 is due, at the poses extrapolated then (no turn).
    If the ego then turns, frame 3 must be shown turned: the homography starts from the
    pose the frame was drawn at, not from the track as it stands when the frame is read."""
    from carla_canvas import WorldInfo

    outputs = {}
    for turn in (0.0, 10.0):
        renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                      opendrive="<OpenDRIVE/>"))
        for t, yaw in ((0.0, 0.0), (0.1, turn), (0.2, turn), (0.3, turn)):
            frame = renderer.render_camera(_view(_turning_scene(t, yaw), renderer)).rgb
        outputs[turn] = frame.astype(np.int16)
    assert renderer._stream.index == 1                    # all four ticks read chunk 0
    assert np.abs(outputs[10.0] - outputs[0.0]).mean() > 1.0




def test_a_camera_looking_away_from_the_generated_view_gets_black(renderer):
    """A rear-facing CARLA camera shares no ray with the generated front view: nothing of
    it may show up (a homography also maps points behind the source camera)."""
    from carla_canvas import CameraView, WorldInfo
    from carla_canvas.geometry import PinholeCamera, Transform

    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    scene = _scene(0.0, 15.0)
    rear = PinholeCamera(512, 256, 62.0, Transform((-0.5, 0.0, 1.9), (0.0, 180.0, 0.0)))
    frame = renderer.render_camera(CameraView(camera=rear, scene=scene, outputs=frozenset({"rgb"}),
                                              sensor_id=3, frame=renderer.coordinate_frame)).rgb
    assert frame.max() == 0


@pytest.mark.parametrize("steps", [2, 4])
def test_the_stream_sets_the_schedule_whatever_the_configs_default(monkeypatch, steps):
    """A configuration may default to four steps: `steps` must still pick the schedule, two
    included, without touching FlashDreams' own object and without deep-copying it
    (multi-view configurations hold mapping proxies)."""
    import sys
    import types

    from cosmos_canvas.omnidreams import stream as S

    built = []

    class Pipe:
        def to(self, **_):
            return self

        def eval(self):
            return self

    class Cfg:
        def __init__(self):
            self.view_ids = types.MappingProxyType({"camera_front": 0})
            self.diffusion_model = types.SimpleNamespace(scheduler=types.SimpleNamespace(
                num_inference_steps=4, denoising_timesteps=[1000, 750, 500, 250]))

        def setup(self):
            built.append(self)
            return Pipe()

    registered = Cfg()
    fake = types.ModuleType("omnidreams.config")
    fake.OMNIDREAMS_CONFIGS = {"four-step-default": registered}
    monkeypatch.setenv("FAKE_CKPT", "")            # restored after the test (the stream sets it)
    monkeypatch.setitem(sys.modules, "omnidreams", types.ModuleType("omnidreams"))
    monkeypatch.setitem(sys.modules, "omnidreams.config", fake)
    monkeypatch.setattr(sys.modules["omnidreams"], "config", fake, raising=False)
    S.OmniDreamsStream(config="four-step-default", checkpoint="/x.pt", checkpoint_env="FAKE_CKPT", steps=steps,
                       device="cpu")
    (cfg,) = built
    assert cfg.diffusion_model.scheduler.num_inference_steps == steps
    assert cfg.diffusion_model.scheduler.denoising_timesteps == S.TIMESTEPS[steps]
    assert cfg.view_ids is registered.view_ids
    assert registered.diffusion_model.scheduler.num_inference_steps == 4      # FlashDreams' object untouched


def test_centre_crop_keeps_the_planners_band():
    """1280x704 -> 1024x512 takes rows 32-672 at full width (a planner's 2:1 crop of
    the camera); a frame wider than the camera keeps its centred columns."""
    from cosmos_canvas.omnidreams.rig import centre_crop

    frame = np.zeros((704, 1280, 3), np.uint8)
    frame[:32] = (255, 0, 0)
    frame[672:] = (0, 0, 255)
    frame[32:672] = (0, 255, 0)
    out = centre_crop(frame, 1024, 512)
    assert out.shape == (512, 1024, 3)
    assert (out == (0, 255, 0)).all()
    wide = np.zeros((100, 400, 3), np.uint8)
    wide[:, 150:250] = 200
    assert (centre_crop(wide, 50, 50) == 200).all()


def test_without_reprojection_a_camera_gets_the_generated_frame_itself(renderer):
    """reproject: false hands any CARLA camera the generated frame's centred band, its pose
    ignored: the camera stands in for the generated one."""
    from carla_canvas import CameraView, WorldInfo
    from carla_canvas.geometry import PinholeCamera, Transform

    from cosmos_canvas.omnidreams.rig import centre_crop

    renderer.options["reproject"] = False
    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    scene = _scene(0.0, 15.0)
    rear = PinholeCamera(512, 256, 62.0, Transform((-0.5, 0.0, 1.9), (0.0, 180.0, 0.0)))
    frame = renderer.render_camera(CameraView(camera=rear, scene=scene, outputs=frozenset({"rgb"}),
                                              sensor_id=3, frame=renderer.coordinate_frame)).rgb
    generated = renderer._rollout.frames[0]
    assert generated.max() > 0
    np.testing.assert_array_equal(frame, centre_crop(generated, 512, 256))


def test_a_raster_is_named_by_module_and_class_and_must_draw_the_cameras_lens(tmp_path):
    from cosmos_canvas.omnidreams.raster import make_raster
    from cosmos_canvas.omnidreams.rig import GeneratedCamera

    (tmp_path / "cam.json").write_text(json.dumps(CAMERA))
    cam = GeneratedCamera.from_json(tmp_path / "cam.json")
    raster = make_raster("test_omnidreams:BoxRaster", cam, {}, None)
    assert isinstance(raster, BoxRaster)
    with pytest.raises(ValueError, match="f-theta camera"):
        make_raster("cosmos_canvas.omnidreams.ludus:LudusRaster", cam, {}, None)
    with pytest.raises(ValueError, match="module:Class"):
        make_raster("nosuch", cam, {}, None)


def test_a_chunk_hands_over_every_road_user_in_range_in_any_of_its_frames(tmp_path):
    """The raster decides per frame what it draws: a road user coming into max_distance
    during the chunk (101 m -> 94 m) must be in it, one far away throughout must not."""
    from cosmos_canvas.omnidreams.renderer import OmniDreamsRenderer, _Rollout
    from cosmos_canvas.omnidreams.rig import GeneratedCamera
    from cosmos_canvas.trajectory import Track

    (tmp_path / "cam.json").write_text(json.dumps(CAMERA))
    r = OmniDreamsRenderer({"fps": 10, "max_distance": 100.0})
    r._camera = GeneratedCamera.from_json(tmp_path / "cam.json")
    rollout = _Rollout(t0=0.0, last_time=0.0, ego_id=1, ego_half_length=2.0)
    for t in (0.0, 1.0):
        rollout.ego.add(t, np.eye(4))

    def track(x0, x1):
        tr = Track()
        for t, x in ((0.0, x0), (1.0, x1)):
            pose = np.eye(4)
            pose[0, 3] = x
            tr.add(t, pose)
        return tr

    # base_link is 1 m behind the ego's origin (half its half length): +1 m
    rollout.actors = {2: (track(101.0 + 1.0, 94.0 + 1.0), np.array([4.6, 2.0, 1.5]), "car"),
                      3: (track(150.0, 150.0), np.array([4.6, 2.0, 1.5]), "car")}
    chunk = r.chunk(rollout, range(0, 8))
    assert [cls for cls, _, _ in chunk.boxes] == ["car"] and len(chunk.boxes[0][1]) == 8


def test_a_camera_on_another_vehicle_does_not_restart_the_rollout(renderer):
    """One camera is generated, for one vehicle: a camera on another vehicle gets black,
    and the ego's rollout goes on (it used to restart on every call, each vehicle's in turn)."""
    from carla_canvas import CameraView, WorldInfo
    from carla_canvas.geometry import PinholeCamera
    from carla_canvas.scene import ActorInfo, ActorState

    renderer.load_world(WorldInfo("Carla/Maps/Town10HD_Opt", 1, {}, base_dir=renderer.test_dir,
                                  opendrive="<OpenDRIVE/>"))
    for k in range(8):
        scene = _scene(0.1 * k, 20.0)
        scene.add_actor(ActorInfo(4, "sensor.camera.rgb", parent_id=2))
        scene.states[4] = ActorState(scene.states[2].transform)
        assert renderer.render_camera(_view(scene, renderer)).rgb.any()
        other = CameraView(camera=PinholeCamera(512, 256, 62.0, scene.states[4].transform), scene=scene,
                           outputs=frozenset({"rgb"}), sensor_id=4, frame=renderer.coordinate_frame)
        assert not renderer.render_camera(other).rgb.any()
    assert renderer._stream.started == 1
    assert renderer._rollout.ego_id == 1

    # the ego is gone and the other vehicle carries the only camera: the rollout is its own
    scene = _scene(0.8, 20.0)
    scene.add_actor(ActorInfo(4, "sensor.camera.rgb", parent_id=2))
    scene.states[4] = ActorState(scene.states[2].transform)
    del scene.states[1]
    other = CameraView(camera=PinholeCamera(512, 256, 62.0, scene.states[4].transform), scene=scene,
                       outputs=frozenset({"rgb"}), sensor_id=4, frame=renderer.coordinate_frame)
    renderer.render_camera(other)
    assert renderer._rollout.ego_id == 2 and renderer._stream.started == 2
