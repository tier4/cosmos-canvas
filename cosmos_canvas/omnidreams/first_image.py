"""The first image of an OmniDreams rollout, made by NVIDIA's public Cosmos3-Nano-Transfer-Auto.

OmniDreams continues an image of the scene's start, which CARLA does not have in the
generated camera. Transfer-Auto (https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)
makes one from the scene alone: text-to-video from a WSM v4 control video and a caption,
of which the first frame is kept. Three steps, each in the environment it needs:

1. ``WsmCapture``, a carla_canvas renderer (``renderer.backend:
   cosmos_canvas.omnidreams.first_image:WsmCapture``), run in a scenario or through
   ``scripts/replay_tracks.py``: the scene's first ``frames`` frames at 30 Hz drawn as
   Transfer-Auto's WSM v4 control for ``camera_front_wide_120fov``, with the Sim-Auto
   backend's per-frame WSM renderer (``cosmos_canvas.sim_auto.controls``), from the
   generated camera's mount (the OmniDreams ``camera`` file) on the ego's ``base_link``.
   It writes to ``out_dir``: ``wsm.mp4``, the caption, ``request.jsonl`` (cosmos-framework's
   inference request) and ``lens.json`` (the lens Transfer-Auto generates the view in).
   Needs the cookbook's ``cosmos_auto`` (cosmos-framework) and a GPU for its renderer;
   CARLA cameras get black.
2. cosmos-framework's inference on ``request.jsonl`` with the Transfer-Auto snapshot.
3. ``python -m cosmos_canvas.omnidreams.first_image pick <generated.mp4> <out_dir>/lens.json
   <camera.json> <first_image.png>``: the generated video's first frame, turned from
   Transfer-Auto's lens (the Hyperion ``camera_front_wide_120fov``, 1280x720) into the
   OmniDreams camera's, from the same point. Needs numpy and OpenCV only.

Options (``renderer.options``)::

    camera: /path/to/camera_front_wide_120fov.json   # the OmniDreams camera (its mount is used)
    out_dir: /path/to/first_image                    # what the capture writes
    caption: "A forward-facing dashcam view ..."
    frames: 93                                       # 30 Hz frames of control (1 + 4k)
    map_extras: /path/to/Town10HD_Opt.extras.json    # optional: parked vehicles
    seed: 0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from carla_canvas import CameraFrame, CameraView, Renderer, WorldInfo
from carla_canvas.geometry import Transform, to_frame

from ..trajectory import Track
from . import mapping
from .raster import road_user_class
from .rig import FTheta, GeneratedCamera, base_link, resample

log = logging.getLogger("cosmos_canvas.omnidreams.first_image")

VIEW = "camera_front_wide_120fov"
FPS = 30.0              # Transfer-Auto's frame rate
FRAMES = 93
WSM_TYPE = {"car": "Car", "truck": "Truck", "bus": "Truck", "motorcycle": "Cyclist", "bicycle": "Cyclist",
            "pedestrian": "Pedestrian"}
STATIC_TYPE = {"car": "Car", "truck": "Truck", "bus": "Truck", "motorcycle": "Cyclist", "bicycle": "Cyclist"}
#: how far past its first or last sample an actor still has a box (s)
PRESENCE_SLACK = 0.06


@dataclass
class _Capture:
    t0: float
    ego_id: int
    ego_half_length: float
    ego: Track = field(default_factory=lambda: Track(maxlen=100_000))
    actors: dict[int, tuple[Track, np.ndarray, str]] = field(default_factory=dict)   # box track, lwh, WSM type
    written: bool = False


class WsmCapture(Renderer):
    """Records Transfer-Auto's WSM v4 control of the scene's first frames (module doc)."""

    name = "omnidreams-first-image"
    version = "0.1.0"
    camera_outputs = frozenset({"rgb"})
    coordinate_frame = "right_handed"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        o = self.options
        self.frames = int(o.get("frames", FRAMES))
        if self.frames < 1 or (self.frames - 1) % 4:
            raise ValueError(f"frames {self.frames}: 1 + 4k (the video tokenizer's)")
        self.caption = str(o.get("caption", ""))
        if not self.caption:
            raise ValueError("caption: Transfer-Auto needs the scene's caption")
        self.seed = int(o.get("seed", 0))
        self.rear_axle = o.get("rear_axle_m")
        self.actor_patterns = list(o.get("actors", ["vehicle.*", "walker.pedestrian.*"]))
        self._camera: GeneratedCamera | None = None
        self._out: Path | None = None
        self._wsm = None
        self._static: list = []
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._capture: _Capture | None = None
        self._lock = threading.RLock()

    # -- carla_canvas -----------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        with self._lock:
            o = self.options
            self._camera = GeneratedCamera.from_json(world.resolve(o["camera"]))
            self._out = world.resolve(o["out_dir"])
            self._out.mkdir(parents=True, exist_ok=True)
            if not world.opendrive:
                raise RuntimeError("the capture needs the map's OpenDRIVE (carla_canvas protocol 2)")
            self._wsm = self._wsm_camera(world.opendrive)
            self._static = []
            if o.get("map_extras"):
                _, parked = mapping.extra_boxes(world.resolve(o["map_extras"]))
                self._static = parked
            self._capture = None

    def render_camera(self, view: CameraView) -> CameraFrame:
        with self._lock:
            cam = view.camera
            black = CameraFrame(rgb=np.zeros((cam.height, cam.width, 3), np.uint8))
            scene = view.scene
            ego_id = scene.actors[view.sensor_id].parent_id if view.sensor_id in scene.actors else 0
            if not ego_id or ego_id not in scene.states:
                return black
            t = float(scene.elapsed_seconds)
            c = self._capture
            if c is None:
                c = self._capture = _Capture(t0=t, ego_id=ego_id,
                                             ego_half_length=float(scene.actors[ego_id].bounding_box.extent[0]))
            if c.written:
                return black
            self._track(c, scene, t)
            if t >= c.t0 + (self.frames - 1) / FPS - 1e-6:
                self.write(c)
            return black

    def close(self) -> None:
        with self._lock:
            c = self._capture
            if c is not None and not c.written and c.ego.samples:
                log.warning("the run ended %.2f s in, before the control's %.2f s: the rest is extrapolated",
                            c.ego.latest_time - c.t0, (self.frames - 1) / FPS)
                self.write(c)
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    # -- the scene ------------------------------------------------------------------

    def _wsm_camera(self, xodr: str):
        import roadgen

        from ..sim_auto.controls import WsmCamera

        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix="cosmos_canvas_first_image_")
        path = Path(self._tmp.name) / "map.xodr"
        path.write_text(xodr)
        root = Path(self._tmp.name) / "rds_hq"
        clip = roadgen.read_opendrive(str(path), sampling=0.5).export_rdshq(str(root), trace=False)
        return WsmCamera(root, clip, VIEW)

    def _track(self, c: _Capture, scene, t: float) -> None:
        frame = self.coordinate_frame
        c.ego.add(t, scene.transform(c.ego_id).matrix(frame))
        for info, transform in scene.bodies(exclude={c.ego_id}):
            if info.parent_id == c.ego_id or not any(info.matches(p) for p in self.actor_patterns):
                continue
            b = info.bounding_box
            box_to_world = to_frame(transform.matrix() @ Transform(b.location, b.rotation).matrix(), frame)
            entry = c.actors.get(info.id)
            if entry is None:
                entry = c.actors[info.id] = (Track(maxlen=100_000), 2 * np.asarray(b.extent, np.float64),
                                             WSM_TYPE.get(road_user_class(info), "Car"))
            entry[0].add(t, box_to_world)

    def boxes(self, c: _Capture, t: float) -> list:
        """The WSM boxes at time *t*: the actors seen around *t* and the parked vehicles."""
        from ..sim_auto.controls import Box

        out = []
        for actor_id, (track, lwh, kind) in c.actors.items():
            first, last = track.samples[0][0], track.samples[-1][0]
            if first - PRESENCE_SLACK <= t <= last + PRESENCE_SLACK:
                out.append(Box(str(actor_id), kind, lwh, track.at(t)))
        for i, b in enumerate(self._static):
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = b.rotation, b.centre
            out.append(Box(f"parked_{i}", STATIC_TYPE.get(b.cls, "Car"), np.asarray(b.size_lwh), pose))
        return out

    def camera_pose(self, c: _Capture, t: float) -> np.ndarray:
        rear = float(self.rear_axle) if self.rear_axle is not None else -0.5 * c.ego_half_length
        return self._camera.pose_cv(base_link(c.ego.at(t), rear))

    # -- output -------------------------------------------------------------------------

    def write(self, c: _Capture) -> None:
        times = [c.t0 + k / FPS for k in range(self.frames)]
        frames = [self._wsm.render(self.camera_pose(c, t), self.boxes(c, t)) for t in times]
        write_mp4(self._out / "wsm.mp4", frames, FPS)
        cap = self._out / "captions" / f"{VIEW}.json"
        cap.parent.mkdir(parents=True, exist_ok=True)
        cap.write_text(json.dumps({"caption_structured": {f"chunk_0_{self.frames}": {
            "caption": self.caption, "start_frame": 0, "end_frame": self.frames}}}, indent=1))
        (self._out / "request.jsonl").write_text(json.dumps(request(
            self._out.name, self._out / "wsm.mp4", cap, self.frames, self.seed)) + "\n")
        (self._out / "lens.json").write_text(json.dumps(transfer_lens(), indent=1))
        c.written = True
        log.info("WSM control of %.2f-%.2f s (%d frames) written to %s", times[0], times[-1], len(frames), self._out)


def request(name: str, control: Path, caption: Path, frames: int, seed: int) -> dict:
    """cosmos-framework's inference request: one view, text-to-video from its WSM control
    (the released model's reference settings: 720p, guidance 3, shift 10, 35 steps)."""
    return {
        "name": name, "model_mode": "text2video", "resolution": "720", "aspect_ratio": "16,9",
        "fps": int(FPS), "num_frames": frames, "num_video_frames_per_chunk": frames, "num_outputs": 1,
        "seed": seed, "guidance": 3, "control_guidance": 1, "normalize_cfg": False, "autoregressive": False,
        "wsm": {}, "num_conditional_frames": 0, "prompt": "",
        "multiview": {"load_caption_from_data": True, "use_first_chunk_only": True,
                      "views": [{"camera_key": VIEW, "control_path": str(control), "caption_path": str(caption)}]},
        "num_steps": 35, "shift": 10,
    }


def transfer_lens(size: tuple[int, int] = (720, 1280)) -> dict:
    """The lens Transfer-Auto generates ``camera_front_wide_120fov`` in at *size* (H, W): the
    cookbook's Hyperion intrinsics the WSM is drawn with, as an ``ftheta`` lens (rig.py)."""
    from ..sim_auto.camera import lens

    g = lens(VIEW, {(720, 1280): "720", (480, 832): "480"}[tuple(size)])
    return {"width": g.width, "height": g.height,
            "ftheta": {"principal_point": list(g.principal_point), "fw_poly": list(g.angle_to_pixeldist_poly),
                       "max_ray_angle": g.max_angle, "linear_cde": list(g.linear_cde)}}


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12",
                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def pick(video: Path, lens_json: Path, camera_json: Path, out: Path, frame: int = 0) -> float:
    """The generated video's *frame*, as the OmniDreams camera sees it; returns the share of
    the camera's pixels outside Transfer-Auto's view (black)."""
    import cv2

    lens_doc = json.loads(Path(lens_json).read_text())
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"{video}: no frame {frame}")
    if (bgr.shape[1], bgr.shape[0]) != (lens_doc["width"], lens_doc["height"]):
        raise ValueError(f"{video} is {bgr.shape[1]}x{bgr.shape[0]}; the lens is for "
                         f"{lens_doc['width']}x{lens_doc['height']}")
    image, outside = resample(bgr, FTheta.from_dict(lens_doc["ftheta"]), GeneratedCamera.from_json(camera_json))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)
    return outside


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="the generated video's first frame in the OmniDreams camera")
    p.add_argument("video", type=Path)
    p.add_argument("lens", type=Path, help="the capture's lens.json")
    p.add_argument("camera", type=Path, help="the OmniDreams camera file")
    p.add_argument("out", type=Path)
    p.add_argument("--frame", type=int, default=0)
    a = ap.parse_args(argv)
    outside = pick(a.video, a.lens, a.camera, a.out, a.frame)
    print(json.dumps({"out": str(a.out), "outside_pct": round(100 * outside, 2)}))
    if outside > 0.01:
        log.warning("%.1f %% of the camera's pixels are outside Transfer-Auto's view (black)", 100 * outside)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

