"""OmniDreams (Cosmos 2.5, distilled) as a carla_canvas renderer: one camera, from the live scene.

carla_canvas (https://github.com/hakuturu583/carla_canvas) swaps CARLA's sensor
rendering behind the CARLA API. This backend answers the ego's RGB cameras with
OmniDreams run in-process through FlashDreams (``stream.py``):

* LoadWorld: the map's OpenDRIVE is drawn in the HD-map raster the checkpoint
  was trained on (``raster.py``): by default FlashDreams' ludus, the raster of
  NVIDIA's public checkpoint (``ludus.py``), or a raster named by
  ``module:Class`` for a fine-tune trained on another one;
* the model generates the camera of its rig (``rig.py``) at ``fps`` (NVIDIA's
  public checkpoint: 30 Hz) in chunks of 5, then 8 frames. A chunk is generated
  when a camera tick needs one of its frames: its rasters are drawn from the
  ego's ``base_link`` and every road user's box at the frames' times, the later
  ones extrapolated from the last ticks (``trajectory.py``), so the road users
  the scenario moves are in every chunk;
* a CARLA camera on the ego reads the frame nearest its tick's time, turned
  into its own view, rotation only (any pinhole within the generated view).
  With ``reproject: false`` the CARLA camera stands in for the generated one
  instead: it gets the generated frame's centred band at its aspect, resized,
  whatever its pose and field of view (for a planner trained on the generated
  camera itself).

The first chunk continues an image (``first_image``, the generated size): a
real or generated frame of the scene's start (``first_image.py``).

Configuration (``renderer`` section of the carla_canvas file)::

    renderer:
      backend: cosmos_canvas.omnidreams:OmniDreamsRenderer
      startup_timeout: 600     # LoadWorld: the model loads
      timeout: 120             # a tick that generates a chunk
      options:
        config: omnidreams                     # FlashDreams' OmniDreams configuration
        checkpoint: /path/to/weights.pt        # optional: written to checkpoint_env for the configuration
        checkpoint_env: ""                     # the variable the configuration reads its weights from
        fps: 30                                # the checkpoint's frame rate
        raster: ludus                          # or package.module:Class
        camera: /path/to/camera_front_wide_120fov.json   # the generated camera (rig.py)
        first_image: /path/to/first_frame.png
        caption: "A forward-facing dashcam view ..."
        map_extras: /path/to/Town10HD_Opt.extras.json    # optional
        steps: 2
        seed: 0
        reproject: true                        # false: cameras get the generated frame itself

Traffic-light states are not drawn: carla_canvas does not send them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from carla_canvas import CameraFrame, CameraView, Renderer, WorldInfo
from carla_canvas.geometry import Transform, to_frame

from ..trajectory import Track
from .raster import Chunk, make_raster, road_user_class
from .rig import GeneratedCamera, base_link, centre_crop, look_through
from .stream import FPS, chunk_of_frame

log = logging.getLogger("cosmos_canvas.omnidreams")

DEFAULT_CAPTION = ("A forward-facing dashcam view of an urban drive in clear daytime, "
                   "on a dry road with clear lane markings.")


@dataclass
class _Rollout:
    """One episode's generation."""

    t0: float
    last_time: float
    ego_id: int
    ego_half_length: float = 0.0
    ego: Track = field(default_factory=lambda: Track(maxlen=4))
    actors: dict[int, tuple[Track, np.ndarray, str]] = field(default_factory=dict)  # box track, lwh, class
    frames: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> generated (H, W, 3)
    drawn_at: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> camera pose it was drawn at
    started: bool = False


class OmniDreamsRenderer(Renderer):
    name = "omnidreams"
    version = "0.2.0"
    camera_outputs = frozenset({"rgb"})
    coordinate_frame = "right_handed"  # OpenDRIVE's and roadgen's: x forward, y left, z up

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        o = self.options
        self.config = str(o.get("config", "omnidreams"))
        self.caption = str(o.get("caption", DEFAULT_CAPTION))
        self.seed = int(o.get("seed", 0))
        self.steps = int(o.get("steps", 2))
        self.fps = float(o.get("fps", FPS))
        if not self.fps > 0:
            raise ValueError(f"fps {self.fps}: the checkpoint's frame rate, positive")
        self.actor_patterns = list(o.get("actors", ["vehicle.*", "walker.pedestrian.*"]))
        self.max_distance = float(o.get("max_distance", 100.0))
        self.reset_gap = float(o.get("reset_after_seconds", 1.0))
        self.rear_axle = o.get("rear_axle_m")  # None: half the ego's half length behind its origin
        self.raster_spec = str(o.get("raster", "ludus"))
        self.reproject = True
        self._camera: GeneratedCamera | None = None
        self._camera_key: str | None = None
        self._raster = None
        self._first_image: np.ndarray | None = None
        self._map_key: str | None = None
        self._stream = None
        self._rollout: _Rollout | None = None
        self._refused: set[int] = set()   # cameras on another vehicle, warned about once
        self._record: Path | None = None
        self._lock = threading.RLock()
        self.timings: list[dict[str, float]] = []

    # -- carla_canvas -----------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        with self._lock:
            self._rollout = None
            self._refused.clear()
            o = self.options
            self.reproject = bool(o.get("reproject", True))
            camera_path = world.resolve(o["camera"])
            camera = GeneratedCamera.from_json(camera_path)
            camera_key = hashlib.sha256(Path(camera_path).read_bytes()).hexdigest()
            if camera_key != self._camera_key or self._raster is None:    # a raster holds its camera
                self._raster = self._make_raster(camera, world)
                self._map_key = None
                self._camera_key = camera_key
            self._camera = camera
            self._first_image = self._load_first_image(world.resolve(o["first_image"]))
            if o.get("record_dir"):
                self._record = world.resolve(o["record_dir"])
                self._record.mkdir(parents=True, exist_ok=True)
            if not world.opendrive:
                raise RuntimeError("omnidreams needs the map's OpenDRIVE (carla_canvas protocol 2)")
            key = hashlib.sha256((world.opendrive + json.dumps(o, sort_keys=True, default=str)).encode()).hexdigest()
            if key != self._map_key:
                counts = self._raster.load_map(world)
                log.info("map %s (%s): %s", world.map_basename, self.raster_spec, counts)
                self._map_key = key
            if self._stream is None:
                self._stream = self._load_stream(world)

    def render_camera(self, view: CameraView) -> CameraFrame:
        with self._lock:
            cam = view.camera
            rollout = self._advance(view.scene, view.sensor_id)
            if rollout is None:
                return CameraFrame(rgb=np.zeros((cam.height, cam.width, 3), np.uint8))
            frame_index = max(0, round((view.scene.elapsed_seconds - rollout.t0) * self.fps))
            self._ensure(rollout, chunk_of_frame(frame_index))
            generated = rollout.frames.get(frame_index)
            if generated is None:
                return CameraFrame(rgb=np.zeros((cam.height, cam.width, 3), np.uint8))
            if not self.reproject:
                return CameraFrame(rgb=centre_crop(generated, cam.width, cam.height))
            # the generated frame was drawn for the camera's pose at its time as it was
            # extrapolated then (a chunk's later frames); turn it into where the camera is
            drawn = rollout.drawn_at[frame_index]
            actual = np.linalg.inv(view.world_to_camera_cv())
            return CameraFrame(rgb=look_through(generated, self._camera, drawn, cam.intrinsics(), actual,
                                                cam.width, cam.height))

    def close(self) -> None:
        with self._lock:
            self._rollout = None
            self._stream = None

    # -- setup (overridable for tests) ----------------------------------------------

    def _make_raster(self, camera: GeneratedCamera, world: WorldInfo):
        return make_raster(self.raster_spec, camera, self.options, world)

    def _load_first_image(self, path: Path) -> np.ndarray:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        size = (self._camera.width, self._camera.height)
        if im.size != size:
            log.info("first image %s: %s -> %s", path, im.size, size)
            im = im.resize(size, Image.BICUBIC)
        return np.asarray(im)

    def _load_stream(self, world: WorldInfo):
        from .stream import OmniDreamsStream

        o = self.options
        checkpoint = str(world.resolve(o["checkpoint"])) if o.get("checkpoint") else None
        log.info("loading OmniDreams %s (%s)", self.config, checkpoint or "the configuration's checkpoint")
        stream = OmniDreamsStream(self.config, checkpoint, checkpoint_env=str(o.get("checkpoint_env", "")),
                                  steps=self.steps, view=str(o.get("view", "camera_front_wide_120fov")))
        log.info("OmniDreams loaded in %.1f s", stream.load_seconds)
        return stream

    # -- the episode's tracks -----------------------------------------------------

    def _advance(self, scene, sensor_id: int) -> _Rollout | None:
        if self._stream is None:
            raise RuntimeError("load_world has not been called")
        t = float(scene.elapsed_seconds)
        ego_id = scene.actors[sensor_id].parent_id if sensor_id in scene.actors else 0
        if not ego_id or ego_id not in scene.states:
            log.warning("camera %d is not on a vehicle: nothing to generate for it", sensor_id)
            return None
        rollout = self._rollout
        if rollout is not None and (t < rollout.last_time - 1e-6 or t - rollout.last_time > self.reset_gap
                                    or rollout.ego_id not in scene.states):
            log.info("time jumped %.3f -> %.3f s (or the ego is gone): new rollout", rollout.last_time, t)
            rollout = None
        if rollout is not None and ego_id != rollout.ego_id:
            # one camera is generated, for one vehicle: a camera on another one would
            # otherwise restart the rollout on every call, each vehicle's in turn
            if sensor_id not in self._refused:
                self._refused.add(sensor_id)
                log.warning("camera %d is on vehicle %d; the rollout is vehicle %d's: nothing is generated "
                            "for it", sensor_id, ego_id, rollout.ego_id)
            return None
        if rollout is None:
            ego = scene.actors[ego_id]
            rollout = self._rollout = _Rollout(t0=t, last_time=t, ego_id=ego_id,
                                               ego_half_length=float(ego.bounding_box.extent[0]))
        rollout.last_time = t
        self._track(rollout, scene, t)
        return rollout

    def _track(self, rollout: _Rollout, scene, t: float) -> None:
        frame = self.coordinate_frame
        rollout.ego.add(t, scene.transform(rollout.ego_id).matrix(frame))
        present = set()
        for info, transform in scene.bodies(exclude={rollout.ego_id}):
            if info.parent_id == rollout.ego_id or not any(info.matches(p) for p in self.actor_patterns):
                continue
            b = info.bounding_box
            box_to_world = to_frame(transform.matrix() @ Transform(b.location, b.rotation).matrix(), frame)
            entry = rollout.actors.get(info.id)
            if entry is None:
                entry = rollout.actors[info.id] = (Track(maxlen=4), 2 * np.asarray(b.extent, np.float64),
                                                   road_user_class(info))
            entry[0].add(t, box_to_world)
            present.add(info.id)
        for gone in set(rollout.actors) - present:
            del rollout.actors[gone]

    def _base_link(self, rollout: _Rollout, t: float) -> np.ndarray:
        rear = float(self.rear_axle) if self.rear_axle is not None else -0.5 * rollout.ego_half_length
        return base_link(rollout.ego.at(t), rear)

    # -- chunks -------------------------------------------------------------------

    def _ensure(self, rollout: _Rollout, chunk: int) -> None:
        if not rollout.started:
            self._stream.start(self.caption, self._first_image, self.seed)
            rollout.started = True
        while self._stream.index <= chunk:
            self._generate(rollout)

    def chunk(self, rollout: _Rollout, frames: range) -> Chunk:
        """What *frames* are drawn from: the ego, the generated camera and the road users
        within ``max_distance`` of the ego's base_link in any of the frames (a raster that
        limits what it draws decides per frame)."""
        times = [rollout.t0 + f / self.fps for f in frames]
        base_links = np.stack([self._base_link(rollout, t) for t in times])
        boxes = []
        for track, lwh, cls in rollout.actors.values():
            poses = np.stack([track.at(t) for t in times])
            if np.hypot(*(poses[:, :2, 3] - base_links[:, :2, 3]).T).min() <= self.max_distance:
                boxes.append((cls, poses, lwh))
        return Chunk(times_us=[round(f / self.fps * 1e6) for f in frames], base_links=base_links,
                     cameras=np.stack([self._camera.pose_cv(bl) for bl in base_links]), boxes=boxes)

    def _generate(self, rollout: _Rollout) -> None:
        k = self._stream.index
        frames = self._stream.next_chunk_frames
        t_start = time.perf_counter()
        chunk = self.chunk(rollout, frames)
        rasters = np.asarray(self._raster.draw(chunk), np.uint8)
        raster_seconds = time.perf_counter() - t_start
        rgb, model_seconds = self._stream.step(rasters)
        for i, f in enumerate(frames):
            rollout.frames[f] = rgb[i]
            rollout.drawn_at[f] = chunk.cameras[i]
        for old in [f for f in rollout.frames if f < frames.start - 16]:
            del rollout.frames[old]
            rollout.drawn_at.pop(old, None)
        if self._record is not None:
            self._save(frames, rasters, rgb)
        timing = {"chunk": k, "raster": raster_seconds, "generate": model_seconds}
        self.timings.append(timing)
        log.info("chunk %d (frames %d-%d, %.1f-%.1f s): raster %.2f s, generate %.2f s", k, frames.start,
                 frames.stop - 1, frames.start / self.fps, frames.stop / self.fps, raster_seconds, model_seconds)

    def _save(self, frames: range, rasters: np.ndarray, rgb: np.ndarray) -> None:
        from PIL import Image

        for i, f in enumerate(frames):
            Image.fromarray(rasters[i]).save(self._record / f"raster_{f:05d}.png")
            Image.fromarray(rgb[i]).save(self._record / f"generated_{f:05d}.jpg", quality=92)
