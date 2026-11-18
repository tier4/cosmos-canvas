"""Drive the OmniDreams backend through carla_canvas' renderer interface from a recorded run, without CARLA.

A run's road users are replayed tick by tick as carla_canvas snapshots (a
``carla_canvas.Scene``), a CARLA camera rides the ego, and every tick asks the
renderer for that camera's image, exactly as the carla_canvas service would in
a live run. Nothing reacts to the images (open loop), so the output can be set
against earlier open-loop rollouts of the same run, and the renderer's timing
is measured as a scenario would see it.

The run is a tracks document (carla_transfer's ``log_to_tracks.py`` /
``drive --record-tracks``): per actor, ``[t, x, y, z, roll, pitch, yaw]``
samples in CARLA's frame (degrees), its blueprint, box size and offset; the ego
is role ``hero``; role ``parked`` tracks are left out (the renderer takes those
from ``map_extras``).

Run where the renderer runs (FlashDreams' environment, carla-canvas, roadgen):

    python scripts/replay_tracks.py --tracks run.tracks.json --xodr map.xodr \
        --config omnidreams.yaml --out <dir> [--start 0.15] [--seconds 9.3] \
        [--camera 1280 720 90 1.5 0 1.5 0 0 0]

``--camera``: width, height, horizontal fov, then the mount on the ego in
CARLA's frame: x, y, z (m), pitch, yaw, roll (deg). The default is a 90 degree front camera.
Writes ``camera.mp4`` (the CARLA camera at the generated frames' rate, or at the
recording's if that is lower), ``timings.json``, and the renderer's own record
(``record_dir``) if the configuration sets one.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_canvas import CameraView, WorldInfo  # noqa: E402
from carla_canvas.geometry import PinholeCamera, Transform, transform_from_matrix  # noqa: E402
from carla_canvas.scene import ActorInfo, ActorState, BoundingBox, Scene  # noqa: E402

FRAME_HZ = 10.0           # the generated frames' rate when the renderer does not say (its ``fps``)
CAMERA_ID = 10_000_000
FRONT_CAMERA = [1280, 720, 90.0, 1.5, 0.0, 1.5, 0.0, 0.0, 0.0]


def load_tracks(path: Path) -> tuple[dict[int, ActorInfo], dict[int, dict[int, Transform]], int, float]:
    """Actors, their poses by sample index (time x rate), the ego's id and the recording's rate (Hz)."""
    doc = json.loads(path.read_text())
    fps = float(doc.get("fps", 20.0))
    infos, samples, ego = {}, {}, 0
    for tr in doc["tracks"]:
        if tr.get("role_name") == "parked":
            continue
        aid = int(tr["actor_id"])
        lwh = np.asarray(tr["size_lwh"], np.float64)
        infos[aid] = ActorInfo(id=aid, type_id=tr["blueprint"], attributes={"role_name": tr.get("role_name", "")},
                               bounding_box=BoundingBox(tuple(tr.get("bbox_offset", (0.0, 0.0, 0.0))),
                                                        tuple(0.5 * lwh), (0.0, 0.0, 0.0)))
        samples[aid] = {round(s[0] * fps): Transform((s[1], s[2], s[3]), (s[5], s[6], s[4])) for s in tr["samples"]}
        if tr.get("role_name") == "hero":
            ego = aid
    if not ego:
        raise SystemExit(f"{path}: no hero track")
    return infos, samples, ego, fps


def replay(renderer, infos, samples, ego: int, fps: float, *, map_name: str, start: float, seconds: float,
           camera, write, frame_hz: float = FRAME_HZ) -> list[dict]:
    """Ask *renderer* for the camera's image on every tick of the recording (its own rate
    *fps*) from the sample nearest *start* for *seconds*; hand *write* one image per frame
    time at *frame_hz*, from the tick nearest to it. Stops early where the ego's track ends."""
    tick = 1.0 / fps
    w, h, fov = int(camera[0]), int(camera[1]), float(camera[2])
    mount = Transform(tuple(camera[3:6]), tuple(camera[6:9])).matrix()
    scene = Scene()
    scene.reset(map_name, 1)
    for info in infos.values():
        scene.add_actor(info)
    scene.add_actor(ActorInfo(id=CAMERA_ID, type_id="sensor.camera.rgb", parent_id=ego,
                              attributes={"role_name": "front", "image_size_x": str(w),
                                          "image_size_y": str(h), "fov": str(fov)}))
    # each frame time is written from the tick nearest to it (one tick can carry several
    # frames when the recording is slower than frame_hz)
    frames_at = Counter(round(k * fps / frame_hz) for k in range(round(seconds * frame_hz) + 1))
    first = round(start * fps)                    # start on a recorded sample
    calls = []
    for i in range(round(seconds * fps) + 1):
        n = first + i
        t = n / fps
        scene.frame, scene.elapsed_seconds, scene.delta_seconds = i, t, tick
        scene.states = {aid: ActorState(s[n]) for aid, s in samples.items() if n in s}
        if ego not in scene.states:
            print(f"t={t:.3f}: the ego is not in the run")
            break
        cam_pose = transform_from_matrix(scene.states[ego].transform.matrix() @ mount)
        scene.states[CAMERA_ID] = ActorState(cam_pose)
        view = CameraView(camera=PinholeCamera(w, h, fov, cam_pose), scene=scene, outputs=frozenset({"rgb"}),
                          sensor_id=CAMERA_ID, frame=renderer.coordinate_frame)
        c0 = time.perf_counter()
        frame = renderer.render_camera(view)
        calls.append({"t": round(t, 6), "seconds": round(time.perf_counter() - c0, 4)})
        for _ in range(frames_at[i]):
            write(frame.rgb)
    return calls


def video_rate(renderer, fps: float) -> float:
    """The camera video's rate: the renderer's frames (its ``fps``), or every tick of a
    recording slower than that."""
    return min(float(getattr(renderer, "fps", FRAME_HZ)), fps)


class Mp4:
    def __init__(self, path: Path, w: int, h: int, fps: float = 10.0) -> None:
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16",
                                   "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", required=True, type=Path)
    ap.add_argument("--xodr", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path, help="carla_canvas configuration (its renderer section)")
    ap.add_argument("--map-name", default="Carla/Maps/Town10HD_Opt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start", type=float, default=0.15, help="seconds after the run's first tick (the nearest sample)")
    ap.add_argument("--seconds", type=float, default=9.3)
    ap.add_argument("--camera", type=float, nargs=9, default=FRONT_CAMERA)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    section = yaml.safe_load(a.config.read_text())["renderer"]
    backend = section["backend"]
    module, cls = backend.split(":")
    renderer = getattr(__import__(module, fromlist=[cls]), cls)(section.get("options", {}))
    world = WorldInfo(map_name=a.map_name, episode_id=1, config=section, base_dir=a.config.parent,
                      opendrive=a.xodr.read_text())
    t0 = time.perf_counter()
    renderer.load_world(world)
    load_seconds = time.perf_counter() - t0

    infos, samples, ego, fps = load_tracks(a.tracks)
    w, h = int(a.camera[0]), int(a.camera[1])
    a.out.mkdir(parents=True, exist_ok=True)
    frame_hz = video_rate(renderer, fps)
    video = Mp4(a.out / "camera.mp4", w, h, frame_hz)
    calls = replay(renderer, infos, samples, ego, fps, map_name=a.map_name, start=a.start, seconds=a.seconds,
                   camera=a.camera, write=video.write, frame_hz=frame_hz)
    video.close()
    if not calls:
        raise SystemExit(f"{a.tracks}: the ego has no sample near --start {a.start}; nothing was replayed")
    per_call = np.array([c["seconds"] for c in calls])
    summary = {"load_world_s": round(load_seconds, 1), "ticks": len(calls),
               "tick_s_median": float(np.median(per_call)), "tick_s_max": float(per_call.max()),
               "chunks": getattr(renderer, "timings", []), "calls": calls}
    (a.out / "timings.json").write_text(json.dumps(summary, indent=1))
    gen = [c["generate"] for c in summary["chunks"][1:]] or [0.0]
    ras = [c["raster"] for c in summary["chunks"][1:]] or [0.0]
    print(json.dumps({"out": str(a.out), "load_world_s": summary["load_world_s"], "ticks": len(calls),
                      "chunks": len(summary["chunks"]), "generate_s_median": round(float(np.median(gen)), 3),
                      "raster_s_median": round(float(np.median(ras)), 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
