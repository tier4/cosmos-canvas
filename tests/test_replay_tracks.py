# SPDX-License-Identifier: Apache-2.0
"""scripts/replay_tracks.py: a recording is replayed at its own rate, and the camera's
frames are written at the generated frames' rate whatever that rate is."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_tracks.py"


def _load():
    spec = importlib.util.spec_from_file_location("replay_tracks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubRenderer:
    coordinate_frame = "right_handed"

    def __init__(self):
        self.times = []

    def render_camera(self, view):
        from carla_canvas import CameraFrame

        self.times.append(view.scene.elapsed_seconds)
        return CameraFrame(rgb=np.zeros((view.camera.height, view.camera.width, 3), np.uint8))


def _tracks(path: Path, fps: int, seconds: float) -> None:
    n = round(seconds * fps) + 1
    samples = [[round(k / fps, 4), 10.0 * k / fps, 0.0, 0.0, 0.0, 0.0, 0.0] for k in range(n)]
    path.write_text(json.dumps({"fps": fps, "frames": n, "tracks": [{
        "actor_id": 1, "blueprint": "vehicle.lincoln.mkz", "role_name": "hero", "category": "automobile",
        "size_lwh": [4.9, 2.0, 1.5], "bbox_offset": [0.0, 0.0, 0.75], "samples": samples}]}))


@pytest.mark.parametrize("fps", [10, 20, 24, 25, 30])
def test_replay_follows_the_recordings_rate(tmp_path, fps):
    mod = _load()
    _tracks(tmp_path / "run.json", fps, 1.0)
    infos, samples, ego, rate = mod.load_tracks(tmp_path / "run.json")
    stub, written = StubRenderer(), []
    calls = mod.replay(stub, infos, samples, ego, rate, map_name="Carla/Maps/Town10HD_Opt", start=0.0,
                       seconds=1.0, camera=mod.FRONT_CAMERA, write=written.append)
    assert len(calls) == fps + 1
    assert stub.times[-1] == pytest.approx(1.0)
    assert len(written) == 11                      # the 10 Hz frames over 1 s, both ends


def test_a_start_between_samples_snaps_to_the_recording(tmp_path):
    """The default --start (0.15 s) falls between a 10 Hz recording's samples: the replay
    starts at the nearest sample instead of finding no ego."""
    mod = _load()
    _tracks(tmp_path / "run.json", 10, 2.0)
    infos, samples, ego, rate = mod.load_tracks(tmp_path / "run.json")
    stub, written = StubRenderer(), []
    calls = mod.replay(stub, infos, samples, ego, rate, map_name="Carla/Maps/Town10HD_Opt", start=0.15,
                       seconds=1.0, camera=mod.FRONT_CAMERA, write=written.append)
    assert len(calls) == 11 and len(written) == 11
    assert stub.times[0] in (pytest.approx(0.1), pytest.approx(0.2))


@pytest.mark.parametrize(("model_hz", "written_frames"), [(10, 11), (30, 21)])
def test_the_video_keeps_the_generated_frames_or_every_tick(tmp_path, model_hz, written_frames):
    """A 20 Hz recording: a 10 Hz checkpoint's frames are written once each; with a 30 Hz
    one (NVIDIA's public checkpoint) every tick is."""
    mod = _load()
    _tracks(tmp_path / "run.json", 20, 1.0)
    infos, samples, ego, rate = mod.load_tracks(tmp_path / "run.json")
    stub, written = StubRenderer(), []
    stub.fps = float(model_hz)
    frame_hz = mod.video_rate(stub, rate)
    mod.replay(stub, infos, samples, ego, rate, map_name="Carla/Maps/Town10HD_Opt", start=0.0, seconds=1.0,
               camera=mod.FRONT_CAMERA, write=written.append, frame_hz=frame_hz)
    assert frame_hz == min(model_hz, 20)
    assert len(written) == written_frames
