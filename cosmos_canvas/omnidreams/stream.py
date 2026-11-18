"""OmniDreams through FlashDreams, in this process, one chunk at a time.

FlashDreams' OmniDreams pipeline (``integrations_v2/omnidreams``) is driven
directly: ``initialize_cache`` with the caption and a first image, then
``generate`` / ``finalize`` per chunk with the HD-map raster of that chunk's
frames handed in from outside. That is what lets the raster follow the live
scene: FlashDreams' own server takes the moving objects once per session.

Chunks are 5 frames (the first, whose frame 0 reproduces the first image),
then 8, at the checkpoint's frame rate (``FPS``: NVIDIA's public checkpoint is
trained at 30 Hz). FlashDreams is not a dependency of this package: it is the
environment the renderer runs in (its container image), with the configuration
named by ``config`` (``omnidreams``: NVIDIA's public checkpoint). A
configuration that reads its weights from an environment variable gets them
from ``checkpoint`` through ``checkpoint_env``.
"""

from __future__ import annotations

import os
import time

import numpy as np

FPS = 30.0                # the default frame rate (Hz): NVIDIA's public checkpoint's
FIRST_CHUNK = 5
CHUNK = 8
TIMESTEPS = {2: [1000, 500], 4: [1000, 750, 500, 250]}


def chunk_sizes():
    """5, 8, 8, ..."""
    yield FIRST_CHUNK
    while True:
        yield CHUNK


def chunk_of_frame(frame: int) -> int:
    return 0 if frame < FIRST_CHUNK else 1 + (frame - FIRST_CHUNK) // CHUNK


def chunk_frames(k: int) -> range:
    if k == 0:
        return range(FIRST_CHUNK)
    start = FIRST_CHUNK + (k - 1) * CHUNK
    return range(start, start + CHUNK)


def with_schedule(cfg, steps: int):
    """FlashDreams configuration *cfg* denoising in *steps* steps, *cfg* itself untouched.

    Always set: configurations differ in their default (two or four steps). Copied
    shallowly down to the scheduler: FlashDreams' ``derive_config`` deep-copies the whole
    configuration, which fails on configurations holding mapping proxies (multi-view ones).
    """
    import copy

    scheduler = cfg.diffusion_model.scheduler
    for name in ("num_inference_steps", "denoising_timesteps"):
        if not hasattr(scheduler, name):
            raise AttributeError(f"{type(scheduler).__name__} has no {name}: not an OmniDreams scheduler")
    cfg = copy.copy(cfg)
    cfg.diffusion_model = copy.copy(cfg.diffusion_model)
    cfg.diffusion_model.scheduler = copy.copy(scheduler)
    cfg.diffusion_model.scheduler.num_inference_steps = steps
    cfg.diffusion_model.scheduler.denoising_timesteps = list(TIMESTEPS[steps])
    return cfg


class OmniDreamsStream:
    """One rollout at a time; the pipeline (weights) is kept across rollouts."""

    def __init__(self, config: str = "omnidreams", checkpoint: str | None = None,
                 checkpoint_env: str = "", steps: int = 2, device: str = "cuda",
                 view: str = "camera_front_wide_120fov") -> None:
        if checkpoint:
            if not checkpoint_env:
                raise ValueError(f"checkpoint {checkpoint}: name the environment variable {config!r} reads its "
                                 "weights from (checkpoint_env)")
            # read by FlashDreams when omnidreams.config is first imported
            os.environ[checkpoint_env] = str(checkpoint)
        if checkpoint_env and not os.environ.get(checkpoint_env):
            raise RuntimeError(f"{checkpoint_env} is unset (renderer.options.checkpoint): FlashDreams "
                               f"would load other weights for {config!r}")
        if steps not in TIMESTEPS:
            raise ValueError(f"steps {steps}: one of {sorted(TIMESTEPS)}")
        from omnidreams import config as C

        if config not in C.OMNIDREAMS_CONFIGS:
            raise ValueError(f"FlashDreams has no OmniDreams configuration {config!r} "
                             f"(it has {sorted(C.OMNIDREAMS_CONFIGS)})")
        cfg = with_schedule(C.OMNIDREAMS_CONFIGS[config], steps)
        t0 = time.perf_counter()
        self.pipe = cfg.setup().to(device=device).eval()
        self.load_seconds = time.perf_counter() - t0
        self.device = device
        self.view = view
        self.cache = None
        self.index = 0

    def _to_model(self, x: np.ndarray):
        import torch

        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        return t.permute(*range(t.ndim - 3), t.ndim - 1, t.ndim - 3, t.ndim - 2).to(torch.bfloat16) / 127.5 - 1.0

    def start(self, caption: str, first_image: np.ndarray, seed: int = 0) -> None:
        """A new rollout from *first_image* (H, W, 3) uint8, the generated size."""
        import torch

        self.pipe.diffusion_model.rng.manual_seed(int(seed))
        image = self._to_model(first_image)[None, None, None]          # [1,1,1,3,H,W]
        with torch.no_grad():
            self.cache = self.pipe.initialize_cache(text=[[caption]], image=image, view_names=[self.view])
        self.index = 0

    @property
    def next_chunk_frames(self) -> range:
        return chunk_frames(self.index)

    def step(self, rasters: np.ndarray) -> tuple[np.ndarray, float]:
        """The next chunk from its rasters (n, H, W, 3) uint8: frames (n, H, W, 3) uint8, seconds."""
        import torch

        if self.cache is None:
            raise RuntimeError("start() a rollout first")
        n = len(self.next_chunk_frames)
        if len(rasters) != n:
            raise ValueError(f"chunk {self.index} has {n} frames, got {len(rasters)} rasters")
        cond = self._to_model(rasters)[None, None]                        # [1,1,T,3,H,W]
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            rgb = self.pipe.generate(self.index, self.cache, input=cond)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            self.pipe.finalize(self.index, self.cache)
        rgb = ((rgb.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        self.index += 1
        return rgb[0, 0].permute(0, 2, 3, 1).cpu().numpy(), seconds
