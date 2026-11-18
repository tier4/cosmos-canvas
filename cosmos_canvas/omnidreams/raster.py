"""The HD-map raster a checkpoint is conditioned on: how the renderer draws it.

A checkpoint takes the raster it was trained on, so the raster is named in the
renderer's options with the checkpoint:

* ``raster: ludus`` (the default): FlashDreams' ludus renderer, the raster of
  NVIDIA's public checkpoint (``ludus.py``);
* ``raster: package.module:Class``: another one, e.g. a fine-tune's own, from
  outside this package. The class is built with the generated camera, the
  renderer's options and the world, and provides::

      lens: str                                    # the camera lens it draws: "pinhole" or "ftheta"
      def load_map(self, world) -> dict[str, int]  # the map (world.opendrive); counts for the log
      def draw(self, chunk: Chunk) -> np.ndarray   # (n, H, W, 3) uint8, one raster per frame

  ``load_map`` is called again only when the map or the options change.

Road users come as boxes with a class (``road_user_class``: car, truck, bus,
motorcycle, bicycle, pedestrian); what the map extras add (signal heads, parked
vehicles; ``mapping.extra_boxes``) is each raster's to read from the options.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

#: CARLA's ``base_type`` attribute -> road-user class
CLASS_BY_BASE_TYPE = {"car": "car", "van": "car", "truck": "truck", "bus": "bus",
                      "motorcycle": "motorcycle", "bicycle": "bicycle"}
#: blueprints whose base_type is missing (CARLA 0.10 leaves some empty)
CLASS_BY_BLUEPRINT = {
    "bicycle": ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"),
    "motorcycle": ("vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja",
                   "vehicle.vespa.zx125", "vehicle.yamaha.yzf"),
    "bus": ("vehicle.mitsubishi.fusorosa",),
    "truck": ("vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv",
              "vehicle.carlamotors.firetruck"),
}


def road_user_class(info) -> str:
    """A CARLA actor's class: car, truck, bus, motorcycle, bicycle or pedestrian."""
    if info.type_id.startswith("walker."):
        return "pedestrian"
    base = info.attributes.get("base_type", "").lower()
    if base in CLASS_BY_BASE_TYPE:
        return CLASS_BY_BASE_TYPE[base]
    for name, blueprints in CLASS_BY_BLUEPRINT.items():
        if info.type_id in blueprints:
            return name
    return "truck" if 2.0 * info.bounding_box.extent[0] > 6.5 else "car"


@dataclass(frozen=True)
class Chunk:
    """What one chunk's rasters are drawn from, in the renderer's frame."""

    times_us: list[int]        # the frames' times from the rollout's start
    base_links: np.ndarray     # (n, 4, 4) the ego's base_link (rear axle) at each frame
    cameras: np.ndarray        # (n, 4, 4) the generated camera, camera (OpenCV axes) -> world
    boxes: list[tuple]         # (class, (n, 4, 4) box-to-world, (3,) length, width, height): road users


def make_raster(spec: str, camera, options: dict[str, Any], world):
    """The raster *spec* names (``ludus`` or ``module:Class``), for *camera*."""
    if spec == "ludus":
        from .ludus import LudusRaster

        cls = LudusRaster
    elif ":" in spec:
        module, name = spec.split(":", 1)
        cls = getattr(importlib.import_module(module), name)
    else:
        raise ValueError(f"raster {spec!r}: 'ludus' or 'package.module:Class'")
    raster = cls(camera, options, world)
    if raster.lens != camera.lens:
        raise ValueError(f"raster {spec} draws a {raster.lens} camera; the camera file has a {camera.lens} lens")
    return raster
