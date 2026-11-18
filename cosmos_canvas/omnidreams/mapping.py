"""Map helpers shared by the rasters, in the renderer's frame (carla_canvas
``right_handed``: CARLA with y negated, OpenDRIVE's and roadgen's frame).

* intersections: the outlines of roadgen's ClipGT intersection areas, and the
  stretches of a polyline that do not run inside them (roadgen outlines every
  connecting road of a junction; HD-map rasters leave the area empty);
* map extras (``scripts/export_map_extras.py``, once per map): what carla_canvas
  does not send, as boxes: signal heads and parked vehicles (CARLA's
  environment objects).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_HEAD_HALF_HEIGHT = 0.3   # m: a signal box smaller than this is not a housing

#: environment-object labels of the map extras' parked vehicles -> road-user classes
STATIC_CLASS = {"Car": "car", "Truck": "truck", "Bus": "bus", "Motorcycle": "motorcycle",
                "Bicycle": "bicycle"}


@dataclass(frozen=True)
class StaticBox:
    """A box that never moves, in the renderer's frame."""

    cls: str
    centre: np.ndarray        # (3,)
    rotation: np.ndarray      # (3, 3) box-to-world, x along its length
    size_lwh: np.ndarray      # (3,)

    def pose(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.rotation, self.centre
        return m


# -- intersections -------------------------------------------------------------------

def clipgt_intersections(directory: Path, clip: str) -> list[np.ndarray]:
    """The (N, 2) outlines of a ClipGT export's intersection areas."""
    import pyarrow.parquet as pq

    path = Path(directory) / f"{clip}.intersection_area.parquet"
    if not path.exists():
        return []
    out = []
    for row in pq.read_table(path).to_pylist():
        pts = np.array([[p["x"], p["y"]] for p in row["intersection_area"]["location"]], np.float64)
        if len(pts) >= 3:
            out.append(pts)
    return out


def _inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,) whether 2-D *points* lie inside *poly* (even-odd rule)."""
    x, y = points[:, 0:1], points[:, 1:2]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return crosses.sum(axis=1) % 2 == 1


def _edge_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,) distance of 2-D *points* to *poly*'s outline."""
    a, b = poly[None], np.roll(poly, -1, axis=0)[None]
    ab = b - a
    t = np.clip(((points[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points[:, None] - (a + t[..., None] * ab), axis=-1).min(axis=1)


def densify(pts: np.ndarray, step: float = 1.0) -> np.ndarray:
    """A polyline with a vertex at least every *step* metres (a straight edge across
    an area has its two vertices on the area's outline)."""
    out = [pts[:1]]
    for a, b in itertools.pairwise(pts):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


def runs_outside(pts: np.ndarray, areas: list[np.ndarray], margin: float = 0.5,
                 boxes: list[tuple] | None = None) -> list[np.ndarray]:
    """The stretches of polyline *pts* that are not deeper than *margin* inside any of
    *areas* (a line along an area's edge, such as a corner's curb, stays); *pts* itself
    when it does not reach any area."""
    if boxes is None:
        boxes = [(a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()) for a in areas]
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    if not any(lo[0] <= x1 and hi[0] >= x0 and lo[1] <= y1 and hi[1] >= y0 for x0, y0, x1, y1 in boxes):
        return [pts]
    pts = densify(pts)
    drop = np.zeros(len(pts), bool)
    for area, (x0, y0, x1, y1) in zip(areas, boxes, strict=True):
        near = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if near.any():
            idx = np.flatnonzero(near)
            xy = pts[idx, :2]
            deep = _inside(xy, area) & (_edge_distance(xy, area) > margin)
            drop[idx[deep]] = True
    if not drop.any():
        return [pts]
    edges = np.flatnonzero(np.diff(np.r_[0, (~drop).astype(int), 0]))
    return [pts[start:stop] for start, stop in zip(edges[::2], edges[1::2], strict=True) if stop - start >= 2]


# -- map extras -----------------------------------------------------------------------

def _carla_box(location, rotation_pyr_deg) -> tuple[np.ndarray, np.ndarray]:
    """A CARLA world box pose (left-handed) as (centre, 3x3) in the right-handed frame."""
    from carla_canvas.geometry import Transform

    m = Transform(tuple(location), tuple(rotation_pyr_deg)).matrix("right_handed")
    return m[:3, 3], m[:3, :3]


def dedupe_static(boxes: list[dict], radius: float = 1.0) -> list[dict]:
    """One box per parked vehicle: CARLA returns a vehicle's meshes (body, an
    inner part) as separate objects a few cm apart; keep the largest of each
    same-label group within *radius* metres."""
    keep: list[dict] = []
    for b in sorted(boxes, key=lambda b: float(np.prod(b["extent"])), reverse=True):
        c = np.asarray(b["location"][:2])
        if not any(k["label"] == b["label"] and np.linalg.norm(np.asarray(k["location"][:2]) - c) < radius
                   for k in keep):
            keep.append(b)
    return keep


def extra_boxes(path: Path) -> tuple[list[StaticBox], list[StaticBox]]:
    """Signal heads (class ``traffic_light``) and parked vehicles, as boxes in the
    renderer's frame, from ``scripts/export_map_extras.py``'s file (CARLA world frame)."""
    doc = json.loads(Path(path).read_text())
    heads = []
    for b in doc.get("traffic_light_heads", []):
        half = np.asarray(b["extent"], np.float64)
        if half[2] < MIN_HEAD_HALF_HEIGHT:
            continue
        centre, rot = _carla_box(b["location"], b["rotation_pyr_deg"])
        heads.append(StaticBox("traffic_light", centre, rot, 2 * half))
    parked = []
    for b in dedupe_static(doc.get("parked_vehicles", [])):
        centre, rot = _carla_box(b["location"], b["rotation_pyr_deg"])
        parked.append(StaticBox(STATIC_CLASS.get(b["label"], "car"), centre, rot, 2 * np.asarray(b["extent"])))
    return heads, parked
