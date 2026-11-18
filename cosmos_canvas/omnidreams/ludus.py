"""The HD-map raster of NVIDIA's public OmniDreams, drawn by FlashDreams' ludus.

NVIDIA's public checkpoint (FlashDreams' ``omnidreams`` configuration) takes an
HD-map raster that FlashDreams' own renderer, ludus, draws (its ClipGT loader's
"v3" palette); NVIDIA's closed-loop apps in FlashDreams (``interactive_drive``,
``omnidreams_game_engine``) feed the checkpoint with it. This module draws it
the same way, with the same settings, from what the renderer is given, in the
renderer's frame (carla_canvas ``right_handed``, OpenDRIVE's and ClipGT's):

  map          roadgen's ClipGT export of the OpenDRIVE carla_canvas sends, read
               by ludus' ClipGT loader, after ``prepare_clipgt`` (as carla_transfer
               prepares its CARLA scenes): lane-line styles in the ClipGT
               vocabulary, and lane lines and road boundaries cut where they run
               inside an intersection (roadgen outlines every connecting road of a
               junction; ClipGT leaves the area empty). Its traffic lights are
               left out: OpenDRIVE places them at the ground.
  map extras   (optional, ``scripts/export_map_extras.py``) signal heads as
               ClipGT traffic lights (grey: carla_canvas sends no light states),
               parked vehicles as boxes that never move
  road users   boxes at each chunk's frame times, coloured by class as ClipGT
               obstacles are
  camera       the generated camera's f-theta lens (``rig.py``), e.g. NVIDIA's
               ``camera_front_wide_120fov``

ludus and torch come from the FlashDreams environment. ``prepare_clipgt`` and
``box_arrays`` need neither.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import mapping
from .rig import GeneratedCamera, quat_wxyz

log = logging.getLogger("cosmos_canvas.omnidreams")

#: roadgen's lane-line styles -> the ClipGT vocabulary ludus reads (carla_transfer's
#: table); ludus draws SOLID_DOUBLE as one solid line
LANE_LINE_STYLES = {"DASHED_SINGLE": "LONG_DASHED_SINGLE", "SOLID_DOUBLE": "SOLID_GROUP",
                    "DASHED_DOUBLE": "LONG_DASHED_SINGLE"}
# NVIDIA's apps' raster settings (interactive_drive's and omnidreams_game_engine's RasterConfig)
LINE_WIDTH_PX = 12.0
POLE_WIDTH_PX = 5.0
WIREFRAME_PX = 4.0
DEPTH_MAX_M = 200.0
FW_POLY_LEN = 6         # ludus' f-theta takes six coefficients

#: (category, (N, 4, 4) box-to-world at the frames' times, (3,) length, width, height)
Box = tuple
#: an equidistant f-theta camera for a rig that has none (ClipGT's rig JSON)
PLACEHOLDER_CAMERA = {
    "name": "camera:placeholder",
    "properties": {"cx": "640", "cy": "352", "width": "1280", "height": "704", "polynomial": "0 600",
                   "polynomial-type": "angle-to-pixeldistance"},
    "nominalSensor2Rig_FLU": {"roll-pitch-yaw": [0.0, 0.0, 0.0], "t": [0.0, 0.0, 0.0]},
}


# -- the map, before ludus reads it ----------------------------------------------------

def _xyz(points: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"], p["z"]] for p in points], np.float64)


def _points(xyz: np.ndarray) -> list[dict]:
    return [{"x": x, "y": y, "z": z} for x, y, z in np.asarray(xyz, np.float64).tolist()]


def _quat_xyzw(R: np.ndarray) -> list[float]:
    w, x, y, z = quat_wxyz(R)
    return [x, y, z, w]


def prepare_clipgt(directory: Path, clip: str, heads: Sequence[mapping.StaticBox] = ()) -> dict[str, int]:
    """roadgen's ClipGT export *directory* (``{clip}.{layer}.parquet``) as ludus should
    read it, rewritten in place; returns the counts of what is drawn.

    * lane lines: styles in the ClipGT vocabulary (``LANE_LINE_STYLES``);
    * lane lines and road boundaries: cut inside the export's intersection areas
      (``mapping.runs_outside``);
    * traffic lights: *heads* (signal heads, as ``mapping.extra_boxes`` gives them), none
      without them;
    * the rig: a camera when it has none (roadgen writes an empty rig without a route;
      ludus' loader needs one, and ``LudusRaster`` draws its own).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    areas = mapping.clipgt_intersections(directory, clip)
    boxes = [(a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()) for a in areas]
    counts: dict[str, int] = {"intersections": len(areas)}
    for layer, key in (("lane_line", "line_rail"), ("road_boundary", "location")):
        path = directory / f"{clip}.{layer}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path)
        rows = []
        for row in table.to_pylist():
            rec = dict(row[layer])
            if layer == "lane_line":
                rec["styles"] = [LANE_LINE_STYLES.get(s, s) for s in rec.get("styles") or []]
            pts = _xyz(rec[key])
            if len(pts) < 2:
                continue
            for run in runs(pts, areas, boxes):
                rows.append({**row, layer: {**rec, key: _points(run)}})
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        counts[layer] = len(rows)

    path = directory / f"{clip}.traffic_light.parquet"
    rows = [{"traffic_light": {"center": dict(zip("xyz", map(float, h.centre), strict=True)),
                               "dimensions": dict(zip("xyz", map(float, h.size_lwh), strict=True)),
                               "orientation": dict(zip("xyzw", _quat_xyzw(h.rotation), strict=True))}}
            for h in heads]
    if rows:
        schema = pq.read_schema(path) if path.exists() else None
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    elif path.exists():
        path.unlink()
    counts["traffic_lights"] = len(rows)

    path = directory / f"{clip}.calibration_estimate.parquet"
    rig = json.loads(pq.read_table(path).to_pylist()[0]["calibration_estimate"]["rig_json"]) if path.exists() \
        else {"rig": {"sensors": []}}
    sensors = rig.setdefault("rig", {}).setdefault("sensors", [])
    if not any(str(s.get("name", "")).startswith("camera:") and s.get("properties") for s in sensors):
        sensors.append(PLACEHOLDER_CAMERA)
        pq.write_table(pa.Table.from_pylist([{"calibration_estimate": {"rig_json": json.dumps(rig)}}]), path)
    return counts


def runs(pts: np.ndarray, areas: list[np.ndarray], boxes: list[tuple]) -> list[np.ndarray]:
    return mapping.runs_outside(pts, areas, boxes=boxes) if areas else [pts]


# -- boxes ------------------------------------------------------------------------------

def box_arrays(boxes: Sequence[Box], times_us: Sequence[int]) -> dict[str, np.ndarray]:
    """ludus' cube-pool arrays for *boxes*, each with a pose at every one of *times_us*:
    track timestamps, centres, rotations (quaternions xyzw), sizes (length, width,
    height along the box's x, y, z) and the track lengths."""
    times = np.asarray(times_us, np.int64)
    n = len(times)
    for _, poses, _ in boxes:
        if len(poses) != n:
            raise ValueError(f"a box has {len(poses)} poses for {n} frame times")
    poses = np.concatenate([np.asarray(p, np.float64) for _, p, _ in boxes]) if boxes else np.zeros((0, 4, 4))
    return {
        "track_timestamps_us": np.tile(times, len(boxes)),
        "translations": poses[:, :3, 3].astype(np.float32),
        "quaternions": np.array([_quat_xyzw(R) for R in poses[:, :3, :3]], np.float32).reshape(-1, 4),
        "scales": np.array([np.asarray(lwh, np.float64) for _, _, lwh in boxes], np.float32).reshape(-1, 3),
        "track_lengths": np.full(len(boxes), n, np.int32),
    }


# -- ludus ------------------------------------------------------------------------------

#: optical (x right, y down, z forward) -> FLU (x forward, y left, z up): the columns
#: are FLU's axes in optical coordinates, so camera_FLU_to_world = camera_cv_to_world @ this
CV_FROM_FLU = np.eye(4)
CV_FROM_FLU[:3, :3] = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]


class LudusRaster:
    """``raster: ludus`` (raster.py): ludus drawing one f-theta camera, the map uploaded once,
    the boxes per chunk."""

    lens = "ftheta"

    def __init__(self, camera: GeneratedCamera, options: dict | None = None, world=None,
                 device: str = "cuda") -> None:
        self.options = dict(options or {})
        if camera.ftheta is None:
            raise ValueError("raster: ludus draws an f-theta camera: give the camera an 'ftheta' lens "
                             "(scripts/ftheta_camera.py writes NVIDIA's from a ClipGT scene)")
        import torch
        from ludus_renderer import FThetaCamera, LudusCudaTimestampedContext

        self._torch = torch
        self.device = torch.device(device)
        self.camera = camera
        f = camera.ftheta
        poly = np.zeros(FW_POLY_LEN)
        if len(f.fw_poly) > FW_POLY_LEN and np.any(f.fw_poly[FW_POLY_LEN:]):
            raise ValueError(f"ludus takes {FW_POLY_LEN} f-theta coefficients, the camera has {len(f.fw_poly)}")
        poly[:min(len(f.fw_poly), FW_POLY_LEN)] = f.fw_poly[:FW_POLY_LEN]

        def tensor(x):
            return torch.as_tensor(np.asarray(x, np.float32), device=self.device)

        ctx = LudusCudaTimestampedContext(device=self.device)
        ctx.set_depth_scaling(True)
        ctx.set_msaa_samples(4)
        ctx.set_max_tessellation_levels(cube=3)
        ctx.set_line_widths(polyline_regular=LINE_WIDTH_PX, polyline_bev=max(2.0, LINE_WIDTH_PX * 0.4),
                            ego_traj_regular=POLE_WIDTH_PX, ego_traj_bev=max(2.0, POLE_WIDTH_PX * 0.6),
                            wireframe=WIREFRAME_PX)
        ctx.upload_cameras([FThetaCamera(principal_point=tensor(f.principal_point),
                                         image_size=tensor([camera.width, camera.height]),
                                         fw_poly=tensor(poly), max_ray_angle=float(f.max_ray_angle),
                                         linear_distortion=tensor(f.linear), depth_max=DEPTH_MAX_M)])
        self.ctx = ctx
        self._base = None
        self._scene_id: int | None = None
        self._static: list[Box] = []

    def load_map(self, world) -> dict[str, int]:
        """The map: roadgen's ClipGT export of the world's OpenDRIVE (``prepare_clipgt``), with the
        map extras' signal heads and parked vehicles (``map_extras`` option)."""
        import tempfile

        import roadgen

        o = self.options
        heads, parked = mapping.extra_boxes(world.resolve(o["map_extras"])) if o.get("map_extras") else ([], [])
        with tempfile.TemporaryDirectory(prefix="cosmos_canvas_omnidreams_") as tmp:
            path = Path(tmp) / "map.xodr"
            path.write_text(world.opendrive)
            road_map = roadgen.read_opendrive(str(path), sampling=0.5)
            clipgt = Path(tmp) / "clipgt"
            clip = road_map.export_clipgt(str(clipgt), trace=False)
            counts = prepare_clipgt(clipgt, clip, heads)
            self.load_clipgt(clipgt, parked)
        counts["parked_vehicles"] = len(parked)
        return counts

    def load_clipgt(self, directory: Path, parked: Sequence[mapping.StaticBox] = ()) -> None:
        """The map from a ClipGT directory (``prepare_clipgt``), and boxes that never move."""
        from ludus_renderer import load_scene

        scene = load_scene(directory, device=self.device, include_ego_trajectory=False,
                           include_ego_obstacle=False, include_dynamic_obstacles=False)
        self.ctx.clear_scenes()
        self._base = scene.timestamped_scene
        self._scene_id = self.ctx.upload_scene(self._base)
        self._static = [(b.cls, b.centre, b.rotation, b.size_lwh) for b in parked]

    def _pool(self, boxes: Sequence[Box], times_us: Sequence[int]):
        from ludus_renderer import CUBE_FLAG_WIREFRAME, PRIM_OBSTACLE, CubePool
        from ludus_renderer.clipgt import CATEGORY_TO_OBJECT_TYPE, OBSTACLE_COLORS_V3

        torch = self._torch
        a = box_arrays(boxes, times_us)
        colors = np.array([np.ravel(OBSTACLE_COLORS_V3[CATEGORY_TO_OBJECT_TYPE.get(cls, "Other")])
                           for cls, _, _ in boxes], np.float32)

        def t(x):
            return torch.as_tensor(x, device=self.device)

        return CubePool(timestamps_us=t(np.unique(a["track_timestamps_us"])),
                        cube_ts_prefix_sum=t(np.cumsum(a["track_lengths"], dtype=np.int32)),
                        track_timestamps_us=t(a["track_timestamps_us"]), translations=t(a["translations"]),
                        quaternions=t(a["quaternions"]), scales=t(a["scales"]), colors=t(colors),
                        prim_type_id=PRIM_OBSTACLE, render_flags=CUBE_FLAG_WIREFRAME)

    def draw(self, chunk) -> np.ndarray:
        """(n, H, W, 3) uint8: the chunk's rasters (raster.Chunk), with the parked vehicles."""
        return self.draw_at(chunk.cameras, chunk.times_us, chunk.boxes)

    def draw_at(self, cam_cv_to_world: np.ndarray, times_us: Sequence[int], boxes: Sequence[Box]) -> np.ndarray:
        """(N, H, W, 3) uint8: the raster at *N* camera poses (camera-to-world, OpenCV axes)
        and frame times, with *boxes* (each with a pose per frame) and the parked ones."""
        from ludus_renderer import CAMERA_TYPE_REGULAR, TimestampedScene

        if self._base is None:
            raise RuntimeError("load_map first")
        torch = self._torch
        n = len(times_us)
        boxes = list(boxes) + [(cls, np.repeat(_pose(c, R)[None], n, axis=0), lwh)
                               for cls, c, R, lwh in self._static]
        cube_pools = list(self._base.cube_pools or [])
        if boxes:
            cube_pools.append(self._pool(boxes, times_us))
        self.ctx.replace_scene(self._scene_id, TimestampedScene(polyline_pools=self._base.polyline_pools,
                                                                polygon_pools=self._base.polygon_pools,
                                                                cube_pools=cube_pools))
        flu_to_world = np.asarray(cam_cv_to_world, np.float64) @ CV_FROM_FLU
        world_to_cam = torch.linalg.inv(torch.as_tensor(flu_to_world, dtype=torch.float32, device=self.device))
        images = self.ctx.render_uniform(scene_id=self._scene_id, camera_id=0,
                                         timestamps_us=[int(v) for v in times_us],
                                         camera_type_id=CAMERA_TYPE_REGULAR, camera_poses=world_to_cam,
                                         resolution=(self.camera.height, self.camera.width))[..., :3]
        if self.ctx.needs_vflip:
            images = images.flip(1)
        if images.dtype != torch.uint8:
            images = (images.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        return images.cpu().numpy()


def _pose(centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = centre
    return m
