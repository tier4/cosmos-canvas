"""The camera the model generates, and how its frames reach a CARLA camera.

The model generates one camera, mounted on the vehicle's ``base_link`` (the
rear axle, on the ground; NVIDIA's rig frame). Its calibration is read from a
JSON file the renderer is given, with one of two lenses:

* a pinhole (fx and fy may differ, the principal point may be off centre)::

    {"width": 1280, "height": 704, "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
     "cam_to_ego": {"translation": [x, y, z],              # base_link frame (FLU), metres
                    "rotation_wxyz": [w, x, y, z]}}         # camera (OpenCV axes) -> base_link

* an f-theta lens, as NVIDIA's ``camera_front_wide_120fov`` that the public
  checkpoint was trained on (``scripts/ftheta_camera.py`` writes it from a
  ClipGT scene's rig, at the generated size)::

    {"width": 1280, "height": 704,
     "ftheta": {"principal_point": [cx, cy],
                "fw_poly": [k0, k1, ...],                   # ray angle (rad) -> pixels from cx, cy
                "max_ray_angle": a,                         # rad
                "linear_cde": [c, d, e]},                   # [[c, d], [e, 1]] on the image plane
     "cam_to_ego": ...}

A CARLA camera on the ego gets the generated frame turned into its own view,
rotation only: each of its pixels' rays is turned from where the CARLA camera
is to where the generated frame was drawn, and read there. For two pinholes
that is the homography ``K_carla R_carla^T R_gen K_gen^-1`` with R each
camera's camera-to-world rotation (OpenCV axes). It is exact for cameras at the
same point, and close for cameras a few centimetres apart on one vehicle (two
front cameras of one rig). A CARLA camera that looks outside the
generated view gets black there.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_wxyz(R: np.ndarray) -> list[float]:
    """Rotation matrix -> unit quaternion [w, x, y, z]."""
    m = np.asarray(R, np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


@dataclass(frozen=True)
class FTheta:
    """An f-theta lens, as NVIDIA's cameras: a ray at angle theta off the optical axis
    lands ``fw_poly(theta)`` pixels from the principal point, in the ray's direction on
    the image plane, through the linear term ``[[c, d], [e, 1]]``. Past ``max_ray_angle``
    the polynomial is continued by its tangent there (as ludus and NVIDIA's ``ray2pixel``
    do): it bounds the polynomial, not the view."""

    principal_point: np.ndarray   # (2,)
    fw_poly: np.ndarray           # (n,) coefficients, lowest order first
    max_ray_angle: float          # rad
    linear: np.ndarray            # (2, 2)

    @classmethod
    def from_dict(cls, d: dict) -> FTheta:
        c, dd, e = (float(v) for v in d.get("linear_cde", (1.0, 0.0, 0.0)))
        return cls(np.asarray(d["principal_point"], np.float64), np.asarray(d["fw_poly"], np.float64),
                   float(d["max_ray_angle"]), np.array([[c, dd], [e, 1.0]]))

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) rays in the camera's OpenCV axes -> (N, 2) pixels (NVIDIA's ``ray2pixel``)
        and (N,) whether they are in front of the lens: a lens up to 90 degrees does not see
        behind itself (as ludus), wider ones see every ray. A pixel may be off the image."""
        rays = np.asarray(rays, np.float64)
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        rho = np.hypot(x, y)
        r = self.radius(np.arctan2(rho, z))
        scale = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
        uv = np.stack([x * scale, y * scale], axis=1) @ self.linear.T + self.principal_point
        return uv, (z > 0) if self.max_ray_angle <= math.pi / 2 else np.ones(len(rays), bool)

    def radius(self, theta: np.ndarray) -> np.ndarray:
        """The distance from the principal point (pixels, before the linear term) of rays at
        *theta* (rad) off the axis: the polynomial, then its tangent past max_ray_angle."""
        P = np.polynomial.polynomial
        a = self.max_ray_angle
        return np.where(theta <= a, P.polyval(theta, self.fw_poly),
                        P.polyval(a, self.fw_poly) + (theta - a) * P.polyval(a, P.polyder(self.fw_poly)))

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 3) unit rays in the camera's OpenCV axes (``project``'s inverse,
        up to 179 degrees off the axis; the lens must spread monotonically)."""
        d = (np.asarray(uv, np.float64).reshape(-1, 2) - self.principal_point) @ np.linalg.inv(self.linear).T
        r = np.hypot(d[:, 0], d[:, 1])
        grid = np.linspace(0.0, math.radians(179.0), 20001)
        radius = self.radius(grid)
        if np.any(np.diff(radius) <= 0):
            radius = radius[: int(np.argmax(np.diff(radius) <= 0)) + 1]  # the monotonic part
        theta = np.interp(r, radius, grid[: len(radius)])
        unit = np.where(r[:, None] > 1e-12, d / np.maximum(r, 1e-12)[:, None], 0.0)
        return np.column_stack([np.sin(theta)[:, None] * unit, np.cos(theta)])


@dataclass(frozen=True)
class GeneratedCamera:
    width: int
    height: int
    K: np.ndarray | None          # (3, 3) for a pinhole
    translation: np.ndarray       # (3,) camera in base_link
    rotation_wxyz: tuple          # camera (OpenCV) -> base_link
    ftheta: FTheta | None = None  # for an f-theta lens

    @classmethod
    def from_json(cls, path: str | Path) -> GeneratedCamera:
        c = json.loads(Path(path).read_text())
        if ("intrinsic" in c) == ("ftheta" in c):
            raise ValueError(f"{path}: give the lens as either 'intrinsic' (a pinhole) or 'ftheta'")
        return cls(int(c["width"]), int(c["height"]),
                   np.asarray(c["intrinsic"], np.float64) if "intrinsic" in c else None,
                   np.asarray(c["cam_to_ego"]["translation"], np.float64),
                   tuple(float(v) for v in c["cam_to_ego"]["rotation_wxyz"]),
                   FTheta.from_dict(c["ftheta"]) if "ftheta" in c else None)

    @property
    def lens(self) -> str:
        return "ftheta" if self.ftheta is not None else "pinhole"

    @property
    def cam_to_base(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = quat_to_R(self.rotation_wxyz)
        m[:3, 3] = self.translation
        return m

    def pose_cv(self, base_link_to_world: np.ndarray) -> np.ndarray:
        """4x4 camera-to-world, OpenCV axes."""
        return np.asarray(base_link_to_world, np.float64) @ self.cam_to_base


def base_link(ego_to_world: np.ndarray, rear_axle_m: float) -> np.ndarray:
    """The ego's base_link pose: its origin moved *rear_axle_m* along its x axis
    (negative: behind), as autoware_carla_scenario's T4Scene export places it."""
    m = np.array(ego_to_world, np.float64)
    m[:3, 3] = m[:3, 3] + m[:3, 0] * rear_axle_m
    return m


def homography(K_src: np.ndarray, src_cv_to_world: np.ndarray,
               K_dst: np.ndarray, dst_cv_to_world: np.ndarray) -> np.ndarray:
    """Pixels of the source camera -> pixels of the destination camera, rotation only."""
    R_src = np.asarray(src_cv_to_world)[:3, :3]
    R_dst = np.asarray(dst_cv_to_world)[:3, :3]
    return np.asarray(K_dst) @ R_dst.T @ R_src @ np.linalg.inv(K_src)


def warp(frame: np.ndarray, H: np.ndarray, width: int, height: int) -> np.ndarray:
    """*frame* seen through *H* (as :func:`homography` returns it, not rescaled), black where
    the destination's rays point behind the source camera.

    A homography maps those rays too: they come out mirrored through the source image's
    centre, so a camera looking away from the generated one would show its frame upside down.
    """
    import cv2

    width, height = int(width), int(height)
    out = cv2.warpPerspective(np.ascontiguousarray(frame), H, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth = np.linalg.inv(H)[2]                   # a destination pixel's depth along the source's axis
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]])
    if (corners @ depth > 0).all():               # linear in (u, v): positive at the corners is positive everywhere
        return out
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    out[depth[0] * u + depth[1] * v + depth[2] <= 0] = 0
    return out


def look_through(frame: np.ndarray, camera: GeneratedCamera, drawn_cv_to_world: np.ndarray,
                 K_dst: np.ndarray, dst_cv_to_world: np.ndarray, width: int, height: int) -> np.ndarray:
    """A pinhole (*K_dst*, at *dst_cv_to_world*) looking through *frame*, which *camera*
    generated at *drawn_cv_to_world*: rotation only, black outside the generated view."""
    if camera.ftheta is None:
        return warp(frame, homography(camera.K, drawn_cv_to_world, K_dst, dst_cv_to_world), width, height)
    import cv2

    width, height = int(width), int(height)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    # each destination pixel's ray, turned into the generating camera's axes
    R = np.asarray(drawn_cv_to_world)[:3, :3].T @ np.asarray(dst_cv_to_world)[:3, :3]
    rays = (R @ np.linalg.inv(K_dst) @ pixels).T
    uv, seen = camera.ftheta.project(rays)
    uv[~seen] = -10.0                             # behind the lens: read the black border, as off the image
    maps = uv.reshape(height, width, 2).astype(np.float32)
    return cv2.remap(np.ascontiguousarray(frame), maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def pixel_rays(camera: GeneratedCamera) -> np.ndarray:
    """(H, W, 3) the ray of every pixel of *camera*, in its OpenCV axes."""
    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
    if camera.ftheta is not None:
        rays = camera.ftheta.unproject(np.column_stack([u.ravel(), v.ravel()]))
    else:
        rays = (np.linalg.inv(camera.K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T
    return rays.reshape(camera.height, camera.width, 3)


def resample(frame: np.ndarray, lens: FTheta, camera: GeneratedCamera,
             R_camera_to_lens: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """*frame*, taken through f-theta *lens*, as *camera* sees it from the same point (turned by
    *R_camera_to_lens*, OpenCV axes; none: the same orientation); black where the lens saw
    nothing. Returns the image and the share of its pixels that fell outside *frame*."""
    import cv2

    rays = pixel_rays(camera).reshape(-1, 3)
    if R_camera_to_lens is not None:
        rays = rays @ np.asarray(R_camera_to_lens, np.float64).T
    uv, seen = lens.project(rays)
    h, w = frame.shape[:2]
    outside = ~seen | (uv[:, 0] < -0.5) | (uv[:, 0] > w - 0.5) | (uv[:, 1] < -0.5) | (uv[:, 1] > h - 0.5)
    uv[~seen] = -10.0
    maps = uv.reshape(camera.height, camera.width, 2).astype(np.float32)
    out = cv2.remap(np.ascontiguousarray(frame), maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    return out, float(outside.mean())


def centre_crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """*frame*'s centred band at the aspect width:height, resized to (width, height).

    For a CARLA camera that stands in for the generated one rather than looking through
    it: full width and centred rows when the frame is taller than the camera (1280x704 ->
    1024x512 takes rows 32-672, a planner's 2:1 crop of the camera), full height and
    centred columns when it is wider.
    """
    import cv2

    width, height = int(width), int(height)
    h, w = frame.shape[:2]
    if w * height >= h * width:
        cw = int(round(h * width / height))
        x0 = (w - cw) // 2
        band = frame[:, x0:x0 + cw]
    else:
        ch = int(round(w * height / width))
        y0 = (h - ch) // 2
        band = frame[y0:y0 + ch]
    return cv2.resize(np.ascontiguousarray(band), (width, height), interpolation=cv2.INTER_AREA)
