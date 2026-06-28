"""Camera-plane human localization utilities.

The real robot starts with a monocular camera. This module keeps the geometry
and tracking math independent of Picamera2, YOLO, supervision, OpenCV, or ROS so
it can be tested on a normal development machine. The live script adapts those
external libraries into the small dataclasses defined here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection2D:
    """Single image-space person detection.

    ``xyxy`` is ``(x_min, y_min, x_max, y_max)`` in pixels. ``track_id`` should
    come from a tracker such as supervision's ByteTrack. If it is missing, the
    live script should still work, but velocities are only stable when IDs are
    stable across frames.
    """

    xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None


@dataclass(frozen=True)
class HumanTrack:
    """Robot-local human state suitable for policy/shield integration."""

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    confidence: float

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class PlanarCalibration:
    """Homography from image pixels to the robot's ground plane.

    The ground coordinate convention is robot-local: +x is forward, +y is left,
    and units are meters. This matches the local frame used by the trained
    policy observations.
    """

    image_to_ground: np.ndarray
    image_points: tuple[tuple[float, float], ...] = ()
    ground_points: tuple[tuple[float, float], ...] = ()
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        H = np.asarray(self.image_to_ground, dtype=float)
        if H.shape != (3, 3):
            raise ValueError("image_to_ground must be a 3x3 matrix")
        if not np.isfinite(H).all():
            raise ValueError("image_to_ground must contain only finite values")
        object.__setattr__(self, "image_to_ground", H)

    @classmethod
    def from_points(
        cls,
        image_points: Sequence[Sequence[float]],
        ground_points: Sequence[Sequence[float]],
        *,
        frame_id: str = "base_link",
    ) -> "PlanarCalibration":
        """Estimate image->ground homography from at least four correspondences."""

        image = _as_point_array(image_points, "image_points")
        ground = _as_point_array(ground_points, "ground_points")
        if len(image) != len(ground):
            raise ValueError("image_points and ground_points must have the same length")
        if len(image) < 4:
            raise ValueError("at least four point correspondences are required")
        H = _fit_homography(image, ground)
        return cls(
            H,
            image_points=tuple(map(tuple, image.tolist())),
            ground_points=tuple(map(tuple, ground.tolist())),
            frame_id=frame_id,
        )

    def pixel_to_ground(self, point: Sequence[float]) -> np.ndarray:
        """Project one pixel coordinate to ``[x_forward_m, y_left_m]``."""

        p = np.asarray([float(point[0]), float(point[1]), 1.0], dtype=float)
        projected = self.image_to_ground @ p
        if abs(projected[2]) < 1e-12:
            raise ValueError("homogeneous projection has near-zero scale")
        return projected[:2] / projected[2]

    def pixels_to_ground(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        pts = _as_point_array(points, "points")
        homogeneous = np.column_stack([pts, np.ones(len(pts), dtype=float)])
        projected = (self.image_to_ground @ homogeneous.T).T
        scale = projected[:, 2:3]
        if np.any(np.abs(scale) < 1e-12):
            raise ValueError("homogeneous projection has near-zero scale")
        return projected[:, :2] / scale

    def reprojection_errors(self) -> np.ndarray:
        """Return per-point ground-plane reprojection error in meters."""

        if not self.image_points or not self.ground_points:
            return np.array([], dtype=float)
        projected = self.pixels_to_ground(self.image_points)
        ground = np.asarray(self.ground_points, dtype=float)
        return np.linalg.norm(projected - ground, axis=1)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "image_to_ground": self.image_to_ground.tolist(),
            "image_points": [list(p) for p in self.image_points],
            "ground_points_m": [list(p) for p in self.ground_points],
            "reprojection_error_m": self.reprojection_errors().tolist(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlanarCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            np.asarray(data["image_to_ground"], dtype=float),
            image_points=tuple(tuple(p) for p in data.get("image_points", ())),
            ground_points=tuple(tuple(p) for p in data.get("ground_points_m", ())),
            frame_id=str(data.get("frame_id", "base_link")),
        )


class VisionLocalizer:
    """Convert tracked person boxes into robot-local human tracks."""

    def __init__(
        self,
        calibration: PlanarCalibration,
        *,
        min_confidence: float = 0.35,
        max_human_speed: float = 2.0,
        velocity_alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if max_human_speed <= 0.0:
            raise ValueError("max_human_speed must be positive")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        self.calibration = calibration
        self.min_confidence = float(min_confidence)
        self.max_human_speed = float(max_human_speed)
        self.velocity_alpha = float(velocity_alpha)
        self._previous: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
        self._next_synthetic_id = 1

    def update(self, detections: Iterable[Detection2D], timestamp_s: float) -> list[HumanTrack]:
        tracks: list[HumanTrack] = []
        seen_ids: set[int] = set()
        for det in detections:
            if det.confidence < self.min_confidence:
                continue
            track_id = self._resolve_track_id(det)
            foot_px = bbox_bottom_center(det.xyxy)
            position = self.calibration.pixel_to_ground(foot_px)
            velocity = self._estimate_velocity(track_id, position, float(timestamp_s))
            seen_ids.add(track_id)
            tracks.append(
                HumanTrack(
                    track_id=track_id,
                    x=float(position[0]),
                    y=float(position[1]),
                    vx=float(velocity[0]),
                    vy=float(velocity[1]),
                    confidence=float(det.confidence),
                )
            )

        self._previous = {
            track_id: state
            for track_id, state in self._previous.items()
            if track_id in seen_ids
        }
        tracks.sort(key=lambda t: t.distance)
        return tracks

    def _resolve_track_id(self, det: Detection2D) -> int:
        if det.track_id is not None:
            return int(det.track_id)
        synthetic_id = self._next_synthetic_id
        self._next_synthetic_id += 1
        return synthetic_id

    def _estimate_velocity(self, track_id: int, position: np.ndarray, timestamp_s: float) -> np.ndarray:
        previous = self._previous.get(track_id)
        if previous is None:
            velocity = np.zeros(2, dtype=float)
        else:
            prev_t, prev_pos, prev_vel = previous
            dt = max(timestamp_s - prev_t, 1e-6)
            raw_velocity = (position - prev_pos) / dt
            speed = float(np.linalg.norm(raw_velocity))
            if speed > self.max_human_speed:
                raw_velocity = raw_velocity * (self.max_human_speed / speed)
            velocity = self.velocity_alpha * raw_velocity + (1.0 - self.velocity_alpha) * prev_vel
        self._previous[track_id] = (timestamp_s, position.astype(float), velocity.astype(float))
        return velocity


def bbox_bottom_center(xyxy: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy must satisfy x2 > x1 and y2 > y1")
    return ((x1 + x2) / 2.0, y2)


def _as_point_array(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _fit_homography(image: np.ndarray, ground: np.ndarray) -> np.ndarray:
    rows = []
    for (u, v), (x, y) in zip(image, ground):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    A = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("point correspondences produced a singular homography")
    return H
