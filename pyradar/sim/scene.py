"""Deterministic point-scene generation and angular range images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.utils.geometry import (
    cartesian_to_spherical,
    inverse_transform,
    transform_points,
)


@dataclass(frozen=True, slots=True)
class RangeImage:
    """Nearest-return angular projection of a point scene."""

    ranges: NDArray[np.float64]
    azimuth: NDArray[np.float64]
    elevation: NDArray[np.float64]
    pointIndices: NDArray[np.int64]
    features: NDArray | None = None


def sample_sphere(
    *,
    radius: float = 1.0,
    numPoints: int = 1000,
    center: ArrayLike = (0.0, 0.0, 0.0),
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Uniformly sample points on a sphere surface."""

    if not np.isfinite(radius) or radius <= 0.0 or numPoints < 1:
        raise ValueError("radius and numPoints must be positive.")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must contain three finite coordinates.")
    generator = np.random.default_rng(seed)
    direction = generator.normal(size=(numPoints, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return origin + radius * direction


def sample_plane(
    *,
    width: float = 1.0,
    height: float = 1.0,
    numPoints: int = 1000,
    center: ArrayLike = (1.0, 0.0, 0.0),
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Sample a y-z plane facing the radar's forward x direction."""

    if (
        not np.isfinite(width)
        or width <= 0.0
        or not np.isfinite(height)
        or height <= 0.0
        or numPoints < 1
    ):
        raise ValueError("width, height, and numPoints must be positive.")
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must contain three finite coordinates.")
    generator = np.random.default_rng(seed)
    points = np.empty((numPoints, 3), dtype=float)
    points[:, 0] = origin[0]
    points[:, 1] = origin[1] + generator.uniform(-width / 2.0, width / 2.0, numPoints)
    points[:, 2] = origin[2] + generator.uniform(-height / 2.0, height / 2.0, numPoints)
    return points


def linear_trajectory(
    start: ArrayLike,
    stop: ArrayLike,
    *,
    numFrames: int,
) -> NDArray[np.float64]:
    """Return ``(numFrames, 4, 4)`` FLU poses along a straight line."""

    startArray = np.asarray(start, dtype=float)
    stopArray = np.asarray(stop, dtype=float)
    if (
        startArray.shape != (3,)
        or stopArray.shape != (3,)
        or not np.all(np.isfinite(startArray))
        or not np.all(np.isfinite(stopArray))
        or numFrames < 1
    ):
        raise ValueError("start/stop must be length three and numFrames positive.")
    poses = np.repeat(np.eye(4)[None, :, :], numFrames, axis=0)
    poses[:, :3, 3] = np.linspace(startArray, stopArray, numFrames)
    return poses


def render_range_image(
    points: ArrayLike,
    *,
    radarPose: ArrayLike | None = None,
    azimuthFov: tuple[float, float] = (-np.pi / 2.0, np.pi / 2.0),
    elevationFov: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
    angularResolution: tuple[float, float] = (np.deg2rad(1.0), np.deg2rad(1.0)),
    maxRange: float = np.inf,
    features: ArrayLike | None = None,
) -> RangeImage:
    """Project world points into a nearest-return azimuth/elevation image."""

    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.all(np.isfinite(cloud)):
        raise ValueError("points must have shape (N, 3) with finite values.")
    pose = np.eye(4) if radarPose is None else np.asarray(radarPose, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("radarPose must be a finite 4x4 radar-to-world transform.")
    azMin, azMax = azimuthFov
    elMin, elMax = elevationFov
    azResolution, elResolution = angularResolution
    if (
        not np.all(np.isfinite((azMin, azMax, elMin, elMax)))
        or azMin >= azMax
        or elMin >= elMax
        or not np.isfinite(azResolution)
        or azResolution <= 0.0
        or not np.isfinite(elResolution)
        or elResolution <= 0.0
        or np.isnan(maxRange)
        or maxRange <= 0.0
    ):
        raise ValueError("Invalid FOV, angular resolution, or maxRange.")
    featureArray = None if features is None else np.asarray(features)
    if featureArray is not None and (
        featureArray.ndim == 0 or featureArray.shape[0] != cloud.shape[0]
    ):
        raise ValueError("features must have the same first dimension as points.")

    local = transform_points(cloud, inverse_transform(pose))
    spherical = cartesian_to_spherical(local)
    ranges, azimuth, elevation = spherical.T
    visible = (
        (ranges > 0.0)
        & (ranges <= maxRange)
        & (azimuth >= azMin)
        & (azimuth < azMax)
        & (elevation >= elMin)
        & (elevation < elMax)
    )
    azimuthAxis = np.asarray(
        azMin
        + (np.arange(int(np.ceil((azMax - azMin) / azResolution))) + 0.5)
        * azResolution,
        dtype=np.float64,
    )
    elevationAxis = np.asarray(
        elMin
        + (np.arange(int(np.ceil((elMax - elMin) / elResolution))) + 0.5)
        * elResolution,
        dtype=np.float64,
    )
    shape = (elevationAxis.size, azimuthAxis.size)
    rangeImage = np.full(shape, np.inf, dtype=np.float64)
    pointIndices = np.full(shape, -1, dtype=np.int64)
    featureImage = (
        None
        if featureArray is None
        else np.zeros((*shape, *featureArray.shape[1:]), dtype=featureArray.dtype)
    )
    candidates = np.flatnonzero(visible)
    # Far-to-near assignment gives a deterministic nearest-return z-buffer.
    candidates = candidates[np.argsort(ranges[candidates])[::-1]]
    for index in candidates:
        azIndex = int((azimuth[index] - azMin) // azResolution)
        elIndex = int((elevation[index] - elMin) // elResolution)
        rangeImage[elIndex, azIndex] = ranges[index]
        pointIndices[elIndex, azIndex] = index
        if featureImage is not None and featureArray is not None:
            featureImage[elIndex, azIndex] = featureArray[index]
    return RangeImage(
        ranges=rangeImage,
        azimuth=azimuthAxis,
        elevation=elevationAxis,
        pointIndices=pointIndices,
        features=featureImage,
    )


__all__ = [
    "RangeImage",
    "linear_trajectory",
    "render_range_image",
    "sample_plane",
    "sample_sphere",
]
