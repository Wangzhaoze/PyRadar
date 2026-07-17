"""Backend-neutral point-cloud sampling and registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .geometry import transform_points


def _points(values: ArrayLike, name: str = "points") -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (N, 3).")
    return array


@dataclass(frozen=True, slots=True)
class ICPResult:
    """Result of deterministic point-to-point ICP registration."""

    transform: NDArray[np.float64]
    aligned: NDArray[np.float64]
    rmse: float
    iterations: int
    converged: bool


def voxel_downsample(points: ArrayLike, *, voxelSize: float) -> NDArray[np.float64]:
    """Replace points in each cubic voxel by their centroid."""

    array = _points(points)
    if not np.isfinite(voxelSize) or voxelSize <= 0.0:
        raise ValueError("voxelSize must be finite and positive.")
    if not len(array):
        return array.copy()
    keys = np.floor(array / voxelSize).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(np.max(inverse)) + 1
    sums = np.zeros((count, 3), dtype=float)
    totals = np.zeros(count, dtype=np.int64)
    np.add.at(sums, inverse, array)
    np.add.at(totals, inverse, 1)
    return sums / totals[:, None]


def random_sample(
    points: ArrayLike, *, count: int, seed: int | None = None
) -> NDArray[np.float64]:
    """Sample points without replacement using a reproducible generator."""

    array = _points(points)
    if count < 0 or count > len(array):
        raise ValueError("count must lie between zero and the point count.")
    indices = np.random.default_rng(seed).choice(len(array), size=count, replace=False)
    return array[indices]


def axis_aligned_bounds(
    points: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return minimum and maximum FLU corners."""

    array = _points(points)
    if not len(array):
        raise ValueError("At least one point is required.")
    return np.min(array, axis=0), np.max(array, axis=0)


def _rigid_fit(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    sourceCenter = np.mean(source, axis=0)
    targetCenter = np.mean(target, axis=0)
    covariance = (source - sourceCenter).T @ (target - targetCenter)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = targetCenter - rotation @ sourceCenter
    return transform


def icp_register(
    source: ArrayLike,
    target: ArrayLike,
    *,
    initialTransform: ArrayLike | None = None,
    maxIterations: int = 50,
    tolerance: float = 1e-6,
    maxCorrespondenceDistance: float = np.inf,
) -> ICPResult:
    """Register point clouds with nearest-neighbor point-to-point ICP."""

    sourceArray = _points(source, "source")
    targetArray = _points(target, "target")
    if len(sourceArray) < 3 or len(targetArray) < 3:
        raise ValueError("ICP requires at least three source and target points.")
    if (
        maxIterations < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or np.isnan(maxCorrespondenceDistance)
        or maxCorrespondenceDistance <= 0.0
    ):
        raise ValueError("Invalid ICP iteration, tolerance, or distance settings.")
    transform = (
        np.eye(4)
        if initialTransform is None
        else np.asarray(initialTransform, dtype=float).copy()
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("initialTransform must be a finite 4x4 matrix.")
    aligned = transform_points(sourceArray, transform)
    tree = cKDTree(targetArray)
    previous = np.inf
    converged = False
    rmse = np.inf
    for _iteration in range(1, maxIterations + 1):
        distances, indices = tree.query(aligned, workers=1)
        accepted = distances <= maxCorrespondenceDistance
        if np.count_nonzero(accepted) < 3:
            raise ValueError("Fewer than three ICP correspondences passed the gate.")
        delta = _rigid_fit(aligned[accepted], targetArray[indices[accepted]])
        aligned = transform_points(aligned, delta)
        transform = delta @ transform
        updatedDistances, _ = tree.query(aligned, workers=1)
        acceptedUpdated = updatedDistances <= maxCorrespondenceDistance
        rmse = float(np.sqrt(np.mean(updatedDistances[acceptedUpdated] ** 2)))
        if abs(previous - rmse) <= tolerance:
            converged = True
            break
        previous = rmse
    return ICPResult(transform, aligned, rmse, _iteration, converged)


__all__ = [
    "ICPResult",
    "axis_aligned_bounds",
    "icp_register",
    "random_sample",
    "voxel_downsample",
]
