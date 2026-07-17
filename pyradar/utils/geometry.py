"""Right-handed FLU coordinate conversion and rigid transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def spherical_to_cartesian(
    range: ArrayLike,
    azimuth: ArrayLike,
    elevation: ArrayLike | float = 0.0,
) -> NDArray[np.float64]:
    """Convert ``range/azimuth/elevation`` in radians to FLU ``x/y/z``."""

    radius, azimuthArray, elevationArray = np.broadcast_arrays(
        np.asarray(range, dtype=float),
        np.asarray(azimuth, dtype=float),
        np.asarray(elevation, dtype=float),
    )
    horizontal = radius * np.cos(elevationArray)
    return np.column_stack(
        (
            horizontal * np.cos(azimuthArray),
            horizontal * np.sin(azimuthArray),
            radius * np.sin(elevationArray),
        )
    )


def cartesian_to_spherical(points: ArrayLike) -> NDArray[np.float64]:
    """Convert FLU points to columns ``range, azimuth, elevation``."""

    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    radius = np.linalg.norm(array, axis=1)
    horizontal = np.hypot(array[:, 0], array[:, 1])
    return np.column_stack(
        (
            radius,
            np.arctan2(array[:, 1], array[:, 0]),
            np.arctan2(array[:, 2], horizontal),
        )
    )


def transform_points(points: ArrayLike, transform: ArrayLike) -> NDArray[np.float64]:
    """Apply a 4x4 homogeneous transform to Cartesian points."""

    array = np.asarray(points, dtype=float)
    matrix = np.asarray(transform, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4).")
    homogeneous = np.column_stack((array, np.ones(array.shape[0])))
    result = homogeneous @ matrix.T
    if not np.allclose(result[:, 3], 1.0):
        result[:, :3] /= result[:, 3, None]
    return result[:, :3]


__all__ = ["cartesian_to_spherical", "spherical_to_cartesian", "transform_points"]
