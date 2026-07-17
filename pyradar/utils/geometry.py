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
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite.")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix.")
    homogeneous = np.column_stack((array, np.ones(array.shape[0])))
    result = homogeneous @ matrix.T
    if not np.allclose(result[:, 3], 1.0):
        result[:, :3] /= result[:, 3, None]
    return result[:, :3]


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Build a 4x4 rigid transform from a rotation and translation."""

    rotationArray = np.asarray(rotation, dtype=float)
    translationArray = np.asarray(translation, dtype=float)
    if rotationArray.shape != (3, 3) or translationArray.shape != (3,):
        raise ValueError("rotation and translation must have shapes (3, 3) and (3,).")
    if not np.all(np.isfinite(rotationArray)) or not np.all(
        np.isfinite(translationArray)
    ):
        raise ValueError("rotation and translation must be finite.")
    if not np.allclose(rotationArray.T @ rotationArray, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotationArray), 1.0, atol=1e-8):
        raise ValueError("rotation must be right handed.")
    transform = np.eye(4)
    transform[:3, :3] = rotationArray
    transform[:3, 3] = translationArray
    return transform


def inverse_transform(transform: ArrayLike) -> NDArray[np.float64]:
    """Invert a rigid 4x4 transform without a general matrix inversion."""

    matrix = np.asarray(transform, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0))
    ):
        raise ValueError("transform must be a rigid homogeneous 4x4 matrix.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("transform rotation must be orthonormal.")
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ matrix[:3, 3]
    return result


def compose_transforms(*transforms: ArrayLike) -> NDArray[np.float64]:
    """Compose transforms in application order from left to right."""

    result = np.eye(4)
    for value in transforms:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every transform must be a finite 4x4 matrix.")
        result = matrix @ result
    return result


__all__ = [
    "cartesian_to_spherical",
    "compose_transforms",
    "inverse_transform",
    "make_transform",
    "spherical_to_cartesian",
    "transform_points",
]
