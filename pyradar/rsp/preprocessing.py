"""Small composable operations used before detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def remove_static_clutter(values: ArrayLike, *, slowTimeAxis: int = 0) -> NDArray:
    """Suppress zero-Doppler clutter by subtracting the coherent mean."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain a slow-time axis.")
    axis = slowTimeAxis % array.ndim
    return array - np.mean(array, axis=axis, keepdims=True)


def coherent_integrate(values: ArrayLike, *, axis: int = 0) -> NDArray:
    """Coherently sum complex samples along one axis."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain an integration axis.")
    return np.sum(array, axis=axis % array.ndim)


def noncoherent_integrate(
    values: ArrayLike, *, axis: int = 0, power: bool = True
) -> NDArray[np.float64]:
    """Sum magnitudes or powers along one axis."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain an integration axis.")
    magnitude = np.abs(array)
    return np.sum(magnitude**2 if power else magnitude, axis=axis % array.ndim)


__all__ = [
    "coherent_integrate",
    "noncoherent_integrate",
    "remove_static_clutter",
]
