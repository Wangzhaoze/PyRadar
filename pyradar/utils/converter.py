"""Unit-preserving radar-domain conversions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base.waveform import SPEED_OF_LIGHT


def power_to_db(power: ArrayLike, *, floor: float = 1e-12) -> NDArray[np.float64]:
    """Convert nonnegative linear power to dB."""

    if floor <= 0.0:
        raise ValueError("floor must be positive.")
    array = np.asarray(power, dtype=float)
    if np.any(array < 0.0):
        raise ValueError("Power cannot be negative.")
    return 10.0 * np.log10(np.maximum(array, floor))


def db_to_power(db: ArrayLike) -> NDArray[np.float64]:
    """Convert dB to linear power."""

    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


def magnitude_to_db(
    magnitude: ArrayLike, *, floor: float = 1e-12
) -> NDArray[np.float64]:
    """Convert nonnegative linear magnitude to dB."""

    if floor <= 0.0:
        raise ValueError("floor must be positive.")
    array = np.asarray(magnitude, dtype=float)
    if np.any(array < 0.0):
        raise ValueError("Magnitude cannot be negative.")
    return 20.0 * np.log10(np.maximum(array, floor))


def beat_frequency_to_range(
    beatFrequency: ArrayLike, *, slope: float
) -> NDArray[np.float64]:
    """Convert FMCW beat frequency in Hz to range in meters."""

    if slope <= 0.0:
        raise ValueError("slope must be positive.")
    return np.asarray(beatFrequency, dtype=float) * SPEED_OF_LIGHT / (2.0 * slope)


def doppler_frequency_to_velocity(
    dopplerFrequency: ArrayLike, *, wavelength: float
) -> NDArray[np.float64]:
    """Convert monostatic Doppler frequency in Hz to radial velocity in m/s."""

    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive.")
    return np.asarray(dopplerFrequency, dtype=float) * wavelength / 2.0


__all__ = [
    "beat_frequency_to_range",
    "db_to_power",
    "doppler_frequency_to_velocity",
    "magnitude_to_db",
    "power_to_db",
]
