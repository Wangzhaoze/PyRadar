"""Unit-preserving radar-domain conversions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base.waveform import SPEED_OF_LIGHT


def range_resolution(*, bandwidth: float) -> float:
    """Return ideal FMCW range resolution in meters."""

    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")
    return SPEED_OF_LIGHT / (2.0 * bandwidth)


def max_unambiguous_range(
    *, sampleRate: float, slope: float, complexSampling: bool = True
) -> float:
    """Return the beat-frequency-limited unambiguous range in meters."""

    if (
        not np.isfinite(sampleRate)
        or sampleRate <= 0.0
        or not np.isfinite(slope)
        or slope <= 0.0
    ):
        raise ValueError("sampleRate and slope must be positive.")
    bandwidth = sampleRate if complexSampling else sampleRate / 2.0
    return SPEED_OF_LIGHT * bandwidth / (2.0 * slope)


def velocity_resolution(
    *, wavelength: float, numSlowTimeSamples: int, slowTimeInterval: float
) -> float:
    """Return monostatic radial-velocity resolution in meters per second."""

    if (
        not np.isfinite(wavelength)
        or wavelength <= 0.0
        or numSlowTimeSamples < 1
        or not np.isfinite(slowTimeInterval)
        or slowTimeInterval <= 0.0
    ):
        raise ValueError("wavelength, sample count, and interval must be positive.")
    return wavelength / (2.0 * numSlowTimeSamples * slowTimeInterval)


def max_unambiguous_velocity(*, wavelength: float, slowTimeInterval: float) -> float:
    """Return one-sided monostatic unambiguous radial velocity."""

    if (
        not np.isfinite(wavelength)
        or wavelength <= 0.0
        or not np.isfinite(slowTimeInterval)
        or slowTimeInterval <= 0.0
    ):
        raise ValueError("wavelength and slowTimeInterval must be positive.")
    return wavelength / (4.0 * slowTimeInterval)


def range_axis(*, fftSize: int, sampleRate: float, slope: float) -> NDArray[np.float64]:
    """Return unshifted FMCW range-bin centers in meters."""

    if fftSize < 1:
        raise ValueError("fftSize must be positive.")
    maximum = max_unambiguous_range(sampleRate=sampleRate, slope=slope)
    return np.arange(fftSize, dtype=float) * maximum / fftSize


def velocity_axis(
    *, fftSize: int, wavelength: float, slowTimeInterval: float
) -> NDArray[np.float64]:
    """Return the centered Doppler velocity coordinate in meters per second."""

    if fftSize < 1:
        raise ValueError("fftSize must be positive.")
    binSize = velocity_resolution(
        wavelength=wavelength,
        numSlowTimeSamples=fftSize,
        slowTimeInterval=slowTimeInterval,
    )
    return (np.arange(fftSize, dtype=float) - fftSize // 2) * binSize


def ula_unambiguous_fov(*, wavelength: float, spacing: float) -> tuple[float, float]:
    """Return a ULA's alias-free broadside field of view in radians."""

    if (
        not np.isfinite(wavelength)
        or wavelength <= 0.0
        or not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError("wavelength and spacing must be finite and positive.")
    limit = float(np.arcsin(min(1.0, wavelength / (2.0 * spacing))))
    return -limit, limit


def ula_angle_axis(
    *, fftSize: int, wavelength: float, spacing: float
) -> NDArray[np.float64]:
    """Map a centered spatial FFT coordinate to ULA broadside angles."""

    if fftSize < 1:
        raise ValueError("fftSize must be positive.")
    ula_unambiguous_fov(wavelength=wavelength, spacing=spacing)
    spatialFrequency = (np.arange(fftSize, dtype=float) - fftSize // 2) / fftSize
    directionCosine = spatialFrequency * wavelength / spacing
    return np.arcsin(np.clip(directionCosine, -1.0, 1.0))


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

    if not np.isfinite(slope) or slope <= 0.0:
        raise ValueError("slope must be finite and positive.")
    frequency = np.asarray(beatFrequency, dtype=float)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("beatFrequency must be finite.")
    return frequency * SPEED_OF_LIGHT / (2.0 * slope)


def doppler_frequency_to_velocity(
    dopplerFrequency: ArrayLike, *, wavelength: float
) -> NDArray[np.float64]:
    """Convert monostatic Doppler frequency in Hz to radial velocity in m/s."""

    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be finite and positive.")
    frequency = np.asarray(dopplerFrequency, dtype=float)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("dopplerFrequency must be finite.")
    return frequency * wavelength / 2.0


__all__ = [
    "beat_frequency_to_range",
    "db_to_power",
    "doppler_frequency_to_velocity",
    "magnitude_to_db",
    "max_unambiguous_range",
    "max_unambiguous_velocity",
    "power_to_db",
    "range_axis",
    "range_resolution",
    "ula_angle_axis",
    "ula_unambiguous_fov",
    "velocity_axis",
    "velocity_resolution",
]
