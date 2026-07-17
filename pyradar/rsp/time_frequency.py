"""Short-time and zoomed spectral analysis for radar signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal

from .fft import window as make_window

if TYPE_CHECKING:
    from pyradar.base import Radar


@dataclass(frozen=True, slots=True)
class TimeFrequencyResult:
    """Complex STFT with frequency and segment-center coordinates."""

    spectrum: NDArray[np.complex128]
    frequency: NDArray[np.float64]
    time: NDArray[np.float64]
    velocity: NDArray[np.float64] | None = None

    @property
    def power(self) -> NDArray[np.float64]:
        """Linear spectrogram power."""

        return np.asarray(np.abs(self.spectrum) ** 2, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ZoomFFTResult:
    """Spectrum evaluated on a requested uniform frequency interval."""

    spectrum: NDArray[np.complex128]
    frequency: NDArray[np.float64]
    axis: int


def stft(
    values: ArrayLike,
    *,
    sampleRate: float,
    segmentLength: int = 128,
    overlap: int | None = None,
    fftSize: int | None = None,
    window: str = "hann",
    axis: int = -1,
    detrend: bool = False,
    centered: bool = True,
) -> TimeFrequencyResult:
    """Compute a two-sided STFT with output shape ``(..., frequency, time)``."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain a sampled axis.")
    axis %= array.ndim
    if not np.isfinite(sampleRate) or sampleRate <= 0.0:
        raise ValueError("sampleRate must be finite and positive.")
    if segmentLength < 2 or segmentLength > array.shape[axis]:
        raise ValueError("segmentLength must be in [2, signal length].")
    overlapValue = segmentLength // 2 if overlap is None else int(overlap)
    if not 0 <= overlapValue < segmentLength:
        raise ValueError("overlap must be smaller than segmentLength.")
    size = segmentLength if fftSize is None else int(fftSize)
    if size < segmentLength:
        raise ValueError("fftSize cannot be smaller than segmentLength.")
    work = np.moveaxis(array, axis, -1)
    frequency, time, spectrum = scipy_signal.stft(
        work,
        fs=sampleRate,
        window=make_window(segmentLength, window),
        nperseg=segmentLength,
        noverlap=overlapValue,
        nfft=size,
        detrend=detrend,
        return_onesided=False,
        boundary=None,
        padded=False,
        axis=-1,
    )
    if centered:
        frequency = np.fft.fftshift(frequency)
        spectrum = np.fft.fftshift(spectrum, axes=-2)
    return TimeFrequencyResult(
        spectrum=np.asarray(spectrum, dtype=np.complex128),
        frequency=np.asarray(frequency, dtype=np.float64),
        time=np.asarray(time, dtype=np.float64),
    )


def micro_doppler_spectrogram(
    values: ArrayLike,
    *,
    radar: Radar | None = None,
    sampleRate: float | None = None,
    segmentLength: int = 128,
    overlap: int | None = None,
    fftSize: int | None = None,
    window: str = "hann",
    axis: int = -1,
    detrend: bool = False,
) -> TimeFrequencyResult:
    """Compute slow-time micro-Doppler and an optional velocity coordinate."""

    rate = sampleRate
    if rate is None:
        if radar is None:
            raise ValueError("Provide sampleRate or a Radar model.")
        rate = 1.0 / radar.slowTimeInterval
    result = stft(
        values,
        sampleRate=rate,
        segmentLength=segmentLength,
        overlap=overlap,
        fftSize=fftSize,
        window=window,
        axis=axis,
        detrend=detrend,
        centered=True,
    )
    velocity = None if radar is None else result.frequency * radar.wavelength / 2.0
    return TimeFrequencyResult(
        spectrum=result.spectrum,
        frequency=result.frequency,
        time=result.time,
        velocity=velocity,
    )


def zoom_fft(
    values: ArrayLike,
    *,
    frequencyRange: tuple[float, float],
    sampleRate: float,
    fftSize: int,
    axis: int = -1,
    window: str = "rectangular",
    endpoint: bool = False,
) -> ZoomFFTResult:
    """Evaluate the DFT only over ``frequencyRange`` using Bluestein's CZT."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must contain a sampled axis.")
    axis %= array.ndim
    low, high = (float(value) for value in frequencyRange)
    if (
        not np.isfinite(sampleRate)
        or sampleRate <= 0.0
        or fftSize < 1
        or not 0.0 <= low < high <= sampleRate
    ):
        raise ValueError(
            "Require positive sampleRate/fftSize and 0 <= low < high <= sampleRate."
        )
    weights = make_window(array.shape[axis], window)
    shape = [1] * array.ndim
    shape[axis] = weights.size
    spectrum = scipy_signal.zoom_fft(
        array * weights.reshape(shape),
        (low, high),
        m=fftSize,
        fs=sampleRate,
        endpoint=endpoint,
        axis=axis,
    )
    frequency = np.linspace(low, high, fftSize, endpoint=endpoint)
    return ZoomFFTResult(
        spectrum=np.asarray(spectrum, dtype=np.complex128),
        frequency=np.asarray(frequency, dtype=np.float64),
        axis=axis,
    )


__all__ = [
    "TimeFrequencyResult",
    "ZoomFFTResult",
    "micro_doppler_spectrogram",
    "stft",
    "zoom_fft",
]
