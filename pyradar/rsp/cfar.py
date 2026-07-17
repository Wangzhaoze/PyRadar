"""Statistically parameterized CFAR detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.optimize import brentq

CFARMethod = Literal["ca", "goca", "soca", "os"]


@dataclass(frozen=True, slots=True)
class CFARResult:
    """Full-size CFAR products."""

    detections: NDArray[np.bool_]
    noise: NDArray[np.float64]
    threshold: NDArray[np.float64]
    snr: NDArray[np.float64]


def _ca_scale(numTraining: int, pfa: float) -> float:
    return numTraining * (pfa ** (-1.0 / numTraining) - 1.0)


def _os_scale(numTraining: int, rank: int, pfa: float) -> float:
    # Laplace transform of the kth exponential order statistic.
    def probability(scale: float) -> float:
        indices = np.arange(rank + 1, dtype=float)
        rates = numTraining - indices
        return float(np.prod(rates / (rates + scale)))

    upper = max(1.0, -np.log(pfa))
    while probability(upper) > pfa:
        upper *= 2.0
    return float(brentq(lambda value: probability(value) - pfa, 0.0, upper))


def _validate_power(power: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(power, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("CFAR input must be finite, nonnegative power.")
    return array


def _finish(
    power: NDArray[np.float64],
    noise: NDArray[np.float64],
    threshold: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    minSnrDb: float,
    peakGrouping: bool,
    footprint: NDArray[np.bool_],
) -> CFARResult:
    floor = np.finfo(float).tiny
    numericalFloor = max(float(np.max(power)), 1.0) * np.finfo(float).eps * 64.0
    threshold = np.maximum(threshold, numericalFloor)
    snr = 10.0 * np.log10(np.maximum(power, floor) / np.maximum(noise, floor))
    detections = valid & (power > threshold) & (snr >= minSnrDb)
    if peakGrouping:
        localMax = ndimage.maximum_filter(power, footprint=footprint, mode="nearest")
        detections &= power >= localMax
    noise = np.where(valid, noise, np.nan)
    threshold = np.where(valid, threshold, np.nan)
    snr = np.where(valid, snr, np.nan)
    return CFARResult(detections=detections, noise=noise, threshold=threshold, snr=snr)


def cfar_1d(
    power: ArrayLike,
    *,
    method: CFARMethod = "ca",
    trainingCells: int = 8,
    guardCells: int = 2,
    pfa: float = 1e-4,
    rankFraction: float = 0.75,
    axis: int = -1,
    minSnrDb: float = 0.0,
    peakGrouping: bool = False,
    wrap: bool = False,
) -> CFARResult:
    """One-dimensional CA/GOCA/SOCA/OS-CFAR."""

    array = _validate_power(power)
    if trainingCells < 1 or guardCells < 0 or not 0.0 < pfa < 1.0:
        raise ValueError("Invalid CFAR cell counts or pfa.")
    axis %= array.ndim
    width = trainingCells + guardCells
    kernel = np.ones(2 * width + 1, dtype=float)
    kernel[width - guardCells : width + guardCells + 1] = 0.0
    numTraining = int(kernel.sum())
    mode = "wrap" if wrap else "constant"
    if method == "ca":
        noise = ndimage.convolve1d(array, kernel, axis=axis, mode=mode) / numTraining
        scale = _ca_scale(numTraining, pfa)
    elif method in {"goca", "soca"}:
        leading = np.zeros_like(kernel)
        trailing = np.zeros_like(kernel)
        leading[:trainingCells] = 1.0
        trailing[-trainingCells:] = 1.0
        left = ndimage.convolve1d(array, leading, axis=axis, mode=mode) / trainingCells
        right = (
            ndimage.convolve1d(array, trailing, axis=axis, mode=mode) / trainingCells
        )
        noise = np.maximum(left, right) if method == "goca" else np.minimum(left, right)
        scale = _ca_scale(trainingCells, pfa)
    elif method == "os":
        rank = int(np.ceil(rankFraction * numTraining)) - 1
        footprint = kernel.astype(bool)
        footprintShape = [1] * array.ndim
        footprintShape[axis] = footprint.size
        noise = ndimage.rank_filter(
            array,
            rank=rank,
            footprint=footprint.reshape(footprintShape),
            mode=mode,
        )
        scale = _os_scale(numTraining, rank, pfa)
    else:
        raise ValueError(f"Unsupported CFAR method {method!r}.")
    valid = np.ones_like(array, dtype=bool)
    if not wrap:
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(0, width)
        valid[tuple(selection)] = False
        selection[axis] = slice(array.shape[axis] - width, None)
        valid[tuple(selection)] = False
    peakFootprint = np.ones(3, dtype=bool)
    peakShape = [1] * array.ndim
    peakShape[axis] = 3
    return _finish(
        array,
        noise,
        noise * scale,
        valid,
        minSnrDb=minSnrDb,
        peakGrouping=peakGrouping,
        footprint=peakFootprint.reshape(peakShape),
    )


def cfar_2d(
    power: ArrayLike,
    *,
    method: CFARMethod = "ca",
    trainingCells: tuple[int, int] = (8, 4),
    guardCells: tuple[int, int] = (2, 1),
    pfa: float = 1e-4,
    rankFraction: float = 0.75,
    minSnrDb: float = 0.0,
    peakGrouping: bool = True,
) -> CFARResult:
    """Two-dimensional CFAR over a range-Doppler power map.

    Doppler is treated as circular. Range edge cells without a complete
    training set are marked invalid.
    """

    array = _validate_power(power)
    if array.ndim != 2:
        raise ValueError("cfar_2d expects a two-dimensional power map.")
    tr, td = (int(value) for value in trainingCells)
    gr, gd = (int(value) for value in guardCells)
    if tr < 1 or td < 1 or gr < 0 or gd < 0 or not 0.0 < pfa < 1.0:
        raise ValueError("Invalid CFAR cell counts or pfa.")
    hr, hd = tr + gr, td + gd
    footprint = np.ones((2 * hr + 1, 2 * hd + 1), dtype=bool)
    footprint[hr - gr : hr + gr + 1, hd - gd : hd + gd + 1] = False
    numTraining = int(np.count_nonzero(footprint))
    if method == "ca":
        noise = (
            ndimage.convolve(array, footprint.astype(float), mode="wrap") / numTraining
        )
        scale = _ca_scale(numTraining, pfa)
    elif method in {"goca", "soca"}:
        leading = footprint.copy()
        leading[hr:, :] = False
        trailing = footprint.copy()
        trailing[: hr + 1, :] = False
        count = int(np.count_nonzero(leading))
        left = ndimage.convolve(array, leading.astype(float), mode="wrap") / count
        right = ndimage.convolve(array, trailing.astype(float), mode="wrap") / count
        noise = np.maximum(left, right) if method == "goca" else np.minimum(left, right)
        scale = _ca_scale(count, pfa)
    elif method == "os":
        rank = int(np.ceil(rankFraction * numTraining)) - 1
        noise = ndimage.rank_filter(array, rank=rank, footprint=footprint, mode="wrap")
        scale = _os_scale(numTraining, rank, pfa)
    else:
        raise ValueError(f"Unsupported CFAR method {method!r}.")
    valid = np.ones_like(array, dtype=bool)
    valid[:hr] = False
    valid[-hr:] = False
    return _finish(
        array,
        noise,
        noise * scale,
        valid,
        minSnrDb=minSnrDb,
        peakGrouping=peakGrouping,
        footprint=np.ones((3, 3), dtype=bool),
    )


def ca_cfar_1d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_1d(power, method="ca", **kwargs)


def goca_cfar_1d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_1d(power, method="goca", **kwargs)


def soca_cfar_1d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_1d(power, method="soca", **kwargs)


def os_cfar_1d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_1d(power, method="os", **kwargs)


def ca_cfar_2d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_2d(power, method="ca", **kwargs)


def goca_cfar_2d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_2d(power, method="goca", **kwargs)


def soca_cfar_2d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_2d(power, method="soca", **kwargs)


def os_cfar_2d(power: ArrayLike, **kwargs) -> CFARResult:
    return cfar_2d(power, method="os", **kwargs)


__all__ = [
    "CFARResult",
    "ca_cfar_1d",
    "ca_cfar_2d",
    "cfar_1d",
    "cfar_2d",
    "goca_cfar_1d",
    "goca_cfar_2d",
    "os_cfar_1d",
    "os_cfar_2d",
    "soca_cfar_1d",
    "soca_cfar_2d",
]
