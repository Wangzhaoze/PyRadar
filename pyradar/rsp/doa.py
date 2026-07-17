"""Geometry-aware direction-of-arrival estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

DoAMethod = Literal["auto", "fft", "bartlett", "capon", "music", "esprit"]

if TYPE_CHECKING:
    from pyradar.base.radar import Radar


@dataclass(frozen=True, slots=True)
class DoAResult:
    """Estimated directions and the search spectrum."""

    azimuth: NDArray[np.float64]
    elevation: NDArray[np.float64]
    power: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    azimuthAxis: NDArray[np.float64]
    elevationAxis: NDArray[np.float64]
    method: str


def steering_vector(
    arrayPositions: ArrayLike,
    wavelength: float,
    azimuth: ArrayLike,
    elevation: ArrayLike | float = 0.0,
) -> NDArray[np.complex128]:
    """Return far-field steering vectors for FLU array coordinates.

    ``azimuth`` and ``elevation`` are broadcast together. The returned shape is
    ``broadcast_shape + (numChannels,)``.
    """

    positions = np.asarray(arrayPositions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("arrayPositions must have shape (channels, 3).")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive.")
    azimuthArray, elevationArray = np.broadcast_arrays(
        np.asarray(azimuth, dtype=float), np.asarray(elevation, dtype=float)
    )
    direction = np.stack(
        (
            np.cos(elevationArray) * np.cos(azimuthArray),
            np.cos(elevationArray) * np.sin(azimuthArray),
            np.sin(elevationArray),
        ),
        axis=-1,
    )
    phase = (2.0 * np.pi / wavelength) * np.einsum(
        "...d,md->...m", direction, positions, optimize=True
    )
    return np.exp(1j * phase)


def spatial_covariance(
    signal: ArrayLike,
    *,
    channelAxis: int = -1,
    forwardBackward: bool = False,
) -> NDArray[np.complex128]:
    """Estimate the spatial covariance matrix from one or more snapshots."""

    array = np.asarray(signal)
    if array.ndim == 0:
        raise ValueError("signal must contain a channel dimension.")
    snapshots = np.moveaxis(array, channelAxis, -1).reshape(
        -1, array.shape[channelAxis]
    )
    covariance = snapshots.T @ snapshots.conj() / max(1, snapshots.shape[0])
    if forwardBackward:
        exchange = np.fliplr(np.eye(covariance.shape[0]))
        covariance = 0.5 * (covariance + exchange @ covariance.conj() @ exchange)
    return np.asarray(covariance, dtype=np.complex128)


def _search_vectors(
    arrayPositions: ArrayLike,
    wavelength: float,
    azimuthAxis: ArrayLike,
    elevationAxis: ArrayLike | None,
) -> tuple[NDArray[np.complex128], tuple[int, int]]:
    azimuth = np.asarray(azimuthAxis, dtype=float)
    elevation = np.asarray(
        [0.0] if elevationAxis is None else elevationAxis, dtype=float
    )
    azGrid, elGrid = np.meshgrid(azimuth, elevation)
    vectors = steering_vector(arrayPositions, wavelength, azGrid, elGrid)
    return vectors.reshape(-1, vectors.shape[-1]), azGrid.shape


def doa_bartlett(
    signal: ArrayLike,
    *,
    arrayPositions: ArrayLike,
    wavelength: float,
    azimuthAxis: ArrayLike,
    elevationAxis: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Conventional (Bartlett) beamforming spectrum."""

    covariance = spatial_covariance(signal)
    vectors, shape = _search_vectors(
        arrayPositions, wavelength, azimuthAxis, elevationAxis
    )
    spectrum = np.einsum(
        "gm,mn,gn->g", vectors.conj(), covariance, vectors, optimize=True
    ).real
    return np.maximum(spectrum, 0.0).reshape(shape)


def doa_capon(
    signal: ArrayLike,
    *,
    arrayPositions: ArrayLike,
    wavelength: float,
    azimuthAxis: ArrayLike,
    elevationAxis: ArrayLike | None = None,
    diagonalLoading: float = 1e-3,
) -> NDArray[np.float64]:
    """MVDR/Capon spatial spectrum."""

    covariance = spatial_covariance(signal)
    loading = (
        diagonalLoading
        * max(float(np.trace(covariance).real), 1.0)
        / covariance.shape[0]
    )
    inverse = np.linalg.pinv(covariance + loading * np.eye(covariance.shape[0]))
    vectors, shape = _search_vectors(
        arrayPositions, wavelength, azimuthAxis, elevationAxis
    )
    denominator = np.einsum(
        "gm,mn,gn->g", vectors.conj(), inverse, vectors, optimize=True
    ).real
    return (1.0 / np.maximum(denominator, np.finfo(float).tiny)).reshape(shape)


def doa_music(
    signal: ArrayLike,
    *,
    arrayPositions: ArrayLike,
    wavelength: float,
    azimuthAxis: ArrayLike,
    elevationAxis: ArrayLike | None = None,
    numSources: int = 1,
    forwardBackward: bool = False,
) -> NDArray[np.float64]:
    """MUSIC pseudospectrum for arbitrary array geometry."""

    covariance = spatial_covariance(signal, forwardBackward=forwardBackward)
    channels = covariance.shape[0]
    if not 1 <= numSources < channels:
        raise ValueError("numSources must be in [1, numChannels).")
    _, eigenvectors = np.linalg.eigh(covariance)
    noiseSubspace = eigenvectors[:, : channels - numSources]
    vectors, shape = _search_vectors(
        arrayPositions, wavelength, azimuthAxis, elevationAxis
    )
    projection = vectors.conj() @ noiseSubspace
    denominator = np.sum(np.abs(projection) ** 2, axis=1)
    return (1.0 / np.maximum(denominator, np.finfo(float).tiny)).reshape(shape)


def spatial_smoothing(
    signal: ArrayLike,
    *,
    subarraySize: int,
    forwardBackward: bool = True,
) -> NDArray[np.complex128]:
    """Spatially smooth ULA snapshots for coherent-source estimation."""

    array = np.asarray(signal)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("signal must have shape (snapshots, channels).")
    channels = array.shape[1]
    if not 1 < subarraySize <= channels:
        raise ValueError("subarraySize must lie in [2, channels].")
    covariance = np.zeros((subarraySize, subarraySize), dtype=np.complex128)
    count = channels - subarraySize + 1
    for start in range(count):
        covariance += spatial_covariance(array[:, start : start + subarraySize])
    covariance /= count
    if forwardBackward:
        exchange = np.fliplr(np.eye(subarraySize))
        covariance = np.asarray(
            0.5 * (covariance + exchange @ covariance.conj() @ exchange),
            dtype=np.complex128,
        )
    return covariance


def doa_esprit(
    signal: ArrayLike,
    *,
    numSources: int,
    spacing: float,
    wavelength: float,
) -> NDArray[np.float64]:
    """Estimate ULA azimuths with rotational-invariance ESPRIT."""

    covariance = spatial_covariance(signal, forwardBackward=True)
    channels = covariance.shape[0]
    if not 1 <= numSources < channels or spacing <= 0.0 or wavelength <= 0.0:
        raise ValueError("Invalid ESPRIT dimensions or physical spacing.")
    _, eigenvectors = np.linalg.eigh(covariance)
    signalSubspace = eigenvectors[:, -numSources:]
    rotation = np.linalg.pinv(signalSubspace[:-1]) @ signalSubspace[1:]
    phase = np.angle(np.linalg.eigvals(rotation))
    angles = np.arcsin(np.clip(phase * wavelength / (2.0 * np.pi * spacing), -1.0, 1.0))
    return np.sort(angles.real)


def combine_duplicate_channels(
    signal: ArrayLike,
    positions: ArrayLike,
    *,
    policy: Literal["first", "noncoherent", "coherent"] = "coherent",
    tolerance: float = 1e-9,
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Combine repeated virtual phase centers according to ``policy``."""

    data = np.asarray(signal)
    locations = np.asarray(positions, dtype=float)
    if data.shape[-1] != locations.shape[0]:
        raise ValueError("Signal channel count does not match positions.")
    keys = np.rint(locations / tolerance).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groupKey = (int(key[0]), int(key[1]), int(key[2]))
        groups.setdefault(groupKey, []).append(index)
    combined = []
    uniquePositions = []
    for indices in groups.values():
        values = data[..., indices]
        if policy == "first":
            value = values[..., 0]
        elif policy == "coherent":
            value = np.mean(values, axis=-1)
        elif policy == "noncoherent":
            amplitude = np.sqrt(np.mean(np.abs(values) ** 2, axis=-1))
            value = amplitude * np.exp(1j * np.angle(values[..., 0]))
        else:
            raise ValueError(f"Unknown duplicate policy {policy!r}.")
        combined.append(value)
        uniquePositions.append(np.mean(locations[indices], axis=0))
    return np.stack(combined, axis=-1), np.asarray(uniquePositions)


def _geometry(
    positions: NDArray[np.float64], tolerance: float
) -> tuple[str, float | None, float | None]:
    y = np.unique(np.rint(positions[:, 1] / tolerance).astype(np.int64)) * tolerance
    z = np.unique(np.rint(positions[:, 2] / tolerance).astype(np.int64)) * tolerance

    def uniform(values: NDArray[np.float64]) -> float | None:
        if values.size < 2:
            return None
        differences = np.diff(np.sort(values))
        return (
            float(differences[0]) if np.allclose(differences, differences[0]) else None
        )

    dy, dz = uniform(y), uniform(z)
    keys = np.unique(np.rint(positions[:, 1:3] / tolerance).astype(np.int64), axis=0)
    if z.size == 1 and dy is not None:
        return "ula", dy, None
    if keys.shape[0] == y.size * z.size and dy is not None and dz is not None:
        return "ura", dy, dz
    return "sparse", None, None


def _peaks(
    spectrum: NDArray[np.float64],
    azimuthAxis: NDArray[np.float64],
    elevationAxis: NDArray[np.float64],
    count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    maxima = spectrum >= ndimage.maximum_filter(spectrum, size=3, mode="nearest")
    indices = np.argwhere(maxima)
    if indices.size == 0:
        indices = np.asarray([np.unravel_index(np.argmax(spectrum), spectrum.shape)])
    powers = spectrum[indices[:, 0], indices[:, 1]]
    order = np.argsort(powers, kind="stable")[::-1][:count]
    indices = indices[order]
    return (
        azimuthAxis[indices[:, 1]],
        elevationAxis[indices[:, 0]],
        powers[order],
    )


def estimate_doa(
    signal: ArrayLike,
    *,
    radar: Radar,
    method: DoAMethod | None = None,
    numSources: int | None = None,
) -> DoAResult:
    """Select and run a model-aware DoA method for one RD cell or snapshots."""

    config = radar.processing.doa
    selected = method or config.method
    count = numSources or config.numSources
    data, positions = combine_duplicate_channels(
        signal,
        radar.virtualArray,
        policy=config.duplicatePolicy,
        tolerance=max(radar.wavelength * 1e-6, 1e-12),
    )
    tolerance = max(radar.wavelength * 1e-6, 1e-12)
    geometry, dy, dz = _geometry(positions, tolerance)
    if selected == "auto":
        selected = "fft" if geometry in {"ula", "ura"} else "bartlett"
    azimuthAxis = np.linspace(*config.azimuthFov, config.azimuthBins)
    elevationAxis = np.linspace(*config.elevationFov, config.elevationBins)
    snapshots = data[None, :] if data.ndim == 1 else data.reshape(-1, data.shape[-1])

    if selected == "fft" and geometry == "ula":
        if dy is None:
            raise RuntimeError("ULA geometry did not provide an element spacing.")
        order = np.argsort(positions[:, 1])
        size = radar.processing.fft.azimuthFftSize
        weights = np.hanning(order.size)
        transform = np.fft.fftshift(
            np.fft.fft(snapshots[:, order] * weights, n=size, axis=1), axes=1
        )
        spectrum = np.sum(np.abs(transform) ** 2, axis=0, keepdims=True)
        spatial = (np.arange(size, dtype=float) - size // 2) / size
        azimuthAxis = np.arcsin(
            np.clip(spatial * radar.wavelength / float(dy), -1.0, 1.0)
        )
        elevationAxis = np.asarray([0.0])
    elif selected == "fft" and geometry == "ura":
        if dy is None or dz is None:
            raise RuntimeError("URA geometry did not provide both element spacings.")
        yValues = np.sort(np.unique(positions[:, 1]))
        zValues = np.sort(np.unique(positions[:, 2]))
        grid = np.empty((snapshots.shape[0], zValues.size, yValues.size), dtype=complex)
        for channel, position in enumerate(positions):
            yIndex = int(np.argmin(np.abs(yValues - position[1])))
            zIndex = int(np.argmin(np.abs(zValues - position[2])))
            grid[:, zIndex, yIndex] = snapshots[:, channel]
        shape = (
            radar.processing.fft.elevationFftSize,
            radar.processing.fft.azimuthFftSize,
        )
        transform = np.fft.fftshift(
            np.fft.fft2(grid, s=shape, axes=(-2, -1)), axes=(-2, -1)
        )
        spectrum = np.sum(np.abs(transform) ** 2, axis=0)
        uy = (
            (np.arange(shape[1]) - shape[1] // 2)
            / shape[1]
            * radar.wavelength
            / float(dy)
        )
        uz = (
            (np.arange(shape[0]) - shape[0] // 2)
            / shape[0]
            * radar.wavelength
            / float(dz)
        )
        azimuthAxis = np.arcsin(np.clip(uy, -1.0, 1.0))
        elevationAxis = np.arcsin(np.clip(uz, -1.0, 1.0))
    elif selected == "esprit":
        if geometry != "ula" or dy is None:
            raise ValueError("ESPRIT requires a uniform linear array.")
        azimuth = doa_esprit(
            snapshots, numSources=count, spacing=dy, wavelength=radar.wavelength
        )
        spectrum = np.zeros((1, azimuthAxis.size), dtype=float)
        bins = np.asarray([np.argmin(np.abs(azimuthAxis - item)) for item in azimuth])
        spectrum[0, bins] = 1.0
        elevationAxis = np.asarray([0.0])
    else:
        if selected == "bartlett":
            spectrum = doa_bartlett(
                snapshots,
                arrayPositions=positions,
                wavelength=radar.wavelength,
                azimuthAxis=azimuthAxis,
                elevationAxis=elevationAxis,
            )
        elif selected == "capon":
            spectrum = doa_capon(
                snapshots,
                arrayPositions=positions,
                wavelength=radar.wavelength,
                azimuthAxis=azimuthAxis,
                elevationAxis=elevationAxis,
                diagonalLoading=config.diagonalLoading,
            )
        elif selected == "music":
            spectrum = doa_music(
                snapshots,
                arrayPositions=positions,
                wavelength=radar.wavelength,
                azimuthAxis=azimuthAxis,
                elevationAxis=elevationAxis,
                numSources=count,
            )
        else:
            raise ValueError(
                f"Unsupported DoA method {selected!r} for {geometry} geometry."
            )

    spectrum = np.asarray(spectrum, dtype=float)
    azimuth, elevation, power = _peaks(spectrum, azimuthAxis, elevationAxis, count)
    return DoAResult(
        azimuth=azimuth,
        elevation=elevation,
        power=power,
        spectrum=spectrum,
        azimuthAxis=azimuthAxis,
        elevationAxis=elevationAxis,
        method=selected,
    )


__all__ = [
    "DoAResult",
    "combine_duplicate_channels",
    "doa_bartlett",
    "doa_capon",
    "doa_esprit",
    "doa_music",
    "estimate_doa",
    "spatial_covariance",
    "spatial_smoothing",
    "steering_vector",
]
