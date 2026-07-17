"""Windowing and FFT operations for FMCW radar data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window

from pyradar.base.cube import ADCFrame, RadarCube


def window(length: int, name: str = "hann") -> NDArray[np.float64]:
    """Return a periodicity-independent symmetric processing window."""

    if length < 1:
        raise ValueError("Window length must be positive.")
    normalized = name.lower()
    if normalized in {"rectangular", "rect", "boxcar", "none"}:
        return np.ones(length, dtype=float)
    supported = {"hann", "hamming", "blackman", "blackmanharris"}
    if normalized not in supported:
        raise ValueError(
            f"Unsupported window {name!r}; choose one of {sorted(supported)}."
        )
    return np.asarray(get_window(normalized, length, fftbins=False), dtype=float)


def _input(
    signal: ArrayLike | RadarCube | ADCFrame,
    radar: Any = None,
) -> tuple[NDArray[Any], Any, tuple[str, ...] | None, RadarCube | None]:
    if isinstance(signal, RadarCube):
        return signal.data, radar or signal.radar, signal.dims, signal
    if isinstance(signal, ADCFrame):
        model = radar or signal.radar
        frame = signal.canonical(model) if model is not None else signal
        return frame.data, model, frame.dims, None
    return np.asarray(signal), radar, None, None


def _axis(
    dims: tuple[str, ...] | None, names: Sequence[str], explicit: int, ndim: int
) -> int:
    if dims is not None:
        for name in names:
            if name in dims:
                return dims.index(name)
    axis = explicit % ndim
    return axis


def _transform(
    data: NDArray[Any],
    *,
    axis: int,
    fftSize: int | None,
    windowName: str,
    removeMean: bool,
    shift: bool,
) -> NDArray[np.complex128]:
    if fftSize is not None and fftSize < 1:
        raise ValueError("fftSize must be positive.")
    work = np.asarray(data)
    if removeMean:
        work = work - np.mean(work, axis=axis, keepdims=True)
    weights = window(work.shape[axis], windowName)
    shape = [1] * work.ndim
    shape[axis] = weights.size
    result = np.fft.fft(work * weights.reshape(shape), n=fftSize, axis=axis)
    return np.fft.fftshift(result, axes=axis) if shift else result


def _cube(
    data: NDArray[Any],
    *,
    radar: Any,
    source: RadarCube | None,
    dims: Sequence[str],
    stage: str,
    coords: dict[str, NDArray[Any]] | None = None,
) -> RadarCube | NDArray[Any]:
    if radar is None and source is None:
        return data
    metadata = source.metadata if source is not None else {}
    return RadarCube(
        data=data,
        dims=tuple(dims),
        stage=stage,
        radar=radar,
        coords=coords or {},
        metadata=metadata,
    )


def range_fft(
    adc: ArrayLike | RadarCube | ADCFrame,
    *,
    radar: Any = None,
    fftSize: int | None = None,
    sampleAxis: int = -1,
    window: str | None = None,
    removeMean: bool | None = None,
) -> RadarCube | NDArray[Any]:
    """Apply the fast-time FFT.

    Explicit keyword arguments override the attached Radar defaults. Raw
    ndarray input without a Radar returns an ndarray; model-aware input returns
    a :class:`~pyradar.base.RadarCube`.
    """

    data, model, dims, source = _input(adc, radar)
    axis = _axis(dims, ("sample", "range"), sampleAxis, data.ndim)
    config = model.processing.fft if model is not None else None
    size = fftSize if fftSize is not None else (config.rangeFftSize if config else None)
    windowName = (
        window if window is not None else (config.rangeWindow if config else "hann")
    )
    subtract = (
        removeMean
        if removeMean is not None
        else (config.removeRangeMean if config else False)
    )
    result = _transform(
        data,
        axis=axis,
        fftSize=size,
        windowName=windowName,
        removeMean=subtract,
        shift=False,
    )
    outputDims = list(dims or tuple(f"axis{index}" for index in range(data.ndim)))
    outputDims[axis] = "range"
    coords: dict[str, NDArray[Any]] = {}
    if model is not None:
        coords["range"] = np.arange(result.shape[axis]) * model.rangeBinSize
    return _cube(
        result,
        radar=model,
        source=source,
        dims=outputDims,
        stage="range",
        coords=coords,
    )


def doppler_fft(
    signal: ArrayLike | RadarCube,
    *,
    radar: Any = None,
    fftSize: int | None = None,
    loopAxis: int = -2,
    window: str | None = None,
    removeMean: bool | None = None,
) -> RadarCube | NDArray[Any]:
    """Apply a centered slow-time Doppler FFT."""

    data, model, dims, source = _input(signal, radar)
    axis = _axis(dims, ("loop", "slowTime", "doppler"), loopAxis, data.ndim)
    config = model.processing.fft if model is not None else None
    size = (
        fftSize if fftSize is not None else (config.dopplerFftSize if config else None)
    )
    windowName = (
        window if window is not None else (config.dopplerWindow if config else "hann")
    )
    subtract = (
        removeMean
        if removeMean is not None
        else (config.removeDopplerMean if config else False)
    )
    result = _transform(
        data,
        axis=axis,
        fftSize=size,
        windowName=windowName,
        removeMean=subtract,
        shift=True,
    )
    outputDims = list(dims or tuple(f"axis{index}" for index in range(data.ndim)))
    outputDims[axis] = "doppler"
    coords: dict[str, NDArray[Any]] = {}
    if model is not None:
        coords["doppler"] = (
            np.arange(result.shape[axis]) - result.shape[axis] // 2
        ) * model.velocityBinSize
    if source is not None:
        coords.update(
            {key: value for key, value in source.coords.items() if key in outputDims}
        )
    return _cube(
        result,
        radar=model,
        source=source,
        dims=outputDims,
        stage="range_doppler",
        coords=coords,
    )


def angle_fft(
    signal: ArrayLike | RadarCube,
    *,
    radar: Any = None,
    fftSize: int | None = None,
    antennaAxis: int = -1,
    window: str | None = None,
    dimension: str = "azimuth",
) -> RadarCube | NDArray[Any]:
    """Apply a centered FFT over an already ordered uniform aperture."""

    if dimension not in {"azimuth", "elevation"}:
        raise ValueError("dimension must be 'azimuth' or 'elevation'.")
    data, model, dims, source = _input(signal, radar)
    axis = _axis(dims, ("virtual", "antenna", dimension), antennaAxis, data.ndim)
    config = model.processing.fft if model is not None else None
    defaultSize = None
    if config is not None:
        defaultSize = (
            config.azimuthFftSize if dimension == "azimuth" else config.elevationFftSize
        )
    size = fftSize if fftSize is not None else defaultSize
    windowName = (
        window if window is not None else (config.angleWindow if config else "hann")
    )
    result = _transform(
        data,
        axis=axis,
        fftSize=size,
        windowName=windowName,
        removeMean=False,
        shift=True,
    )
    outputDims = list(dims or tuple(f"axis{index}" for index in range(data.ndim)))
    outputDims[axis] = dimension
    coords: dict[str, NDArray[Any]] = {}
    if model is not None:
        modelAxis = model.azimuthAxis if dimension == "azimuth" else model.elevationAxis
        if modelAxis.size == result.shape[axis]:
            coords[dimension] = modelAxis
    if source is not None:
        coords.update(
            {key: value for key, value in source.coords.items() if key in outputDims}
        )
    return _cube(
        result,
        radar=model,
        source=source,
        dims=outputDims,
        stage="angle",
        coords=coords,
    )


def range_doppler_fft(
    adc: ArrayLike | RadarCube | ADCFrame,
    *,
    radar: Any = None,
    dims: Sequence[str] | None = None,
    rangeFftSize: int | None = None,
    dopplerFftSize: int | None = None,
    sampleAxis: int = -1,
    loopAxis: int = -2,
    rangeWindow: str | None = None,
    dopplerWindow: str | None = None,
    removeRangeMean: bool | None = None,
    removeDopplerMean: bool | None = None,
) -> RadarCube | NDArray[Any]:
    """Apply range and Doppler processing with optional model-aware MIMO decode."""

    model = radar or (adc.radar if isinstance(adc, ADCFrame | RadarCube) else None)
    if model is not None and not isinstance(adc, RadarCube):
        frame = (
            adc
            if isinstance(adc, ADCFrame)
            else ADCFrame.from_array(adc, dims=dims, radar=model)
        )
        canonical = frame.canonical(model)
        ranged = range_fft(
            canonical,
            radar=model,
            fftSize=rangeFftSize,
            window=rangeWindow,
            removeMean=removeRangeMean,
        )
        assert isinstance(ranged, RadarCube)
        decoded = model.mimo.decode(ranged.data)
        decodedCube = RadarCube(
            data=decoded,
            dims=("loop", "virtual", "range"),
            stage="range_mimo",
            radar=model,
            coords={"range": ranged.coords["range"]},
        )
        transformed = doppler_fft(
            decodedCube,
            radar=model,
            fftSize=dopplerFftSize,
            loopAxis=0,
            window=dopplerWindow,
            removeMean=removeDopplerMean,
        )
        assert isinstance(transformed, RadarCube)
        result = transformed.transpose("range", "doppler", "virtual")
        correction = model.mimo.phase_correction(
            result.shape[1], model.waveform.chirpInterval
        )
        corrected = result.data * correction[None, :, :]
        rangeIndices = model.rangeBinIndices
        corrected = corrected[rangeIndices]
        return RadarCube(
            data=corrected,
            dims=result.dims,
            stage="range_doppler",
            radar=model,
            coords={"range": model.rangeAxis, "doppler": model.velocityAxis},
            metadata={"rangeBinIndices": rangeIndices},
        )

    ranged = range_fft(
        adc,
        radar=model,
        fftSize=rangeFftSize,
        sampleAxis=sampleAxis,
        window=rangeWindow,
        removeMean=removeRangeMean,
    )
    return doppler_fft(
        ranged,
        radar=model,
        fftSize=dopplerFftSize,
        loopAxis=loopAxis,
        window=dopplerWindow,
        removeMean=removeDopplerMean,
    )


def range_doppler_azimuth_fft(
    adc: ArrayLike | RadarCube | ADCFrame,
    *,
    radar: Any = None,
    dims: Sequence[str] | None = None,
    **kwargs: Any,
) -> RadarCube | NDArray[Any]:
    """Convenience range, Doppler, and azimuth FFT composition."""

    rd = range_doppler_fft(adc, radar=radar, dims=dims, **kwargs)
    return angle_fft(rd, radar=radar, dimension="azimuth")


__all__ = [
    "angle_fft",
    "doppler_fft",
    "range_doppler_azimuth_fft",
    "range_doppler_fft",
    "range_fft",
    "window",
]
