"""Radar-model-aware FMCW ADC synthesis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pyradar.base import BPM, DDM, SIMO, TDM, ADCFrame, Radar
from pyradar.base.waveform import SPEED_OF_LIGHT

from .targets import PointTarget, PropagationPath, TargetScene


def _emission_state(
    radar: Radar, loop: int, emission: int
) -> tuple[float, NDArray[np.complex128]]:
    """Return emission time and per-TX complex code weights."""

    mimo = radar.mimo
    chirpInterval = radar.waveform.chirpInterval
    weights = np.zeros(mimo.numTx, dtype=np.complex128)
    if isinstance(mimo, SIMO):
        time = loop * chirpInterval
        weights[0] = 1.0
    elif isinstance(mimo, TDM):
        offset = (
            mimo.emissionTimeOffsets[emission]
            if mimo.emissionTimeOffsets
            else emission * chirpInterval
        )
        time = loop * mimo.slow_time_interval(chirpInterval) + offset
        weights[mimo.txOrder[emission]] = 1.0
    elif isinstance(mimo, BPM):
        time = loop * mimo.slow_time_interval(chirpInterval) + emission * chirpInterval
        weights[:] = np.asarray(mimo.codeMatrix[emission], dtype=np.complex128)
    elif isinstance(mimo, DDM):
        time = loop * chirpInterval
        weights[:] = mimo.codes[:, loop % mimo.codeLength]
    else:  # pragma: no cover - protected by Radar's MIMOScheme contract
        raise TypeError(f"Unsupported MIMO strategy {type(mimo).__name__}.")
    return time, weights


def targets_to_paths(
    radar: Radar,
    targets: PointTarget | Sequence[PointTarget] | TargetScene,
    *,
    frameTime: float = 0.0,
    propagationLoss: bool = False,
) -> tuple[PropagationPath, ...]:
    """Convert point targets into one bistatic path per TX/RX pair."""

    if not np.isfinite(frameTime):
        raise ValueError("frameTime must be finite.")
    scene = TargetScene.coerce(targets)
    paths: list[PropagationPath] = []
    for target in scene.targets:
        position = target.position_at(frameTime)
        for txId, txPosition in enumerate(radar.transceivers.txPositions):
            txVector = position - txPosition
            txRange = float(np.linalg.norm(txVector))
            if txRange <= 0.0:
                raise ValueError("A target cannot occupy a TX phase center.")
            for rxId, rxPosition in enumerate(radar.transceivers.rxPositions):
                rxVector = position - rxPosition
                rxRange = float(np.linalg.norm(rxVector))
                if rxRange <= 0.0:
                    raise ValueError("A target cannot occupy an RX phase center.")
                pathRate = float(
                    np.dot(target.velocity, txVector / txRange)
                    + np.dot(target.velocity, rxVector / rxRange)
                )
                amplitude = target.reflection
                if propagationLoss:
                    amplitude *= radar.wavelength / (
                        (4.0 * np.pi) ** 1.5 * txRange * rxRange
                    )
                paths.append(
                    PropagationPath(
                        pathLength=txRange + rxRange,
                        pathRate=pathRate,
                        amplitude=amplitude,
                        txId=txId,
                        rxId=rxId,
                    )
                )
    return tuple(paths)


def simulate_paths(
    radar: Radar,
    paths: Sequence[PropagationPath],
    *,
    noisePower: float = 0.0,
    seed: int | None = None,
    timestamp: float | None = None,
    frameId: int | str | None = None,
) -> ADCFrame:
    """Synthesize canonical raw ADC from fixed linear propagation paths.

    The phase convention matches :func:`pyradar.rsp.steering_vector`: an
    approaching path has negative ``pathRate`` and produces positive Doppler.
    MIMO coding and true emission times are taken from ``radar.mimo``.
    """

    pathTuple = tuple(paths)
    if any(path.txId >= radar.mimo.numTx for path in pathTuple):
        raise ValueError("A propagation path references an unavailable TX.")
    if any(path.rxId >= radar.mimo.numRx for path in pathTuple):
        raise ValueError("A propagation path references an unavailable RX.")
    if not np.isfinite(noisePower) or noisePower < 0.0:
        raise ValueError("noisePower must be finite and nonnegative.")

    shape = (
        radar.sampler.numLoops,
        radar.mimo.numEmissions,
        radar.mimo.numRx,
        radar.sampler.numSamples,
    )
    adc = np.zeros(shape, dtype=np.complex128)
    # The model wavelength is evaluated at the sampled-band center.  Centering
    # fast time here keeps the narrowband phase and beat-frequency terms from
    # counting the FMCW frequency offset twice.
    sampleTimes = (
        np.arange(radar.sampler.numSamples, dtype=float) / radar.sampler.sampleRate
        - 0.5 * radar.sampler.captureDuration
    )
    for loop in range(shape[0]):
        for emission in range(shape[1]):
            emissionTime, txWeights = _emission_state(radar, loop, emission)
            for path in pathTuple:
                weight = txWeights[path.txId]
                if weight == 0.0:
                    continue
                pathLength = path.pathLength + path.pathRate * emissionTime
                if pathLength <= 0.0:
                    raise ValueError("A propagation path becomes nonpositive in-frame.")
                beatFrequency = (
                    radar.waveform.slope * pathLength / SPEED_OF_LIGHT
                    - path.pathRate / radar.wavelength
                )
                phaseCycles = beatFrequency * sampleTimes - (
                    pathLength / radar.wavelength
                )
                adc[loop, emission, path.rxId] += (
                    path.amplitude * weight * np.exp(2j * np.pi * phaseCycles)
                )

    if noisePower:
        generator = np.random.default_rng(seed)
        sigma = np.sqrt(noisePower / 2.0)
        adc += sigma * (
            generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
        )
    return ADCFrame(
        data=adc,
        dims=("loop", "emission", "rx", "sample"),
        radar=radar,
        timestamp=timestamp,
        frameId=frameId,
        metadata={
            "source": "pyradar.sim",
            "pathCount": len(pathTuple),
            "noisePower": float(noisePower),
        },
    )


def simulate_adc(
    radar: Radar,
    targets: PointTarget | Sequence[PointTarget] | TargetScene,
    *,
    frameTime: float = 0.0,
    propagationLoss: bool = False,
    noisePower: float = 0.0,
    seed: int | None = None,
    frameId: int | str | None = None,
) -> ADCFrame:
    """Synthesize one raw ADC frame from ideal point targets."""

    paths = targets_to_paths(
        radar,
        targets,
        frameTime=frameTime,
        propagationLoss=propagationLoss,
    )
    return simulate_paths(
        radar,
        paths,
        noisePower=noisePower,
        seed=seed,
        timestamp=frameTime,
        frameId=frameId,
    )


def quantize_adc(
    frame: ADCFrame,
    *,
    bitDepth: int | None = None,
    fullScale: float | None = None,
) -> ADCFrame:
    """Quantize complex ADC components while retaining a complex array."""

    depth = bitDepth or (frame.radar.sampler.bitDepth if frame.radar else 16)
    if depth < 2:
        raise ValueError("bitDepth must be at least two.")
    data = np.asarray(frame.data)
    peak = float(max(np.max(np.abs(data.real)), np.max(np.abs(data.imag))))
    scale = (peak or 1.0) if fullScale is None else float(fullScale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fullScale must be finite and positive.")
    limit = 2 ** (depth - 1) - 1

    def quantize(component: NDArray[np.float64]) -> NDArray[np.float64]:
        normalized = np.clip(component / scale, -1.0, 1.0)
        return np.rint(normalized * limit) * (scale / limit)

    output = quantize(data.real) + 1j * quantize(data.imag)
    metadata = dict(frame.metadata)
    metadata.update({"quantized": True, "bitDepth": depth, "fullScale": scale})
    return ADCFrame(
        output,
        frame.dims,
        radar=frame.radar,
        timestamp=frame.timestamp,
        frameId=frame.frameId,
        metadata=metadata,
    )


__all__ = [
    "quantize_adc",
    "simulate_adc",
    "simulate_paths",
    "targets_to_paths",
]
