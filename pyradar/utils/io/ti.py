"""Pure decoders for common TI raw ADC capture layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base import DDM, ADCFrame, Radar


def _signed(values: NDArray, numAdcBits: int) -> NDArray[np.int32]:
    if not 1 < numAdcBits <= 16:
        raise ValueError("numAdcBits must be in [2, 16].")
    result = np.asarray(values).astype(np.int32, copy=False)
    if numAdcBits < 16:
        mask = (1 << numAdcBits) - 1
        result = result & mask
        sign = 1 << (numAdcBits - 1)
        result = np.where(result & sign, result - (1 << numAdcBits), result)
    elif np.issubdtype(np.asarray(values).dtype, np.unsignedinteger):
        result = np.where(result >= 2**15, result - 2**16, result)
    return np.asarray(result, dtype=np.int32)


def decode_dca1000(
    raw: ArrayLike | bytes,
    *,
    numChirps: int,
    numRx: int,
    numSamples: int,
    numAdcBits: int = 16,
) -> NDArray[np.complex128]:
    """Decode DCA1000 words into ``(chirp, rx, sample)`` complex IQ data."""

    values = (
        np.frombuffer(raw, dtype="<i2")
        if isinstance(raw, bytes)
        else np.asarray(raw).reshape(-1)
    )
    expectedComplex = numChirps * numRx * numSamples
    if min(numChirps, numRx, numSamples) < 1 or values.size != 2 * expectedComplex:
        raise ValueError(
            f"Expected {2 * expectedComplex} DCA1000 words, got {values.size}."
        )
    if expectedComplex % 2:
        raise ValueError("DCA1000 two-lane IQ decoding requires an even sample count.")
    signed = _signed(values, numAdcBits)
    output = np.empty(expectedComplex, dtype=np.complex128)
    output[0::2] = signed[0::4] + 1j * signed[2::4]
    output[1::2] = signed[1::4] + 1j * signed[3::4]
    return output.reshape(numChirps, numRx, numSamples)


def decode_tsw1400(
    raw: ArrayLike | bytes,
    *,
    numFrames: int,
    numChirpsPerFrame: int,
    numRx: int,
    numSamples: int,
    complexSampling: bool = True,
    numAdcBits: int = 16,
) -> NDArray:
    """Decode TSW1400 offset-binary rows into frame/chirp/RX/sample data."""

    values = (
        np.frombuffer(raw, dtype="<u2")
        if isinstance(raw, bytes)
        else np.asarray(raw).reshape(-1)
    )
    channels = 2 if complexSampling else 1
    expected = numFrames * numChirpsPerFrame * numRx * numSamples * channels
    if min(numFrames, numChirpsPerFrame, numRx, numSamples) < 1:
        raise ValueError("TSW1400 dimensions must be positive.")
    if values.size != expected:
        raise ValueError(f"Expected {expected} TSW1400 words, got {values.size}.")
    if not 1 < numAdcBits <= 16:
        raise ValueError("numAdcBits must be in [2, 16].")
    mask = (1 << numAdcBits) - 1
    signed = (np.asarray(values, dtype=np.int32) & mask) - (1 << (numAdcBits - 1))
    rows = signed.reshape(numFrames, numChirpsPerFrame, numRx, channels * numSamples)
    if not complexSampling:
        return rows
    return rows[..., 0::2] + 1j * rows[..., 1::2]


def _frames(
    decoded: NDArray,
    radar: Radar,
    *,
    timestamps: ArrayLike | None = None,
) -> tuple[ADCFrame, ...]:
    count = decoded.shape[0]
    captureDuration = (
        radar.sampler.numLoops * radar.waveform.chirpInterval
        if isinstance(radar.mimo, DDM)
        else radar.sampler.numLoops * radar.slowTimeInterval
    )
    timestampArray = (
        np.arange(count, dtype=float) * (radar.sampler.framePeriod or captureDuration)
        if timestamps is None
        else np.asarray(timestamps, dtype=float)
    )
    if timestampArray.shape != (count,) or not np.all(np.isfinite(timestampArray)):
        raise ValueError("timestamps must contain one finite value per frame.")
    expectedChirps = radar.sampler.numLoops * radar.mimo.numEmissions
    if decoded.shape[1:] != (
        expectedChirps,
        radar.mimo.numRx,
        radar.sampler.numSamples,
    ):
        raise ValueError("Decoded TI data does not match the Radar capture shape.")
    canonical = decoded.reshape(
        count,
        radar.sampler.numLoops,
        radar.mimo.numEmissions,
        radar.mimo.numRx,
        radar.sampler.numSamples,
    )
    return tuple(
        ADCFrame(
            canonical[index],
            ("loop", "emission", "rx", "sample"),
            radar=radar,
            timestamp=float(timestampArray[index]),
            frameId=index,
            metadata={"source": "ti_raw_capture"},
        )
        for index in range(count)
    )


def read_dca1000(
    path: str | Path,
    radar: Radar,
    *,
    numFrames: int | None = None,
    timestamps: ArrayLike | None = None,
) -> tuple[ADCFrame, ...]:
    """Read one or more DCA1000 frames using a Radar capture model."""

    capturePath = Path(path)
    words = np.fromfile(capturePath, dtype="<i2")
    chirpsPerFrame = radar.sampler.numLoops * radar.mimo.numEmissions
    wordsPerFrame = 2 * chirpsPerFrame * radar.mimo.numRx * radar.sampler.numSamples
    count = words.size // wordsPerFrame if numFrames is None else int(numFrames)
    if count < 1 or words.size != count * wordsPerFrame:
        raise ValueError("DCA1000 file size is not an integer number of Radar frames.")
    decoded = decode_dca1000(
        words,
        numChirps=count * chirpsPerFrame,
        numRx=radar.mimo.numRx,
        numSamples=radar.sampler.numSamples,
        numAdcBits=radar.sampler.bitDepth,
    ).reshape(
        count,
        chirpsPerFrame,
        radar.mimo.numRx,
        radar.sampler.numSamples,
    )
    return _frames(decoded, radar, timestamps=timestamps)


def read_tsw1400(
    path: str | Path,
    radar: Radar,
    *,
    numFrames: int | None = None,
    timestamps: ArrayLike | None = None,
) -> tuple[ADCFrame, ...]:
    """Read one or more TSW1400 frames using a Radar capture model."""

    capturePath = Path(path)
    words = np.fromfile(capturePath, dtype="<u2")
    chirpsPerFrame = radar.sampler.numLoops * radar.mimo.numEmissions
    channels = 2 if radar.sampler.complexSampling else 1
    wordsPerFrame = (
        channels * chirpsPerFrame * radar.mimo.numRx * radar.sampler.numSamples
    )
    count = words.size // wordsPerFrame if numFrames is None else int(numFrames)
    if count < 1 or words.size != count * wordsPerFrame:
        raise ValueError("TSW1400 file size is not an integer number of Radar frames.")
    decoded = decode_tsw1400(
        words,
        numFrames=count,
        numChirpsPerFrame=chirpsPerFrame,
        numRx=radar.mimo.numRx,
        numSamples=radar.sampler.numSamples,
        complexSampling=radar.sampler.complexSampling,
        numAdcBits=radar.sampler.bitDepth,
    )
    return _frames(decoded, radar, timestamps=timestamps)


__all__ = [
    "decode_dca1000",
    "decode_tsw1400",
    "read_dca1000",
    "read_tsw1400",
]
