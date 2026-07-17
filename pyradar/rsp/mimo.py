"""Model-aware MIMO compensation and ambiguity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base.mimo import TDM

if TYPE_CHECKING:
    from pyradar.base.radar import Radar


@dataclass(frozen=True, slots=True)
class DopplerUnwrapResult:
    """One TDM Doppler ambiguity decision.

    ``frequency`` is two-way Doppler frequency in hertz, ``velocity`` is radial
    velocity in metres per second, and ``ambiguityOrder`` is the integer alias
    multiple relative to the decoded slow-time interval.
    """

    frequency: float
    velocity: float
    ambiguityOrder: int
    score: float
    phaseCorrection: NDArray[np.complex128]


def _channel_offsets(radar: Radar) -> NDArray[np.float64]:
    if not isinstance(radar.mimo, TDM):
        raise TypeError("TDM velocity unwrapping requires a TDM Radar model.")
    txOffsets = radar.mimo.tx_time_offsets(radar.waveform.chirpInterval)
    return np.asarray(
        [txOffsets[txId] for txId, _ in radar.mimo.channelMap], dtype=float
    )


def _candidate_orders(baseFrequency: float, radar: Radar) -> NDArray[np.int64]:
    aliasSpacing = 1.0 / radar.slowTimeInterval
    rawLimit = 0.5 / radar.waveform.chirpInterval
    search = np.arange(-radar.mimo.numTx - 1, radar.mimo.numTx + 2)
    frequencies = baseFrequency + search * aliasSpacing
    tolerance = np.finfo(float).eps * rawLimit * 16.0
    valid = (frequencies >= -rawLimit - tolerance) & (
        frequencies < rawLimit - tolerance
    )
    return search[valid].astype(np.int64)


def unwrap_tdm_velocity(
    signal: ArrayLike,
    *,
    radar: Radar,
    dopplerBin: int,
    ambiguityOrders: ArrayLike | None = None,
) -> DopplerUnwrapResult:
    """Resolve a TDM Doppler alias from duplicate virtual phase centres.

    Parameters
    ----------
    signal:
        Uncompensated decoded virtual-channel samples with channels on the last
        axis. Leading dimensions are treated as snapshots.
    radar:
        A TDM radar with calibrated ``overlapPairs``.
    dopplerBin:
        Index on ``radar.velocityAxis`` before ambiguity resolution.
    ambiguityOrders:
        Optional integer alias candidates. By default candidates are limited to
        the raw chirp-rate Nyquist interval.

    Notes
    -----
    Every candidate applies its true emission-time phase correction. The chosen
    candidate minimizes weighted circular phase error between channels sharing a
    physical virtual phase centre. If all overlap samples have zero energy, the
    unaliased candidate (order zero) is retained with an infinite score.
    """

    if not isinstance(radar.mimo, TDM):
        raise TypeError("unwrap_tdm_velocity requires radar.mimo to be TDM.")
    data = np.asarray(signal)
    if data.ndim < 1 or data.shape[-1] != len(radar.mimo.channelMap):
        raise ValueError("signal must end with the Radar virtual-channel count.")
    if not 0 <= dopplerBin < radar.velocityAxis.size:
        raise ValueError("dopplerBin is outside radar.velocityAxis.")
    pairs = radar.calibration.overlapPairs
    if pairs is None or pairs.shape[0] == 0:
        raise ValueError("TDM velocity unwrapping requires overlapPairs.")
    pairIndices = np.asarray(pairs[:, :2], dtype=np.int64)
    if np.any(pairIndices < 0) or np.any(pairIndices >= data.shape[-1]):
        raise ValueError("overlapPairs contains an invalid virtual-channel index.")

    baseVelocity = float(radar.velocityAxis[dopplerBin])
    baseFrequency = 2.0 * baseVelocity / radar.wavelength
    if ambiguityOrders is None:
        orders = _candidate_orders(baseFrequency, radar)
    else:
        rawOrders = np.asarray(ambiguityOrders)
        if rawOrders.ndim != 1 or rawOrders.size == 0:
            raise ValueError(
                "ambiguityOrders must be a nonempty one-dimensional array."
            )
        if not np.all(np.equal(rawOrders, np.rint(rawOrders))):
            raise ValueError("ambiguityOrders must contain integers.")
        orders = rawOrders.astype(np.int64)
    if orders.size == 0:
        raise ValueError("No Doppler ambiguity candidate lies in the search interval.")

    aliasSpacing = 1.0 / radar.slowTimeInterval
    offsets = _channel_offsets(radar)
    arrayGain = radar.calibration.arrayPhase
    if arrayGain is not None and arrayGain.shape != (data.shape[-1],):
        raise ValueError("arrayPhase must match virtual channels before unwrapping.")

    best: tuple[float, int, float, NDArray[np.complex128]] | None = None
    for order in orders:
        frequency = baseFrequency + int(order) * aliasSpacing
        correction = np.exp(-2j * np.pi * frequency * offsets)
        corrected = data * correction
        if arrayGain is not None:
            corrected = corrected * arrayGain
        first = corrected[..., pairIndices[:, 0]].reshape(-1)
        second = corrected[..., pairIndices[:, 1]].reshape(-1)
        cross = first * second.conj()
        weights = np.abs(cross)
        valid = weights > np.finfo(float).tiny
        if np.any(valid):
            phaseError = np.angle(cross[valid])
            score = float(np.average(phaseError**2, weights=weights[valid]))
        else:
            score = float("inf")
        candidate = (score, abs(int(order)), frequency, correction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    assert best is not None
    score, _, frequency, correction = best
    if not np.isfinite(score) and np.any(orders == 0):
        frequency = baseFrequency
        correction = np.exp(-2j * np.pi * frequency * offsets)
    selectedOrder = int(np.rint((frequency - baseFrequency) / aliasSpacing))
    return DopplerUnwrapResult(
        frequency=float(frequency),
        velocity=float(frequency * radar.wavelength / 2.0),
        ambiguityOrder=selectedOrder,
        score=float(score),
        phaseCorrection=np.asarray(correction, dtype=np.complex128),
    )


__all__ = ["DopplerUnwrapResult", "unwrap_tdm_velocity"]
