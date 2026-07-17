"""Strictly validated radar calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _frozen_array(value: ArrayLike | None, dtype: Any = None) -> NDArray[Any] | None:
    if value is None:
        return None
    array = np.array(value, dtype=dtype, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("Calibration arrays must contain finite values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Calibration:
    """Calibration terms grouped by the signal-processing stage.

    ``adcGain``, ``adcPhase``, and ``frequencySlope`` are indexed by
    ``(emission, rx)`` (or by ``rx`` for SIMO). ``rangeCoupling`` is indexed by
    ``(virtual, range)``. ``arrayPhase`` has one complex coefficient per
    virtual channel. Any supplied shape is checked at the point of use.
    """

    adcGain: NDArray[Any] | None = None
    adcPhase: NDArray[Any] | None = None
    frequencySlope: NDArray[Any] | None = None
    rangeCoupling: NDArray[Any] | None = None
    arrayPhase: NDArray[Any] | None = None
    overlapPairs: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "adcGain", _frozen_array(self.adcGain))
        object.__setattr__(self, "adcPhase", _frozen_array(self.adcPhase))
        object.__setattr__(self, "frequencySlope", _frozen_array(self.frequencySlope))
        object.__setattr__(self, "rangeCoupling", _frozen_array(self.rangeCoupling))
        object.__setattr__(self, "arrayPhase", _frozen_array(self.arrayPhase))
        pairs = _frozen_array(self.overlapPairs, np.int64)
        if pairs is not None and (pairs.ndim != 2 or pairs.shape[1] < 2):
            raise ValueError("overlapPairs must have shape (N, 2+) when supplied.")
        object.__setattr__(self, "overlapPairs", pairs)

    def apply_adc(self, data: NDArray[Any]) -> NDArray[Any]:
        """Apply gain, phase, and frequency calibration to canonical ADC data."""

        array = np.asarray(data)
        if array.ndim != 4:
            raise ValueError("ADC calibration expects (loop, emission, rx, sample).")
        channelShape = array.shape[1:3]
        coefficient = np.ones(channelShape, dtype=np.complex128)
        for name, value, convert in (
            ("adcGain", self.adcGain, lambda item: item),
            ("adcPhase", self.adcPhase, lambda item: np.exp(1j * item)),
        ):
            if value is None:
                continue
            item = np.asarray(value)
            if item.shape == (channelShape[1],) and channelShape[0] == 1:
                item = item[None, :]
            if item.shape != channelShape:
                raise ValueError(
                    f"{name} shape {item.shape} does not match {channelShape}."
                )
            coefficient *= convert(item)
        output = array * coefficient[None, :, :, None]
        if self.frequencySlope is not None:
            slope = np.asarray(self.frequencySlope)
            if slope.shape == (channelShape[1],) and channelShape[0] == 1:
                slope = slope[None, :]
            if slope.shape != channelShape:
                raise ValueError(
                    f"frequencySlope shape {slope.shape} does not match {channelShape}."
                )
            samples = np.arange(array.shape[-1], dtype=float)
            output = output * np.exp(1j * slope[None, :, :, None] * samples)
        return output

    def apply_range(self, data: NDArray[Any]) -> NDArray[Any]:
        """Subtract range-domain coupling from ``(loop, virtual, range)`` data."""

        array = np.asarray(data)
        if array.ndim != 3:
            raise ValueError("Range calibration expects (loop, virtual, range).")
        if self.rangeCoupling is None:
            return array
        coupling = np.asarray(self.rangeCoupling)
        expected = array.shape[1:]
        if coupling.shape != expected:
            raise ValueError(
                f"rangeCoupling shape {coupling.shape} does not match {expected}."
            )
        return array - coupling[None, :, :]

    def apply_array(self, data: NDArray[Any], channelAxis: int = -1) -> NDArray[Any]:
        """Apply calibrated complex phase coefficients along a virtual axis."""

        array = np.asarray(data)
        if self.arrayPhase is None:
            return array
        coefficients = np.asarray(self.arrayPhase)
        axis = channelAxis % array.ndim
        if coefficients.ndim != 1 or coefficients.size != array.shape[axis]:
            raise ValueError(
                "arrayPhase must be one-dimensional and match the virtual channel axis."
            )
        shape = [1] * array.ndim
        shape[axis] = coefficients.size
        return array * coefficients.reshape(shape)
