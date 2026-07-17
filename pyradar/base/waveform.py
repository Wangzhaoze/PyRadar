"""FMCW waveform models.

All quantities in :mod:`pyradar.base` use SI units. Frequencies are in hertz
and times are in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True, slots=True)
class FMCW:
    """Sawtooth FMCW chirp definition.

    Parameters
    ----------
    startFrequency:
        Frequency at the beginning of the ramp, in Hz.
    slope:
        Ramp slope, in Hz/s. v1 supports positive-slope sawtooth chirps.
    adcStartTime:
        Delay from ramp start to the first ADC sample, in seconds.
    rampEndTime:
        Ramp duration measured from ramp start, in seconds.
    idleTime:
        Idle time after each ramp, in seconds.
    """

    startFrequency: float
    slope: float
    adcStartTime: float
    rampEndTime: float
    idleTime: float = 0.0
    shape: Literal["sawtooth"] = "sawtooth"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.startFrequency,
                self.slope,
                self.adcStartTime,
                self.rampEndTime,
                self.idleTime,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("FMCW parameters must be finite.")
        if self.startFrequency <= 0.0:
            raise ValueError("startFrequency must be positive.")
        if self.slope <= 0.0:
            raise ValueError("Only positive-slope sawtooth FMCW is supported in v1.")
        if self.adcStartTime < 0.0 or self.idleTime < 0.0:
            raise ValueError("adcStartTime and idleTime cannot be negative.")
        if self.rampEndTime <= self.adcStartTime:
            raise ValueError("rampEndTime must be later than adcStartTime.")

    @property
    def chirpInterval(self) -> float:
        """Time between starts of adjacent chirps."""

        return self.rampEndTime + self.idleTime

    @property
    def rampBandwidth(self) -> float:
        """Bandwidth swept during the complete ramp."""

        return self.slope * self.rampEndTime

    @property
    def centerFrequency(self) -> float:
        """Carrier frequency at the center of the complete ramp."""

        return self.startFrequency + 0.5 * self.rampBandwidth

    @property
    def wavelength(self) -> float:
        """Wavelength at :attr:`centerFrequency`."""

        return SPEED_OF_LIGHT / self.centerFrequency


# Short scientific alias retained for equations and downstream code.
C = SPEED_OF_LIGHT
