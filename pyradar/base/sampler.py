"""ADC and frame sampling models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Sampler:
    """ADC sampling and frame configuration.

    Parameters are the samples captured per emission and the number of MIMO
    loops in one frame. Complex IQ sampling is the default for mmWave radars.
    """

    numSamples: int
    numLoops: int
    sampleRate: float
    bitDepth: int = 16
    complexSampling: bool = True
    framePeriod: float | None = None

    def __post_init__(self) -> None:
        if self.numSamples < 1 or self.numLoops < 1:
            raise ValueError("numSamples and numLoops must be positive integers.")
        if not np.isfinite(self.sampleRate) or self.sampleRate <= 0.0:
            raise ValueError("sampleRate must be finite and positive.")
        if self.bitDepth < 1:
            raise ValueError("bitDepth must be positive.")
        if self.framePeriod is not None and self.framePeriod <= 0.0:
            raise ValueError("framePeriod must be positive when provided.")

    @property
    def samplePeriod(self) -> float:
        """ADC sampling period in seconds."""

        return 1.0 / self.sampleRate

    @property
    def captureDuration(self) -> float:
        """Duration covered by the ADC samples of one emission."""

        return self.numSamples / self.sampleRate
