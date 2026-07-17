"""Immutable processing defaults attached to a radar model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Literal

WindowName = Literal["rectangular", "hann", "hamming", "blackman", "blackmanharris"]


@dataclass(frozen=True, slots=True)
class FFTConfig:
    """FFT sizes, windows, and optional output cropping."""

    rangeFftSize: int | None = None
    dopplerFftSize: int | None = None
    azimuthFftSize: int = 256
    elevationFftSize: int = 128
    rangeWindow: WindowName = "hann"
    dopplerWindow: WindowName = "hann"
    angleWindow: WindowName = "hann"
    removeRangeMean: bool = False
    removeDopplerMean: bool = True
    rangeCrop: tuple[int, int | None] | None = None
    azimuthCrop: tuple[int, int | None] | None = None
    elevationCrop: tuple[int, int | None] | None = None

    def __post_init__(self) -> None:
        for name in (
            "rangeFftSize",
            "dopplerFftSize",
            "azimuthFftSize",
            "elevationFftSize",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive.")

    def crop(self, dimension: str) -> slice:
        value = getattr(self, f"{dimension}Crop")
        return slice(None) if value is None else slice(*value)


@dataclass(frozen=True, slots=True)
class CFARConfig:
    """Range-Doppler CFAR and peak-grouping configuration."""

    method: Literal["ca", "goca", "soca", "os"] = "ca"
    trainingCells: tuple[int, int] = (8, 4)
    guardCells: tuple[int, int] = (2, 1)
    pfa: float = 1e-4
    rankFraction: float = 0.75
    minSnrDb: float = 0.0
    peakGrouping: bool = True
    maxDetections: int | None = 512

    def __post_init__(self) -> None:
        if any(value < 0 for value in (*self.trainingCells, *self.guardCells)):
            raise ValueError("CFAR training and guard cell counts cannot be negative.")
        if not 0.0 < self.pfa < 1.0:
            raise ValueError("pfa must lie strictly between zero and one.")
        if not 0.0 < self.rankFraction <= 1.0:
            raise ValueError("rankFraction must lie in (0, 1].")
        if self.maxDetections is not None and self.maxDetections < 1:
            raise ValueError("maxDetections must be positive.")


@dataclass(frozen=True, slots=True)
class DoAConfig:
    """Direction-of-arrival defaults in radians."""

    method: Literal["auto", "fft", "bartlett", "capon", "music", "esprit"] = "auto"
    azimuthFov: tuple[float, float] = (-pi / 2.0, pi / 2.0)
    elevationFov: tuple[float, float] = (-pi / 4.0, pi / 4.0)
    azimuthBins: int = 181
    elevationBins: int = 61
    numSources: int = 1
    diagonalLoading: float = 1e-3
    duplicatePolicy: Literal["first", "noncoherent", "coherent"] = "coherent"

    def __post_init__(self) -> None:
        if self.azimuthBins < 2 or self.elevationBins < 2 or self.numSources < 1:
            raise ValueError("DoA grid sizes and numSources must be positive.")
        for name in ("azimuthFov", "elevationFov"):
            low, high = getattr(self, name)
            if not -pi / 2 <= low < high <= pi / 2:
                raise ValueError(f"{name} must be ordered inside [-pi/2, pi/2].")
        if self.diagonalLoading < 0.0:
            raise ValueError("diagonalLoading cannot be negative.")


@dataclass(frozen=True, slots=True)
class PointCloudConfig:
    """Point-cloud filtering defaults."""

    minRange: float = 0.0
    maxRange: float | None = None
    unwrapTdmVelocity: bool = True

    def __post_init__(self) -> None:
        if self.minRange < 0.0:
            raise ValueError("minRange cannot be negative.")
        if self.maxRange is not None and self.maxRange <= self.minRange:
            raise ValueError("maxRange must be greater than minRange.")


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    """Velocity-aware deterministic DBSCAN defaults."""

    enabled: bool = False
    eps: float = 1.0
    minSamples: int = 3
    velocityScale: float = 1.0
    dimensions: Literal[2, 3] = 3

    def __post_init__(self) -> None:
        if self.eps <= 0.0 or self.minSamples < 1:
            raise ValueError("Clustering eps and minSamples must be positive.")
        if self.velocityScale < 0.0:
            raise ValueError("velocityScale cannot be negative.")
        if self.dimensions not in (2, 3):
            raise ValueError("Clustering dimensions must be 2 or 3.")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Constant-velocity multi-target tracking defaults."""

    enabled: bool = False
    dimensions: Literal[2, 3] = 3
    processNoise: float = 2.0
    measurementNoise: float = 0.5
    radialVelocityNoise: float = 1.0
    gatingThreshold: float = 11.34
    confirmationHits: int = 3
    deletionMisses: int = 5

    def __post_init__(self) -> None:
        if self.dimensions not in (2, 3):
            raise ValueError("Tracking dimensions must be 2 or 3.")
        if self.processNoise < 0.0:
            raise ValueError("processNoise cannot be negative.")
        if self.measurementNoise <= 0.0 or self.radialVelocityNoise <= 0.0:
            raise ValueError("Measurement noise values must be positive.")
        if self.gatingThreshold <= 0.0:
            raise ValueError("gatingThreshold must be positive.")
        if self.confirmationHits < 1 or self.deletionMisses < 1:
            raise ValueError("Track lifecycle counts must be positive.")


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Layered algorithm defaults used by :class:`~pyradar.base.Radar`."""

    fft: FFTConfig = field(default_factory=FFTConfig)
    cfar: CFARConfig = field(default_factory=CFARConfig)
    doa: DoAConfig = field(default_factory=DoAConfig)
    pointCloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    retainRangeDoppler: bool = True
    retainRangeAngle: bool = False
    retainRangeDopplerAngle: bool = False
