"""Immutable radar models and hardware capture profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .calibration import Calibration
from .mimo import BPM, DDM, SIMO, TDM, MIMOScheme
from .processing import (
    CFARConfig,
    ClusteringConfig,
    DoAConfig,
    FFTConfig,
    PointCloudConfig,
    ProcessingConfig,
    TrackingConfig,
)
from .sampler import Sampler
from .transceivers import Transceivers
from .waveform import FMCW, SPEED_OF_LIGHT


def _uniform_spacing(values: NDArray[np.float64], tolerance: float) -> float | None:
    unique = np.unique(np.rint(np.asarray(values) / tolerance).astype(np.int64))
    if unique.size < 2:
        return None
    differences = np.diff(unique).astype(float) * tolerance
    if np.allclose(differences, differences[0], rtol=1e-5, atol=tolerance):
        return float(differences[0])
    return None


def _mimo_from_mapping(config: Mapping[str, Any], numRx: int) -> MIMOScheme:
    values = dict(config)
    kind = str(values.pop("type", values.pop("scheme", "simo"))).lower()
    values.setdefault("numRx", numRx)
    classes = {"simo": SIMO, "tdm": TDM, "bpm": BPM, "ddm": DDM}
    if kind not in classes:
        raise ValueError(f"Unknown MIMO type {kind!r}.")
    return classes[kind](**values)


def _processing_from_mapping(config: Mapping[str, Any] | None) -> ProcessingConfig:
    values = dict(config or {})
    constructors = {
        "fft": FFTConfig,
        "cfar": CFARConfig,
        "doa": DoAConfig,
        "pointCloud": PointCloudConfig,
        "clustering": ClusteringConfig,
        "tracking": TrackingConfig,
    }
    for name, constructor in constructors.items():
        if name in values and not isinstance(values[name], constructor):
            values[name] = constructor(**values[name])
    return ProcessingConfig(**values)


@dataclass(frozen=True)
class Radar:
    """Validated radar aggregate used as the context for all algorithms."""

    waveform: FMCW
    sampler: Sampler
    transceivers: Transceivers
    mimo: MIMOScheme
    calibration: Calibration = field(default_factory=Calibration)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    name: str = "radar"

    def __post_init__(self) -> None:
        if self.transceivers.numTx != self.mimo.numTx:
            raise ValueError("MIMO numTx does not match transceiver geometry.")
        if self.transceivers.numRx != self.mimo.numRx:
            raise ValueError("MIMO numRx does not match transceiver geometry.")
        sampleEnd = self.waveform.adcStartTime + self.sampler.captureDuration
        tolerance = max(1e-12, self.waveform.rampEndTime * 1e-9)
        if sampleEnd > self.waveform.rampEndTime + tolerance:
            raise ValueError(
                "ADC capture extends past rampEndTime; fix waveform or sampler parameters."
            )
        if not self.name:
            raise ValueError("Radar name cannot be empty.")

    @classmethod
    def from_config(cls, path: str | Path) -> Radar:
        """Build a radar from a v1 YAML or JSON configuration file."""

        configPath = Path(path)
        if not configPath.is_file():
            raise FileNotFoundError(configPath)
        with configPath.open("r", encoding="utf-8") as stream:
            if configPath.suffix.lower() == ".json":
                config = json.load(stream)
            elif configPath.suffix.lower() in {".yaml", ".yml"}:
                config = yaml.safe_load(stream)
            else:
                raise ValueError("Radar config must be YAML or JSON.")
        try:
            waveform = FMCW(**config["waveform"])
            sampler = Sampler(**config["sampler"])
            transceivers = Transceivers(**config["transceivers"])
            mimo = _mimo_from_mapping(config.get("mimo", {}), transceivers.numRx)
            calibration = Calibration(**config.get("calibration", {}))
            processing = _processing_from_mapping(config.get("processing"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid radar config {configPath}: {exc}") from exc
        return cls(
            waveform=waveform,
            sampler=sampler,
            transceivers=transceivers,
            mimo=mimo,
            calibration=calibration,
            processing=processing,
            name=config.get("name", configPath.stem),
        )

    @classmethod
    def radelft_cascade(cls) -> TI2243CascadeRadar:
        return TI2243CascadeRadar.radelft()

    @classmethod
    def coloradar_cascade(cls) -> TI2243CascadeRadar:
        return TI2243CascadeRadar.coloradar()

    @classmethod
    def coloradar_plus_cascade(cls) -> TI2243CascadeRadar:
        return TI2243CascadeRadar.coloradar_plus()

    @classmethod
    def awr1843_rampcnn(cls) -> AWR1843Radar:
        return AWR1843Radar.rampcnn()

    @property
    def sampleDuration(self) -> float:
        return self.sampler.captureDuration

    @property
    def sampledBandwidth(self) -> float:
        return self.waveform.slope * self.sampleDuration

    @property
    def centerFrequency(self) -> float:
        centerTime = self.waveform.adcStartTime + 0.5 * self.sampleDuration
        return self.waveform.startFrequency + self.waveform.slope * centerTime

    @property
    def wavelength(self) -> float:
        return SPEED_OF_LIGHT / self.centerFrequency

    @property
    def rangeResolution(self) -> float:
        return SPEED_OF_LIGHT / (2.0 * self.sampledBandwidth)

    @property
    def rangeBinSize(self) -> float:
        size = self.processing.fft.rangeFftSize or self.sampler.numSamples
        return (
            SPEED_OF_LIGHT
            * self.sampler.sampleRate
            / (2.0 * self.waveform.slope * size)
        )

    @property
    def maxUnambiguousRange(self) -> float:
        beatBandwidth = self.sampler.sampleRate
        if not self.sampler.complexSampling:
            beatBandwidth *= 0.5
        return SPEED_OF_LIGHT * beatBandwidth / (2.0 * self.waveform.slope)

    @property
    def numSlowTimeSamples(self) -> int:
        if isinstance(self.mimo, DDM):
            return self.sampler.numLoops // self.mimo.codeLength
        return self.sampler.numLoops

    @property
    def slowTimeInterval(self) -> float:
        return self.mimo.slow_time_interval(self.waveform.chirpInterval)

    @property
    def velocityResolution(self) -> float:
        return self.wavelength / (2.0 * self.numSlowTimeSamples * self.slowTimeInterval)

    @property
    def velocityBinSize(self) -> float:
        size = self.processing.fft.dopplerFftSize or self.numSlowTimeSamples
        return self.wavelength / (2.0 * size * self.slowTimeInterval)

    @property
    def maxUnambiguousVelocity(self) -> float:
        return self.wavelength / (4.0 * self.slowTimeInterval)

    @property
    def virtualArray(self) -> NDArray[np.float64]:
        return self.transceivers.virtual_positions(self.mimo.channelMap)

    @property
    def duplicatePhaseCenters(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            group
            for group in self.transceivers.phase_center_groups(
                self.virtualArray, tolerance=max(self.wavelength * 1e-6, 1e-12)
            )
            if len(group) > 1
        )

    @property
    def arrayGeometry(self) -> str:
        positions = self.virtualArray
        tolerance = max(self.wavelength * 1e-6, 1e-12)
        keys = np.unique(
            np.rint(positions[:, 1:3] / tolerance).astype(np.int64), axis=0
        )
        y = np.unique(keys[:, 0])
        z = np.unique(keys[:, 1])
        ySpacing = _uniform_spacing(keys[:, 0] * tolerance, tolerance)
        zSpacing = _uniform_spacing(keys[:, 1] * tolerance, tolerance)
        if z.size == 1 and ySpacing is not None:
            return "ula"
        if (
            keys.shape[0] == y.size * z.size
            and ySpacing is not None
            and zSpacing is not None
        ):
            return "ura"
        return "sparse"

    @property
    def unambiguousFov(self) -> dict[str, tuple[float, float]]:
        tolerance = max(self.wavelength * 1e-6, 1e-12)

        def fov(values: NDArray[np.float64]) -> tuple[float, float]:
            unique = np.unique(np.rint(values / tolerance).astype(np.int64)) * tolerance
            if unique.size < 2:
                return (-np.pi / 2.0, np.pi / 2.0)
            spacing = float(np.min(np.diff(np.sort(unique))))
            limit = float(np.arcsin(min(1.0, self.wavelength / (2.0 * spacing))))
            return (-limit, limit)

        return {
            "azimuth": fov(self.virtualArray[:, 1]),
            "elevation": fov(self.virtualArray[:, 2]),
        }

    @property
    def rangeBinIndices(self) -> NDArray[np.int64]:
        size = self.processing.fft.rangeFftSize or self.sampler.numSamples
        return np.arange(size, dtype=np.int64)[self.processing.fft.crop("range")]

    @property
    def rangeAxis(self) -> NDArray[np.float64]:
        return self.rangeBinIndices.astype(float) * self.rangeBinSize

    @property
    def velocityAxis(self) -> NDArray[np.float64]:
        size = self.processing.fft.dopplerFftSize or self.numSlowTimeSamples
        return (np.arange(size, dtype=float) - size // 2) * self.velocityBinSize

    def _angle_axis(self, dimension: str) -> NDArray[np.float64]:
        config = self.processing
        positions = self.virtualArray[:, 1 if dimension == "azimuth" else 2]
        size = (
            config.fft.azimuthFftSize
            if dimension == "azimuth"
            else config.fft.elevationFftSize
        )
        tolerance = max(self.wavelength * 1e-6, 1e-12)
        spacing = _uniform_spacing(positions, tolerance)
        if spacing is not None:
            spatial = (np.arange(size, dtype=float) - size // 2) / size
            axis = np.arcsin(np.clip(spatial * self.wavelength / spacing, -1.0, 1.0))
            return axis[config.fft.crop(dimension)]
        fov = (
            config.doa.azimuthFov if dimension == "azimuth" else config.doa.elevationFov
        )
        bins = (
            config.doa.azimuthBins
            if dimension == "azimuth"
            else config.doa.elevationBins
        )
        return np.linspace(fov[0], fov[1], bins)

    @property
    def azimuthAxis(self) -> NDArray[np.float64]:
        return self._angle_axis("azimuth")

    @property
    def elevationAxis(self) -> NDArray[np.float64]:
        return self._angle_axis("elevation")

    def build_pipeline(self):
        """Create a stateful ADC-to-track processing pipeline."""

        from pyradar.rsp.pipeline import ADCToPointCloudPipeline

        return ADCToPointCloudPipeline(self)

    def process_adc(
        self, adc: Any, *, dims: Sequence[str] | None = None, **kwargs: Any
    ):
        """Process one frame with a fresh pipeline."""

        return self.build_pipeline().process(adc, dims=dims, **kwargs)


def _ti2243_positions(wavelength: float, layout: str) -> Transceivers:
    if layout == "radelft":
        txY = np.asarray([0, 4, 8, 12, 16, 20, 24, 28, 32, 9, 10, 11], dtype=float)
        txZ = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 6], dtype=float)
    elif layout == "coloradar":
        txY = np.asarray([0, 4, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32], dtype=float)
        txZ = np.asarray([0, 0, 0, 1, 4, 6, 0, 0, 0, 0, 0, 0], dtype=float)
    else:
        raise ValueError("TI2243 layout must be 'radelft' or 'coloradar'.")
    rxY = np.asarray([*range(0, 4), *range(11, 15), *range(46, 54)], dtype=float)
    # The cascade PCB spacing is referenced to its 76.8 GHz design frequency,
    # not to the center frequency of an individual capture profile.
    scale = SPEED_OF_LIGHT / 76.8e9 / 2.0
    tx = np.column_stack((np.zeros(12), txY * scale, txZ * scale))
    rx = np.column_stack((np.zeros(16), rxY * scale, np.zeros(16)))
    return Transceivers(txPositions=tx, rxPositions=rx)


def _overlap_pairs(
    positions: NDArray[np.float64], tolerance: float
) -> NDArray[np.int64]:
    keys = np.rint(positions / tolerance).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groupKey = (int(key[0]), int(key[1]), int(key[2]))
        groups.setdefault(groupKey, []).append(index)
    pairs = [
        (group[left], group[right])
        for group in groups.values()
        for left in range(len(group))
        for right in range(left + 1, len(group))
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class TI2243CascadeRadar(Radar):
    """A TI AWR2243 four-chip cascade with a concrete capture profile."""

    hardware: str = "TI AWR2243 4-chip cascade"
    profileName: str = "custom"

    @classmethod
    def _profile(
        cls,
        *,
        name: str,
        profileName: str,
        layout: str,
        waveform: FMCW,
        sampler: Sampler,
        processing: ProcessingConfig,
    ) -> TI2243CascadeRadar:
        provisionalWavelength = SPEED_OF_LIGHT / (
            waveform.startFrequency
            + waveform.slope * (waveform.adcStartTime + sampler.captureDuration / 2.0)
        )
        transceivers = _ti2243_positions(provisionalWavelength, layout)
        mimo = TDM(numTx=12, numRx=16, txOrder=tuple(range(12)))
        virtual = transceivers.virtual_positions(mimo.channelMap)
        calibration = Calibration(
            overlapPairs=_overlap_pairs(
                virtual, max(provisionalWavelength * 1e-6, 1e-12)
            )
        )
        return cls(
            waveform=waveform,
            sampler=sampler,
            transceivers=transceivers,
            mimo=mimo,
            calibration=calibration,
            processing=processing,
            name=name,
            profileName=profileName,
        )

    @classmethod
    def radelft(cls) -> TI2243CascadeRadar:
        """RaDelft capture profile using its physical chirp parameters."""

        return cls._profile(
            name="radelft_cascade",
            profileName="RaDelft",
            layout="radelft",
            waveform=FMCW(
                startFrequency=76.0e9,
                slope=35.003e12,
                adcStartTime=6e-6,
                rampEndTime=28e-6,
                idleTime=5e-6,
            ),
            sampler=Sampler(numSamples=256, numLoops=128, sampleRate=12e6),
            processing=ProcessingConfig(
                fft=FFTConfig(
                    rangeFftSize=512,
                    dopplerFftSize=128,
                    azimuthFftSize=256,
                    elevationFftSize=128,
                    rangeWindow="hann",
                    dopplerWindow="hann",
                    angleWindow="hann",
                    rangeCrop=(10, -2),
                    azimuthCrop=(8, 248),
                    elevationCrop=(47, 81),
                ),
                cfar=CFARConfig(
                    method="os",
                    trainingCells=(10, 4),
                    guardCells=(1, 1),
                    pfa=1e-4,
                    maxDetections=128,
                ),
                doa=DoAConfig(
                    method="auto", elevationFov=(-0.5, 0.5), elevationBins=31
                ),
                retainRangeAngle=True,
            ),
        )

    @classmethod
    def coloradar(cls) -> TI2243CascadeRadar:
        """Original ColoRadar cascade capture profile."""

        return cls._profile(
            name="coloradar_cascade",
            profileName="ColoRadar",
            layout="coloradar",
            waveform=FMCW(
                startFrequency=76.999999488e9,
                slope=79.000001052672e12,
                adcStartTime=5e-6,
                rampEndTime=40e-6,
                idleTime=5e-6,
            ),
            sampler=Sampler(numSamples=256, numLoops=16, sampleRate=8e6),
            processing=ProcessingConfig(
                fft=FFTConfig(
                    rangeFftSize=256,
                    dopplerFftSize=16,
                    azimuthFftSize=128,
                    elevationFftSize=64,
                    rangeWindow="blackman",
                    dopplerWindow="rectangular",
                    angleWindow="rectangular",
                    removeDopplerMean=False,
                ),
                cfar=CFARConfig(
                    method="os",
                    trainingCells=(8, 3),
                    guardCells=(1, 1),
                    pfa=1e-4,
                    maxDetections=128,
                ),
                doa=DoAConfig(
                    method="auto", elevationFov=(-0.35, 0.35), elevationBins=31
                ),
                retainRangeAngle=True,
            ),
        )

    @classmethod
    def coloradar_plus(cls) -> TI2243CascadeRadar:
        """ColoRadar+ fallback profile; dataset readers replace exact metadata."""

        base = cls.coloradar()
        return cls(
            waveform=base.waveform,
            sampler=base.sampler,
            transceivers=base.transceivers,
            mimo=base.mimo,
            calibration=base.calibration,
            processing=base.processing,
            name="coloradar_plus_cascade",
            profileName="ColoRadar+",
        )


@dataclass(frozen=True)
class AWR1843Radar(Radar):
    """TI AWR1843 capture profile used by RAMPCNN/CRUW."""

    hardware: str = "TI AWR1843"
    profileName: str = "custom"

    @classmethod
    def rampcnn(cls) -> AWR1843Radar:
        waveform = FMCW(
            startFrequency=77e9,
            slope=21.0017e12,
            adcStartTime=6e-6,
            rampEndTime=40e-6,
            idleTime=5e-6,
        )
        sampler = Sampler(numSamples=128, numLoops=255, sampleRate=4e6)
        center = waveform.startFrequency + waveform.slope * (
            waveform.adcStartTime + sampler.captureDuration / 2.0
        )
        wavelength = SPEED_OF_LIGHT / center
        rx = np.column_stack(
            (np.zeros(4), np.arange(4) * wavelength / 2.0, np.zeros(4))
        )
        tx = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0 * wavelength, 0.0]])
        return cls(
            waveform=waveform,
            sampler=sampler,
            transceivers=Transceivers(txPositions=tx, rxPositions=rx),
            mimo=TDM(numTx=2, numRx=4),
            processing=ProcessingConfig(
                fft=FFTConfig(rangeFftSize=128, dopplerFftSize=256, azimuthFftSize=128),
                doa=DoAConfig(method="auto", elevationFov=(-0.1, 0.1)),
                retainRangeAngle=True,
            ),
            name="awr1843_rampcnn",
            profileName="RAMPCNN",
        )
