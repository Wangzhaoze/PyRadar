"""Readers for the original ColoRadar and ColoRadar+ cascade captures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pyradar.base import (
    FMCW,
    TDM,
    ADCFrame,
    Calibration,
    Sampler,
    TI2243CascadeRadar,
    Transceivers,
)
from pyradar.base.waveform import SPEED_OF_LIGHT

from .base import PathLike


def _scalar_config(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, raw = line.split(maxsplit=1)
            values[name.lower()] = float(raw)
    return values


def _colon_config(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, raw = line.split(":", 1)
            parts = [part for part in raw.split(",") if part]
            values[name.lower()] = (
                float(parts[0]) if len(parts) == 1 else [float(part) for part in parts]
            )
    return values


def _antenna_config(path: Path) -> Transceivers:
    designFrequency = None
    tx: dict[int, tuple[float, float]] = {}
    rx: dict[int, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            chunks = line.strip().split()
            if not chunks or chunks[0].startswith("#"):
                continue
            if chunks[0] == "F_design":
                designFrequency = float(chunks[1]) * 1e9
            elif chunks[0] in {"tx", "rx"}:
                destination = tx if chunks[0] == "tx" else rx
                destination[int(chunks[1])] = (float(chunks[2]), float(chunks[3]))
    if designFrequency is None or not tx or not rx:
        raise ValueError(f"Incomplete ColoRadar antenna config: {path}.")
    scale = SPEED_OF_LIGHT / designFrequency / 2.0

    def positions(mapping: dict[int, tuple[float, float]]) -> np.ndarray:
        ordered = [mapping[index] for index in sorted(mapping)]
        return np.asarray([[0.0, y * scale, z * scale] for y, z in ordered])

    return Transceivers(txPositions=positions(tx), rxPositions=positions(rx))


def _normalize_time(name: str, value: float) -> float:
    """Normalize known ColoRadar export scale anomalies with strict bounds."""

    if name == "adc_start_time" and value > 1e-3:
        value *= 1e-6
    if name == "ramp_end_time" and value < 1e-8:
        value *= 1e6
    if not 0.0 <= value < 1e-2:
        raise ValueError(
            f"ColoRadar {name}={value!r} cannot be interpreted as seconds."
        )
    return value


def load_coloradar_calibration(
    calibrationDir: PathLike,
    *,
    radar: TI2243CascadeRadar | None = None,
) -> Calibration:
    """Parse cascade ADC, phase, frequency, and coupling calibration."""

    root = Path(calibrationDir)
    waveform = _scalar_config(root / "waveform_cfg.txt")
    couplingConfig = _colon_config(root / "coupling_calib.txt")
    with (root / "phase_frequency_calib.txt").open("r", encoding="utf-8") as stream:
        phaseConfig = json.load(stream)["antennaCalib"]

    numTx = int(phaseConfig["numTx"])
    numRx = int(phaseConfig["numRx"])
    numSamples = int(waveform["num_adc_samples_per_chirp"])
    rawPhase = np.asarray(phaseConfig["phaseCalibrationMatrix"], dtype=float)
    measured = rawPhase[0::2] + 1j * rawPhase[1::2]
    coefficient = (measured[0] / measured).reshape(numTx, numRx)

    calibrationFrequency = np.asarray(
        phaseConfig["frequencyCalibrationMatrix"], dtype=float
    ).reshape(numTx, numRx)
    delta = calibrationFrequency - calibrationFrequency.flat[0]
    frequencySlope = (
        -2.0
        * np.pi
        * delta
        * (waveform["frequency_slope"] / float(phaseConfig["frequencySlope"]))
        * (float(phaseConfig["samplingRate"]) / waveform["adc_sample_frequency"])
        / numSamples
    )

    coupling = np.asarray(couplingConfig["data"], dtype=float)
    expected = numTx * numRx * numSamples
    if coupling.size != expected:
        raise ValueError(
            f"Coupling calibration has {coupling.size} values; expected {expected}."
        )
    coupling = coupling.reshape(numTx * numRx, numSamples)
    overlap = None if radar is None else radar.calibration.overlapPairs
    return Calibration(
        adcGain=np.abs(coefficient),
        adcPhase=np.angle(coefficient),
        frequencySlope=frequencySlope,
        rangeCoupling=coupling,
        overlapPairs=overlap,
    )


def radar_from_coloradar(
    calibrationDir: PathLike,
    *,
    plus: bool = False,
    applyCalibration: bool = True,
) -> TI2243CascadeRadar:
    """Create a TI2243 capture profile from ColoRadar text/JSON metadata."""

    root = Path(calibrationDir)
    values = _scalar_config(root / "waveform_cfg.txt")
    waveform = FMCW(
        startFrequency=values["start_frequency"],
        slope=values["frequency_slope"],
        adcStartTime=_normalize_time("adc_start_time", values["adc_start_time"]),
        rampEndTime=_normalize_time("ramp_end_time", values["ramp_end_time"]),
        idleTime=_normalize_time("idle_time", values["idle_time"]),
    )
    sampler = Sampler(
        numSamples=int(values["num_adc_samples_per_chirp"]),
        numLoops=int(values["num_chirps_per_frame"]),
        sampleRate=values["adc_sample_frequency"],
    )
    base = (
        TI2243CascadeRadar.coloradar_plus() if plus else TI2243CascadeRadar.coloradar()
    )
    transceivers = _antenna_config(root / "antenna_cfg.txt")
    mimo = TDM(numTx=transceivers.numTx, numRx=transceivers.numRx)
    radar = replace(
        base,
        waveform=waveform,
        sampler=sampler,
        transceivers=transceivers,
        mimo=mimo,
        calibration=Calibration(),
        name="coloradar_plus_cascade" if plus else "coloradar_cascade",
        profileName="ColoRadar+" if plus else "ColoRadar",
    )
    if applyCalibration:
        radar = replace(
            radar,
            calibration=load_coloradar_calibration(root, radar=radar),
        )
    return radar


def read_coloradar_frame(
    path: PathLike,
    *,
    radar: TI2243CascadeRadar,
    timestamp: float | None = None,
    frameId: int | str | None = None,
) -> ADCFrame:
    """Decode one ColoRadar cascade ADC bin into canonical dimensions."""

    raw = np.fromfile(Path(path), dtype="<i2")
    expected = (
        radar.mimo.numTx
        * radar.mimo.numRx
        * radar.sampler.numLoops
        * radar.sampler.numSamples
        * 2
    )
    if raw.size != expected:
        raise ValueError(
            f"ADC file contains {raw.size} int16 values; expected {expected}."
        )
    raw = raw.reshape(
        radar.mimo.numTx,
        radar.mimo.numRx,
        radar.sampler.numLoops,
        radar.sampler.numSamples,
        2,
    )
    adc = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
    canonical = np.transpose(adc, (2, 0, 1, 3)).astype(np.complex64, copy=False)
    return ADCFrame(
        canonical,
        ("loop", "emission", "rx", "sample"),
        radar=radar,
        timestamp=timestamp,
        frameId=frameId,
        metadata={"source": str(Path(path))},
    )


class ColoRadarReader:
    """Sequence reader for an original ColoRadar cascade recording."""

    def __init__(
        self,
        datasetRoot: PathLike,
        sequence: str,
        *,
        applyCalibration: bool = True,
        plus: bool = False,
    ) -> None:
        self.datasetRoot = Path(datasetRoot)
        self.sequence = self.datasetRoot / sequence
        self.calibrationDir = self.datasetRoot / "calib" / "cascade"
        self._radar = radar_from_coloradar(
            self.calibrationDir, plus=plus, applyCalibration=applyCalibration
        )
        dataDir = self.sequence / "cascade" / "adc_samples" / "data"
        self.files = sorted(
            dataDir.glob("frame_*.bin"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not self.files:
            raise FileNotFoundError(f"No cascade ADC frames found in {dataDir}.")
        timestampPath = dataDir.parent / "timestamps.txt"
        self.timestamps = self._timestamps(timestampPath)

    @staticmethod
    def _timestamps(path: Path) -> list[float]:
        if not path.is_file():
            return []
        values = [
            float(line.strip().split()[0])
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if values and max(abs(value) for value in values) > 1e12:
            values = [value * 1e-9 for value in values]
        return values

    @property
    def radar(self) -> TI2243CascadeRadar:
        return self._radar

    def __len__(self) -> int:
        return len(self.files)

    def read(self, frameIndex: int) -> ADCFrame:
        path = self.files[frameIndex]
        timestamp = (
            self.timestamps[frameIndex] if frameIndex < len(self.timestamps) else None
        )
        return read_coloradar_frame(
            path,
            radar=self.radar,
            timestamp=timestamp,
            frameId=int(path.stem.split("_")[-1]),
        )


class ColoRadarPlusReader(ColoRadarReader):
    """Sequence reader for ColoRadar+ (same cascade ADC encoding)."""

    def __init__(
        self,
        datasetRoot: PathLike,
        sequence: str,
        *,
        applyCalibration: bool = True,
    ) -> None:
        super().__init__(
            datasetRoot,
            sequence,
            applyCalibration=applyCalibration,
            plus=True,
        )
