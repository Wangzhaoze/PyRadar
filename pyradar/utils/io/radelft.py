"""RaDelft TI cascade metadata and segmented raw-bin reader."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyradar.base import FMCW, TDM, ADCFrame, Calibration, Sampler, TI2243CascadeRadar

from .base import PathLike


def radar_from_radelft_json(path: PathLike) -> TI2243CascadeRadar:
    """Create a RaDelft profile from mmWave Studio JSON metadata."""

    configPath = Path(path)
    with configPath.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    rf = config["mmWaveDevices"][0]["rfConfig"]
    profile = rf["rlProfiles"][0]["rlProfileCfg_t"]
    frame = rf["rlFrameCfg_t"]
    base = TI2243CascadeRadar.radelft()
    waveform = FMCW(
        startFrequency=float(profile["startFreqConst_GHz"]) * 1e9,
        slope=float(profile["freqSlopeConst_MHz_usec"]) * 1e12,
        adcStartTime=float(profile["adcStartTimeConst_usec"]) * 1e-6,
        rampEndTime=float(profile["rampEndTime_usec"]) * 1e-6,
        idleTime=float(profile["idleTimeConst_usec"]) * 1e-6,
    )
    sampler = Sampler(
        numSamples=int(profile["numAdcSamples"]),
        numLoops=int(frame["numLoops"]),
        sampleRate=float(profile["digOutSampleRate"]) * 1e3,
        framePeriod=float(frame["framePeriodicity_msec"]) * 1e-3,
    )
    numEmissions = int(frame["chirpEndIdx"]) - int(frame["chirpStartIdx"]) + 1
    if numEmissions != base.mimo.numTx:
        raise ValueError(
            f"RaDelft frame contains {numEmissions} emissions, expected {base.mimo.numTx}."
        )
    return replace(
        base,
        waveform=waveform,
        sampler=sampler,
        mimo=TDM(numTx=12, numRx=16),
        name=f"radelft_{configPath.stem}",
    )


def load_radelft_calibration(
    path: PathLike,
    *,
    radar: TI2243CascadeRadar,
    txIds: tuple[int, ...] = tuple(range(11, -1, -1)),
    rxOrder: tuple[int, ...] = (12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7),
    calibrationSlope: float = 35.003e12,
    calibrationRate: float = 12e6,
    interpolation: int = 5,
    phaseOnly: bool = True,
) -> Calibration:
    """Convert TI ``RangeMat``/``PeakValMat`` calibration to ADC terms."""

    values = loadmat(Path(path), simplify_cells=True)
    result = values.get("calibResult", values)
    rangeMatrix = np.asarray(result["RangeMat"])
    peakMatrix = np.asarray(result["PeakValMat"])
    if rangeMatrix.shape[0] < 12 or rangeMatrix.shape[1] < 16:
        raise ValueError("RaDelft RangeMat must contain at least 12 TX x 16 RX values.")
    referenceTx = txIds[0]
    frequency = np.empty((12, 16), dtype=float)
    coefficient = np.empty((12, 16), dtype=complex)
    for emissionId, txId in enumerate(txIds):
        deltaRange = rangeMatrix[txId, rxOrder] - rangeMatrix[referenceTx, rxOrder[0]]
        frequency[emissionId] = (
            2.0
            * np.pi
            * deltaRange
            * calibrationRate
            / radar.sampler.sampleRate
            * radar.waveform.slope
            / calibrationSlope
            / (radar.sampler.numSamples * interpolation)
        )
        coefficient[emissionId] = (
            peakMatrix[referenceTx, rxOrder[0]] / peakMatrix[txId, rxOrder]
        )
    if phaseOnly:
        coefficient /= np.maximum(np.abs(coefficient), np.finfo(float).tiny)
    return Calibration(
        adcGain=np.abs(coefficient),
        adcPhase=np.angle(coefficient),
        frequencySlope=frequency,
        overlapPairs=radar.calibration.overlapPairs,
    )


class RaDelftReader:
    """Random-access reader for segmented 4-device RaDelft raw captures."""

    rxOrder = np.asarray(
        [12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7],
        dtype=np.int64,
    )

    def __init__(
        self,
        rawDirectory: PathLike,
        *,
        configPath: PathLike | None = None,
        calibrationPath: PathLike | None = None,
    ) -> None:
        self.rawDirectory = Path(rawDirectory)
        configs = list(self.rawDirectory.glob("*.mmwave.json"))
        selectedConfig = (
            Path(configPath)
            if configPath is not None
            else (configs[0] if configs else None)
        )
        if selectedConfig is None:
            raise FileNotFoundError(f"No mmWave JSON found in {self.rawDirectory}.")
        radar = radar_from_radelft_json(selectedConfig)
        if calibrationPath is not None:
            radar = replace(
                radar,
                calibration=load_radelft_calibration(calibrationPath, radar=radar),
            )
        self._radar = radar
        pattern = re.compile(r"master_(\d+)_data\.bin$")
        self.segments: list[tuple[int, dict[str, Path], int]] = []
        bytesPerDeviceFrame = self._bytes_per_device_frame()
        for master in self.rawDirectory.glob("master_*_data.bin"):
            match = pattern.match(master.name)
            if match is None:
                continue
            segmentId = int(match.group(1))
            suffix = match.group(1)
            devices = {
                name: self.rawDirectory / f"{name}_{suffix}_data.bin"
                for name in ("master", "slave1", "slave2", "slave3")
            }
            if not all(path.is_file() for path in devices.values()):
                raise FileNotFoundError(f"Incomplete RaDelft segment {suffix}.")
            frameCounts = [
                path.stat().st_size // bytesPerDeviceFrame for path in devices.values()
            ]
            if len(set(frameCounts)) != 1:
                raise ValueError(
                    f"Device files disagree on frame count in segment {suffix}."
                )
            self.segments.append((segmentId, devices, int(frameCounts[0])))
        self.segments.sort(key=lambda item: item[0])
        if not self.segments:
            raise FileNotFoundError(
                f"No segmented RaDelft bins found in {self.rawDirectory}."
            )

    @property
    def radar(self) -> TI2243CascadeRadar:
        return self._radar

    def _bytes_per_device_frame(self) -> int:
        return (
            self.radar.sampler.numSamples
            * self.radar.sampler.numLoops
            * self.radar.mimo.numEmissions
            * 4
            * 2
            * np.dtype("<i2").itemsize
        )

    def __len__(self) -> int:
        return sum(frames for _, _, frames in self.segments)

    def _locate(self, frameIndex: int) -> tuple[dict[str, Path], int, int]:
        if frameIndex < 0:
            frameIndex += len(self)
        if not 0 <= frameIndex < len(self):
            raise IndexError(frameIndex)
        remaining = frameIndex
        for segmentId, devices, frames in self.segments:
            if remaining < frames:
                return devices, remaining, segmentId
            remaining -= frames
        raise RuntimeError("Frame location failed after bounds validation.")

    def _read_device(self, path: Path, localIndex: int) -> np.ndarray:
        scalarCount = self._bytes_per_device_frame() // np.dtype("<i2").itemsize
        raw = np.fromfile(
            path,
            dtype="<i2",
            count=scalarCount,
            offset=localIndex * self._bytes_per_device_frame(),
        )
        if raw.size != scalarCount:
            raise EOFError(f"Incomplete frame {localIndex} in {path}.")
        iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        # Binary order is RX-fastest, then sample, emission, loop.
        return iq.reshape(
            self.radar.sampler.numLoops,
            self.radar.mimo.numEmissions,
            self.radar.sampler.numSamples,
            4,
        ).transpose(0, 1, 3, 2)

    def read(self, frameIndex: int) -> ADCFrame:
        devices, localIndex, segmentId = self._locate(frameIndex)
        adc = np.concatenate(
            [
                self._read_device(devices[name], localIndex)
                for name in ("master", "slave1", "slave2", "slave3")
            ],
            axis=2,
        )
        adc = adc[:, :, self.rxOrder, :].astype(np.complex64, copy=False)
        timestamp = (
            frameIndex * self.radar.sampler.framePeriod
            if self.radar.sampler.framePeriod is not None
            else None
        )
        return ADCFrame(
            adc,
            ("loop", "emission", "rx", "sample"),
            radar=self.radar,
            timestamp=timestamp,
            frameId=frameIndex,
            metadata={"segment": segmentId, "localFrame": localIndex},
        )
