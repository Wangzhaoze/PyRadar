"""RAMPCNN/CRUW AWR1843 MATLAB frame adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyradar.base import ADCFrame, AWR1843Radar

from .base import PathLike


def read_rampcnn_frame(
    path: PathLike,
    *,
    radar: AWR1843Radar | None = None,
    frameId: int | str | None = None,
) -> ADCFrame:
    """Decode ``adcData(sample, loop, rx, tx)`` from a RAMPCNN MAT file."""

    model = radar or AWR1843Radar.rampcnn()
    values = loadmat(Path(path))
    if "adcData" not in values:
        raise KeyError(f"adcData not found in {path}.")
    adc = np.asarray(values["adcData"])
    expected = (
        model.sampler.numSamples,
        model.sampler.numLoops,
        model.mimo.numRx,
        model.mimo.numTx,
    )
    if adc.shape != expected:
        raise ValueError(
            f"RAMPCNN adcData shape {adc.shape} does not match {expected}."
        )
    canonical = np.transpose(adc, (1, 3, 2, 0)).astype(np.complex64, copy=False)
    return ADCFrame(
        canonical,
        ("loop", "emission", "rx", "sample"),
        radar=model,
        frameId=frameId,
        metadata={"source": str(Path(path))},
    )


class RAMPCNNReader:
    """Numeric-order reader for a RAMPCNN ``radar_raw_frame`` directory."""

    def __init__(self, root: PathLike) -> None:
        path = Path(root)
        dataDir = (
            path / "radar_raw_frame" if (path / "radar_raw_frame").is_dir() else path
        )
        self.files = sorted(dataDir.glob("*.mat"), key=lambda item: int(item.stem))
        if not self.files:
            raise FileNotFoundError(f"No RAMPCNN MAT frames found in {dataDir}.")
        self._radar = AWR1843Radar.rampcnn()

    @property
    def radar(self) -> AWR1843Radar:
        return self._radar

    def __len__(self) -> int:
        return len(self.files)

    def read(self, frameIndex: int) -> ADCFrame:
        path = self.files[frameIndex]
        return read_rampcnn_frame(path, radar=self.radar, frameId=int(path.stem))
