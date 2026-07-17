from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyradar.utils.io import (
    ColoRadarPlusReader,
    RaDelftReader,
    RAMPCNNReader,
    radar_from_coloradar,
    read_coloradar_frame,
)


def _require_local_datasets() -> None:
    if os.environ.get("PYRADAR_RUN_DATASET_TESTS") != "1":
        pytest.skip("Set PYRADAR_RUN_DATASET_TESTS=1 for local dataset regression.")


def _rd_peak(result) -> tuple[int, int]:
    cube = result.rangeDopplerCube
    assert cube is not None
    power = np.mean(np.abs(cube.data) ** 2, axis=2)
    return tuple(
        int(value) for value in np.unravel_index(np.argmax(power), power.shape)
    )


def _assert_processed(result) -> None:
    assert result.rangeDopplerCube is not None
    assert np.isfinite(result.pointCloud.to_numpy()).all()
    assert len(result.pointCloud) > 0


def test_bundled_coloradar_frame_and_metadata() -> None:
    root = Path("docs/examples/data/ColoRadar")
    calibration = root / "calib" / "cascade"
    framePath = (
        root
        / "2_28_2021_outdoors_run0"
        / "cascade"
        / "adc_samples"
        / "data"
        / "frame_0.bin"
    )
    if not framePath.is_file():
        pytest.skip("Licensed ColoRadar LFS frame is unavailable.")
    radar = radar_from_coloradar(calibration, applyCalibration=True)
    frame = read_coloradar_frame(framePath, radar=radar)
    assert frame.data.shape == (16, 12, 16, 256)
    assert frame.dims == ("loop", "emission", "rx", "sample")
    assert radar.calibration.rangeCoupling.shape == (192, 256)
    _assert_processed(radar.process_adc(frame))


@pytest.mark.integration
def test_local_coloradar_plus_three_frames_against_public_heatmap() -> None:
    _require_local_datasets()
    root = Path(r"D:\Datasets\ColoRadarPlus")
    if not root.is_dir():
        pytest.skip("Local ColoRadar+ dataset is unavailable.")
    reader = ColoRadarPlusReader(root, "c4c_garage_run0")
    results = [reader.radar.process_adc(reader.read(index)) for index in range(3)]
    for result in results:
        _assert_processed(result)
        assert result.rangeDopplerCube.shape == (256, 16, 192)
    heatmapPath = (
        root / "c4c_garage_run0" / "cascade" / "heatmaps" / "data" / "heatmap_0.bin"
    )
    heatmap = np.fromfile(heatmapPath, dtype=np.float32).reshape(32, 128, 128, 2)
    referenceRangeBin = int(np.argmax(np.max(heatmap[..., 0], axis=(0, 1))))
    assert abs(_rd_peak(results[0])[0] - referenceRangeBin) <= 1


@pytest.mark.integration
def test_local_radelft_reader_three_frames() -> None:
    _require_local_datasets()
    root = Path(r"D:\Datasets\RaDelft\Scene6\RawData")
    if not root.is_dir():
        pytest.skip("Local RaDelft dataset is unavailable.")
    reader = RaDelftReader(root)
    results = [reader.radar.process_adc(reader.read(index)) for index in (1, 2, 3)]
    for result in results:
        _assert_processed(result)
        assert result.adcFrame.data.shape == (128, 12, 16, 256)
        assert result.rangeDopplerCube.shape == (500, 128, 192)
    reference = loadmat(
        root.parent / "RadarCubes" / "Pow_Frame_1.mat",
        variable_names=["radarCube"],
    )["radarCube"]
    referencePower = np.max(reference, axis=2)
    referencePeak = tuple(
        int(value)
        for value in np.unravel_index(np.argmax(referencePower), referencePower.shape)
    )
    assert reference.shape[:2] == results[0].rangeDopplerCube.shape[:2]
    assert _rd_peak(results[0]) == referencePeak


@pytest.mark.integration
def test_local_rampcnn_reader_three_frames() -> None:
    _require_local_datasets()
    root = Path(r"D:\Datasets\RAMPCNN\2019_05_29_pcms005")
    if not root.is_dir():
        pytest.skip("Local RAMPCNN dataset is unavailable.")
    reader = RAMPCNNReader(root)
    results = [reader.radar.process_adc(reader.read(index)) for index in range(3)]
    peaks = []
    for result in results:
        _assert_processed(result)
        assert result.adcFrame.data.shape == (255, 2, 4, 128)
        assert result.rangeDopplerCube.shape == (128, 256, 8)
        peaks.append(_rd_peak(result))
    assert np.max(np.ptp(np.asarray(peaks), axis=0)) <= 2
