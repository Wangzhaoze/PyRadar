from __future__ import annotations

import numpy as np
import pytest

from pyradar.base import ADCFrame, Calibration, PointCloud, RadarCube


def test_adc_frame_requires_explicit_dimensions(ula_radar) -> None:
    data = np.zeros((16, 2, 4, 64), dtype=complex)
    with pytest.raises(ValueError, match="explicit dims"):
        ADCFrame.from_array(data, dims=None, radar=ula_radar)
    frame = ADCFrame.from_array(
        np.transpose(data, (3, 0, 2, 1)),
        dims=("sample", "loop", "rx", "emission"),
        radar=ula_radar,
    ).canonical()
    assert frame.dims == ("loop", "emission", "rx", "sample")
    assert frame.data.shape == data.shape


def test_adc_frame_rejects_model_shape_mismatch(ula_radar) -> None:
    frame = ADCFrame(
        np.zeros((15, 2, 4, 64)),
        ("loop", "emission", "rx", "sample"),
        radar=ula_radar,
    )
    with pytest.raises(ValueError, match="does not match Radar"):
        frame.canonical()


def test_radar_cube_coordinates_and_named_transpose() -> None:
    cube = RadarCube(
        np.zeros((3, 4)),
        ("range", "doppler"),
        "range_doppler",
        coords={"range": np.arange(3), "doppler": np.arange(4)},
    )
    transposed = cube.transpose("doppler", "range")
    assert transposed.shape == (4, 3)
    with pytest.raises(ValueError):
        cube.transpose("range", "range")


def test_calibration_shapes_are_strict() -> None:
    data = np.ones((4, 2, 3, 8), dtype=complex)
    calibration = Calibration(adcGain=np.ones((2, 3)), adcPhase=np.zeros((2, 3)))
    np.testing.assert_allclose(calibration.apply_adc(data), data)
    with pytest.raises(ValueError, match="adcGain shape"):
        Calibration(adcGain=np.ones(3)).apply_adc(data)
    with pytest.raises(ValueError, match="rangeCoupling shape"):
        Calibration(rangeCoupling=np.ones((6, 7))).apply_range(np.ones((4, 6, 8)))


def test_pointcloud_numpy_and_polar_properties() -> None:
    cloud = PointCloud(
        xyz=[[1, 0, 0], [0, 1, 1]],
        power=[2, 3],
        snr=[4, 5],
        radialVelocity=[-1, 2],
        sourceBins=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    assert cloud.to_numpy().shape == (2, 6)
    np.testing.assert_allclose(cloud.range, [1, np.sqrt(2)])
    np.testing.assert_allclose(cloud.azimuth, [0, np.pi / 2])
