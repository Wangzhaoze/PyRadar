from __future__ import annotations

import numpy as np
import pytest

from pyradar.rsp import (
    ca_cfar_1d,
    ca_cfar_2d,
    cfar_1d,
    cfar_2d,
    goca_cfar_1d,
    goca_cfar_2d,
    os_cfar_1d,
    os_cfar_2d,
    soca_cfar_1d,
    soca_cfar_2d,
)


def test_all_cfar_variants_return_noise_threshold_and_snr() -> None:
    power = np.ones((64, 32))
    power[30, 10] = 100.0
    for method in ("ca", "goca", "soca", "os"):
        result = cfar_2d(
            power,
            method=method,
            trainingCells=(4, 3),
            guardCells=(1, 1),
            pfa=1e-3,
        )
        assert result.noise.shape == power.shape
        assert result.threshold.shape == power.shape
        assert result.snr.shape == power.shape
        assert result.detections[30, 10]


def test_ca_cfar_empirical_false_alarm_rate_is_statistically_consistent() -> None:
    rng = np.random.default_rng(2025)
    pfa = 1e-3
    power = rng.exponential(size=(768, 384))
    result = cfar_2d(
        power,
        method="ca",
        trainingCells=(5, 4),
        guardCells=(2, 1),
        pfa=pfa,
        peakGrouping=False,
    )
    valid = np.isfinite(result.threshold)
    trials = int(np.count_nonzero(valid))
    alarms = int(np.count_nonzero(result.detections & valid))
    mean = trials * pfa
    sigma = np.sqrt(trials * pfa * (1.0 - pfa))
    assert abs(alarms - mean) <= 6.0 * sigma


def test_one_dimensional_cfar_and_peak_grouping() -> None:
    power = np.ones(128)
    power[60:63] = [10, 30, 20]
    result = cfar_1d(
        power,
        trainingCells=8,
        guardCells=2,
        pfa=1e-2,
        peakGrouping=True,
    )
    assert result.detections[61]
    assert np.count_nonzero(result.detections[60:63]) == 1


@pytest.mark.parametrize("method", ["ca", "goca", "soca", "os"])
def test_one_dimensional_variants_and_wrapped_edges(method) -> None:
    power = np.ones((2, 64))
    power[:, 0] = 50.0
    result = cfar_1d(
        power,
        method=method,
        trainingCells=5,
        guardCells=1,
        pfa=1e-2,
        rankFraction=0.7,
        axis=1,
        wrap=True,
    )
    assert result.detections[:, 0].all()
    assert np.isfinite(result.threshold).all()


def test_cfar_convenience_functions_and_validation() -> None:
    vector = np.ones(64)
    image = np.ones((32, 16))
    for detector in (ca_cfar_1d, goca_cfar_1d, soca_cfar_1d, os_cfar_1d):
        assert (
            detector(vector, trainingCells=4, guardCells=1).noise.shape == vector.shape
        )
    for detector in (ca_cfar_2d, goca_cfar_2d, soca_cfar_2d, os_cfar_2d):
        result = detector(image, trainingCells=(3, 2), guardCells=(1, 1))
        assert result.noise.shape == image.shape
    with pytest.raises(ValueError, match="finite"):
        cfar_1d(np.asarray([1.0, -1.0]), trainingCells=1)
    with pytest.raises(ValueError, match="Invalid"):
        cfar_1d(vector, trainingCells=0)
    with pytest.raises(ValueError, match="Unsupported"):
        cfar_1d(vector, method="bad", trainingCells=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        cfar_2d(vector)
    with pytest.raises(ValueError, match="Invalid"):
        cfar_2d(image, trainingCells=(0, 2))
    with pytest.raises(ValueError, match="Unsupported"):
        cfar_2d(image, method="bad", trainingCells=(2, 2))
