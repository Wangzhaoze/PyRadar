from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyradar.base import Radar, TI2243CascadeRadar
from pyradar.base.waveform import SPEED_OF_LIGHT


def test_radelft_profile_derives_physics_from_capture_metadata() -> None:
    radar = Radar.radelft_cascade()
    assert isinstance(radar, TI2243CascadeRadar)
    assert radar.virtualArray.shape == (192, 3)
    assert radar.rangeAxis.shape == (500,)
    assert radar.velocityAxis.shape == (128,)
    assert np.isclose(radar.rangeBinSize, 0.10036834, rtol=1e-6)
    expectedBandwidth = radar.waveform.slope * 256 / 12e6
    assert np.isclose(radar.rangeResolution, SPEED_OF_LIGHT / (2 * expectedBandwidth))
    assert radar.maxUnambiguousRange > radar.rangeAxis[-1]
    assert radar.profileName == "RaDelft"


def test_coloradar_and_rampcnn_profiles_are_distinct_hardware_instances() -> None:
    coloradar = Radar.coloradar_cascade()
    plus = Radar.coloradar_plus_cascade()
    rampcnn = Radar.awr1843_rampcnn()
    assert coloradar.profileName == "ColoRadar"
    assert plus.profileName == "ColoRadar+"
    assert coloradar.mimo.numTx == 12
    assert rampcnn.mimo.numTx == 2
    assert rampcnn.arrayGeometry == "ula"


def test_radar_is_frozen(ula_radar) -> None:
    with pytest.raises(FrozenInstanceError):
        ula_radar.name = "changed"
    with pytest.raises(FrozenInstanceError):
        ula_radar.processing.retainRangeAngle = True


def test_derived_velocity_and_fov(ula_radar) -> None:
    expected = ula_radar.wavelength / (
        2 * ula_radar.processing.fft.dopplerFftSize * ula_radar.slowTimeInterval
    )
    assert np.isclose(ula_radar.velocityBinSize, expected)
    assert np.isclose(
        ula_radar.maxUnambiguousVelocity,
        ula_radar.wavelength / (4 * ula_radar.slowTimeInterval),
    )
    assert ula_radar.arrayGeometry == "ula"
    assert np.isclose(ula_radar.unambiguousFov["azimuth"][1], np.pi / 2)


def test_from_config(tmp_path) -> None:
    config = tmp_path / "radar.yaml"
    config.write_text(
        """
name: config_simo
waveform:
  startFrequency: 77000000000.0
  slope: 1000000000000.0
  adcStartTime: 0.0
  rampEndTime: 0.00001
  idleTime: 0.000002
sampler:
  numSamples: 8
  numLoops: 4
  sampleRate: 1000000.0
transceivers:
  txPositions: [[0.0, 0.0, 0.0]]
  rxPositions: [[0.0, 0.0, 0.0], [0.0, 0.002, 0.0]]
mimo:
  type: simo
  numRx: 2
processing:
  fft:
    rangeFftSize: 16
    dopplerFftSize: 4
""".strip(),
        encoding="utf-8",
    )
    radar = Radar.from_config(config)
    assert radar.name == "config_simo"
    assert radar.rangeAxis.shape == (16,)
    assert radar.virtualArray.shape == (2, 3)
