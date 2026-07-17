from __future__ import annotations

import numpy as np
import pytest

from pyradar.base import (
    BPM,
    DDM,
    FMCW,
    SIMO,
    TDM,
    DoAConfig,
    FFTConfig,
    ProcessingConfig,
    Radar,
    RadarCube,
    Sampler,
    Transceivers,
)
from pyradar.rsp import estimate_doa, range_doppler_fft
from pyradar.sim import PointTarget, PropagationPath, quantize_adc, simulate_adc


def _radar(mimo, *, numLoops: int = 16, array: bool = False) -> Radar:
    waveform = FMCW(77e9, 20e12, 2e-6, 40e-6, 10e-6)
    sampler = Sampler(64, numLoops, 2e6, bitDepth=12)
    wavelength = 299_792_458.0 / (
        waveform.startFrequency
        + waveform.slope * (waveform.adcStartTime + sampler.captureDuration / 2.0)
    )
    tx = np.zeros((mimo.numTx, 3))
    rx = np.zeros((mimo.numRx, 3))
    if array:
        rx[:, 1] = np.arange(mimo.numRx) * wavelength / 2.0
    return Radar(
        waveform,
        sampler,
        Transceivers(tx, rx),
        mimo,
        processing=ProcessingConfig(
            fft=FFTConfig(
                rangeFftSize=64,
                dopplerFftSize=(numLoops // mimo.codeLength)
                if isinstance(mimo, DDM)
                else numLoops,
                azimuthFftSize=256,
                rangeWindow="rectangular",
                dopplerWindow="rectangular",
                angleWindow="rectangular",
                removeDopplerMean=False,
            ),
            doa=DoAConfig(
                method="auto",
                azimuthFov=(-np.pi / 2.0, np.pi / 2.0),
                elevationFov=(-0.1, 0.1),
                azimuthBins=181,
                elevationBins=3,
            ),
        ),
        name="sim_test",
    )


@pytest.mark.parametrize(
    "mimo",
    [
        SIMO(numRx=2),
        TDM(numTx=2, numRx=2, txOrder=(1, 0)),
        BPM(numRx=2),
        DDM(numTx=2, numRx=2, dopplerOffsets=(0.0, 0.5), codeLength=2),
    ],
)
def test_simulator_encodes_all_mimo_schemes(mimo) -> None:
    radar = _radar(mimo)
    frame = simulate_adc(radar, PointTarget(np.asarray([12.0, 0.0, 0.0])))

    assert frame.data.shape == (
        radar.sampler.numLoops,
        radar.mimo.numEmissions,
        radar.mimo.numRx,
        radar.sampler.numSamples,
    )
    decoded = radar.mimo.decode(frame.data)
    reference = decoded[:, :1]
    np.testing.assert_allclose(
        decoded, np.broadcast_to(reference, decoded.shape), atol=1e-12
    )


def test_point_target_recovers_range_velocity_and_angle() -> None:
    radar = _radar(SIMO(numRx=8), numLoops=32, array=True)
    rangeBin = 12
    dopplerBin = 3
    azimuth = np.deg2rad(18.0)
    targetRange = rangeBin * radar.rangeBinSize
    direction = np.asarray([np.cos(azimuth), np.sin(azimuth), 0.0])
    # Positive Doppler is approaching, hence Cartesian velocity opposes direction.
    target = PointTarget(
        position=targetRange * direction,
        velocity=-dopplerBin * radar.velocityBinSize * direction,
        rcs=10.0,
    )
    frame = simulate_adc(radar, target)
    cube = range_doppler_fft(frame, radar=radar)

    assert isinstance(cube, RadarCube)
    power = np.mean(np.abs(cube.data) ** 2, axis=2)
    rangeIndex, dopplerIndex = np.unravel_index(np.argmax(power), power.shape)
    assert abs(rangeIndex - rangeBin) <= 1
    assert dopplerIndex == dopplerBin + radar.numSlowTimeSamples // 2
    doa = estimate_doa(cube.data[rangeIndex, dopplerIndex], radar=radar)
    assert abs(doa.azimuth[0] - azimuth) <= np.pi / 128.0


def test_noise_quantization_and_path_validation() -> None:
    radar = _radar(SIMO(numRx=1))
    first = simulate_adc(
        radar,
        PointTarget(np.asarray([8.0, 0.0, 0.0])),
        noisePower=0.2,
        seed=4,
    )
    second = simulate_adc(
        radar,
        PointTarget(np.asarray([8.0, 0.0, 0.0])),
        noisePower=0.2,
        seed=4,
    )
    np.testing.assert_array_equal(first.data, second.data)
    quantized = quantize_adc(first, fullScale=4.0)
    assert quantized.metadata["bitDepth"] == 12
    assert np.max(np.abs(quantized.data.real)) <= 4.0

    with pytest.raises(ValueError, match="unavailable TX"):
        from pyradar.sim import simulate_paths

        simulate_paths(radar, (PropagationPath(10.0, 1, 0),))
    with pytest.raises(ValueError, match="fullScale"):
        quantize_adc(first, fullScale=0.0)

    zero = simulate_adc(radar, PointTarget(np.asarray([8.0, 0.0, 0.0]), rcs=0.0))
    np.testing.assert_array_equal(quantize_adc(zero).data, zero.data)


def test_radar_simulate_convenience_method() -> None:
    radar = _radar(SIMO(numRx=1))
    target = PointTarget(np.asarray([8.0, 0.0, 0.0]))
    direct = simulate_adc(radar, target)
    modelAware = radar.simulate(target)
    np.testing.assert_allclose(modelAware.data, direct.data)


def test_true_tdm_emission_time_is_compensated() -> None:
    mimo = TDM(
        numTx=2,
        numRx=2,
        txOrder=(1, 0),
        emissionTimeOffsets=(0.0, 13e-6),
        cyclePeriod=50e-6,
    )
    radar = _radar(mimo, numLoops=32)
    dopplerBin = 3
    target = PointTarget(
        np.asarray([15.0, 0.0, 0.0]),
        np.asarray([-dopplerBin * radar.velocityBinSize, 0.0, 0.0]),
    )
    cube = range_doppler_fft(simulate_adc(radar, target), radar=radar)
    assert isinstance(cube, RadarCube)
    cell = np.unravel_index(
        np.argmax(np.mean(np.abs(cube.data) ** 2, axis=2)), cube.data.shape[:2]
    )
    normalized = cube.data[cell] / cube.data[cell][0]
    np.testing.assert_allclose(normalized, np.ones_like(normalized), atol=2e-3)
