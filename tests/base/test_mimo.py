from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyradar.base import (
    BPM,
    DDM,
    FMCW,
    SIMO,
    TDM,
    Calibration,
    ProcessingConfig,
    Radar,
    Sampler,
    Transceivers,
)
from pyradar.base.processing import FFTConfig
from pyradar.base.waveform import SPEED_OF_LIGHT
from pyradar.rsp import doppler_fft, unwrap_tdm_velocity


def _relative_error(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def test_simo_decode() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5, 1, 3, 7)) + 1j * rng.normal(size=(5, 1, 3, 7))
    decoded = SIMO(numRx=3).decode(data)
    assert _relative_error(decoded, data[:, 0]) < 1e-12


def test_tdm_arbitrary_tx_order_decode() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(6, 3, 2, 5)) + 1j * rng.normal(size=(6, 3, 2, 5))
    mimo = TDM(numTx=3, numRx=2, txOrder=(2, 0, 1))
    encoded = np.empty_like(source)
    for emission, txId in enumerate(mimo.txOrder):
        encoded[:, emission] = source[:, txId]
    decoded = mimo.decode(encoded).reshape(6, 3, 2, 5)
    assert _relative_error(decoded, source) < 1e-12


def test_bpm_hadamard_decode_error_below_1e6() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(8, 2, 4, 6)) + 1j * rng.normal(size=(8, 2, 4, 6))
    mimo = BPM(numRx=4)
    encoded = np.einsum("et,ltr...->ler...", np.asarray(mimo.codeMatrix), source)
    decoded = mimo.decode(encoded).reshape(source.shape)
    assert _relative_error(decoded, source) < 1e-6


def test_ddm_linear_phase_decode_error_below_1e6() -> None:
    rng = np.random.default_rng(4)
    mimo = DDM(
        numTx=3,
        numRx=2,
        dopplerOffsets=(0.0, 0.25, 0.5),
        codeLength=4,
    )
    source = rng.normal(size=(5, 3, 2, 7)) + 1j * rng.normal(size=(5, 3, 2, 7))
    encoded = np.einsum("tl,btr...->blr...", mimo.codes, source)
    raw = encoded.reshape(20, 1, 2, 7)
    decoded = mimo.decode(raw).reshape(source.shape)
    assert _relative_error(decoded, source) < 1e-6


def test_tdm_phase_correction_uses_true_emission_times() -> None:
    mimo = TDM(
        numTx=3,
        numRx=1,
        txOrder=(2, 0, 1),
        emissionTimeOffsets=(0.0, 7e-6, 19e-6),
        cyclePeriod=30e-6,
    )
    size = 8
    correction = mimo.phase_correction(size, 10e-6)
    frequency = np.fft.fftshift(np.fft.fftfreq(size, d=30e-6))
    physicalOffsets = np.asarray([7e-6, 19e-6, 0.0])
    expected = np.exp(-2j * np.pi * frequency[:, None] * physicalOffsets)
    np.testing.assert_allclose(correction, expected, atol=1e-12)


def test_overlap_phase_centers_unwrap_tdm_velocity() -> None:
    waveform = FMCW(77e9, 20e12, 2e-6, 40e-6, 10e-6)
    sampler = Sampler(numSamples=8, numLoops=16, sampleRate=2e6)
    center = waveform.startFrequency + waveform.slope * (
        waveform.adcStartTime + sampler.captureDuration / 2
    )
    wavelength = SPEED_OF_LIGHT / center
    rx = np.column_stack((np.zeros(4), np.arange(4) * wavelength / 2, np.zeros(4)))
    tx = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.5 * wavelength, 0.0]])
    radar = Radar(
        waveform=waveform,
        sampler=sampler,
        transceivers=Transceivers(txPositions=tx, rxPositions=rx),
        mimo=TDM(numTx=2, numRx=4),
        calibration=Calibration(overlapPairs=np.asarray([[3, 4]])),
        processing=ProcessingConfig(
            fft=FFTConfig(
                dopplerFftSize=16,
                dopplerWindow="rectangular",
                removeDopplerMean=False,
            )
        ),
    )
    aliasedSignedBin = -3
    aliasedFrequency = aliasedSignedBin / (sampler.numLoops * radar.slowTimeInterval)
    trueFrequency = aliasedFrequency + 1.0 / radar.slowTimeInterval
    offsets = radar.mimo.tx_time_offsets(waveform.chirpInterval)
    decoded = np.empty((sampler.numLoops, 8), dtype=np.complex128)
    for loop in range(sampler.numLoops):
        for channel, (txId, _) in enumerate(radar.mimo.channelMap):
            time = loop * radar.slowTimeInterval + offsets[txId]
            decoded[loop, channel] = np.exp(2j * np.pi * trueFrequency * time)
    spectrum = doppler_fft(
        decoded,
        fftSize=16,
        loopAxis=0,
        window="rectangular",
        removeMean=False,
    )
    dopplerBin = aliasedSignedBin + sampler.numLoops // 2
    decision = unwrap_tdm_velocity(
        spectrum[dopplerBin], radar=radar, dopplerBin=dopplerBin
    )
    assert decision.ambiguityOrder == 1
    assert decision.frequency == pytest.approx(trueFrequency)
    assert decision.velocity == pytest.approx(trueFrequency * wavelength / 2)
    assert decision.score < 1e-20
    zero = unwrap_tdm_velocity(np.zeros(8, dtype=complex), radar=radar, dopplerBin=8)
    assert zero.ambiguityOrder == 0 and np.isinf(zero.score)
    with pytest.raises(ValueError, match="virtual-channel"):
        unwrap_tdm_velocity(np.ones(7), radar=radar, dopplerBin=8)
    with pytest.raises(ValueError, match="outside"):
        unwrap_tdm_velocity(np.ones(8), radar=radar, dopplerBin=99)
    with pytest.raises(ValueError, match="nonempty"):
        unwrap_tdm_velocity(np.ones(8), radar=radar, dopplerBin=8, ambiguityOrders=[])
    with pytest.raises(ValueError, match="integers"):
        unwrap_tdm_velocity(
            np.ones(8), radar=radar, dopplerBin=8, ambiguityOrders=[0.5]
        )
    with pytest.raises(ValueError, match="invalid"):
        unwrap_tdm_velocity(
            np.ones(8),
            radar=replace(
                radar, calibration=Calibration(overlapPairs=np.asarray([[3, 99]]))
            ),
            dopplerBin=8,
        )
    with pytest.raises(ValueError, match="arrayPhase"):
        unwrap_tdm_velocity(
            np.ones(8),
            radar=replace(
                radar,
                calibration=Calibration(
                    overlapPairs=np.asarray([[3, 4]]), arrayPhase=np.ones(7)
                ),
            ),
            dopplerBin=8,
        )


def test_mimo_configuration_and_input_validation(ula_radar) -> None:
    assert SIMO(2).phase_correction(4, 1e-6).shape == (4, 2)
    assert SIMO(2).slow_time_interval(3e-6) == 3e-6
    with pytest.raises(ValueError):
        SIMO(0)
    with pytest.raises(ValueError, match="dimensions"):
        SIMO(2).decode(np.ones((2, 1, 2)))
    with pytest.raises(ValueError, match="emissions"):
        SIMO(2).decode(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="RX"):
        SIMO(2).decode(np.ones((2, 1, 3, 1)))

    for kwargs in (
        {"numTx": 0, "numRx": 1},
        {"numTx": 2, "numRx": 1, "txOrder": (0, 0)},
        {
            "numTx": 2,
            "numRx": 1,
            "emissionTimeOffsets": (0.0,),
        },
        {
            "numTx": 2,
            "numRx": 1,
            "emissionTimeOffsets": (1e-6, 0.5e-6),
        },
        {"numTx": 2, "numRx": 1, "cyclePeriod": 0.0},
    ):
        with pytest.raises(ValueError):
            TDM(**kwargs)
    tdm = TDM(2, 1, cyclePeriod=10e-6)
    assert tdm.slow_time_interval(2e-6) == 10e-6
    np.testing.assert_allclose(tdm.tx_time_offsets(2e-6), [0.0, 2e-6])

    with pytest.raises(ValueError):
        BPM(0)
    with pytest.raises(ValueError, match="codeMatrix"):
        BPM(1, codeMatrix=((1.0, 1.0), (2.0, 2.0)))
    assert BPM(1).slow_time_interval(2e-6) == 4e-6

    for arguments in (
        (0, 1, (), 1),
        (2, 1, (0.0,), 2),
        (2, 1, (0.0, 0.0), 2),
    ):
        with pytest.raises(ValueError):
            DDM(*arguments)
    ddm = DDM(2, 1, (0.0, 0.5), 2)
    with pytest.raises(ValueError, match="divisible"):
        ddm.decode(np.ones((3, 1, 1, 2), dtype=complex))
    assert ddm.numEmissions == 1
    assert ddm.slow_time_interval(2e-6) == 4e-6

    simoRadar = Radar(
        FMCW(77e9, 1e12, 0.0, 10e-6),
        Sampler(4, 1, 1e6),
        Transceivers(np.asarray([[0, 0, 0]]), np.asarray([[0, 0, 0]])),
        SIMO(1),
    )
    with pytest.raises(TypeError, match="TDM"):
        unwrap_tdm_velocity(np.ones(1), radar=simoRadar, dopplerBin=0)
    with pytest.raises(ValueError, match="overlapPairs"):
        unwrap_tdm_velocity(
            np.ones(ula_radar.virtualArray.shape[0]),
            radar=ula_radar,
            dopplerBin=8,
        )
