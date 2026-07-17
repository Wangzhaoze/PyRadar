from __future__ import annotations

import numpy as np
import pytest

from pyradar.base import (
    FMCW,
    TDM,
    CFARConfig,
    DoAConfig,
    FFTConfig,
    ProcessingConfig,
    Radar,
    Sampler,
    Transceivers,
)
from pyradar.base.waveform import SPEED_OF_LIGHT


@pytest.fixture
def ula_radar() -> Radar:
    waveform = FMCW(
        startFrequency=77e9,
        slope=20e12,
        adcStartTime=2e-6,
        rampEndTime=40e-6,
        idleTime=10e-6,
    )
    sampler = Sampler(numSamples=64, numLoops=16, sampleRate=2e6)
    center = waveform.startFrequency + waveform.slope * (
        waveform.adcStartTime + sampler.captureDuration / 2.0
    )
    wavelength = SPEED_OF_LIGHT / center
    tx = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0 * wavelength, 0.0]])
    rx = np.column_stack((np.zeros(4), np.arange(4) * wavelength / 2.0, np.zeros(4)))
    processing = ProcessingConfig(
        fft=FFTConfig(
            rangeFftSize=64,
            dopplerFftSize=16,
            azimuthFftSize=128,
            elevationFftSize=16,
            rangeWindow="rectangular",
            dopplerWindow="rectangular",
            angleWindow="rectangular",
            removeDopplerMean=False,
        ),
        cfar=CFARConfig(
            method="ca",
            trainingCells=(2, 2),
            guardCells=(1, 1),
            pfa=1e-3,
            maxDetections=16,
        ),
        doa=DoAConfig(
            method="auto",
            azimuthFov=(-np.pi / 2, np.pi / 2),
            elevationFov=(-0.1, 0.1),
            azimuthBins=181,
            elevationBins=9,
        ),
    )
    return Radar(
        waveform=waveform,
        sampler=sampler,
        transceivers=Transceivers(txPositions=tx, rxPositions=rx),
        mimo=TDM(numTx=2, numRx=4, txOrder=(1, 0)),
        processing=processing,
        name="synthetic_ula",
    )


def synthesize_tdm_target(
    radar: Radar,
    *,
    rangeBin: int,
    dopplerBin: int,
    azimuth: float,
    elevation: float = 0.0,
    amplitude: complex = 1.0,
) -> np.ndarray:
    from pyradar.rsp.doa import steering_vector

    samples = np.arange(radar.sampler.numSamples)
    rangeTone = np.exp(2j * np.pi * rangeBin * samples / radar.sampler.numSamples)
    frequency = dopplerBin / (radar.sampler.numLoops * radar.slowTimeInterval)
    response = steering_vector(radar.virtualArray, radar.wavelength, azimuth, elevation)
    adc = np.empty(
        (
            radar.sampler.numLoops,
            radar.mimo.numEmissions,
            radar.mimo.numRx,
            radar.sampler.numSamples,
        ),
        dtype=np.complex128,
    )
    offsets = radar.mimo.tx_time_offsets(radar.waveform.chirpInterval)
    for loop in range(radar.sampler.numLoops):
        for emission, txId in enumerate(radar.mimo.txOrder):
            time = loop * radar.slowTimeInterval + offsets[txId]
            motion = np.exp(2j * np.pi * frequency * time)
            for rxId in range(radar.mimo.numRx):
                channel = txId * radar.mimo.numRx + rxId
                adc[loop, emission, rxId] = (
                    amplitude * motion * response[channel] * rangeTone
                )
    return adc
