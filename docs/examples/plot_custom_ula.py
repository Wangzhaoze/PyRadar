"""
Custom TDM ULA: ADC to point cloud
==================================

Define a radar from physical parameters, synthesize one moving target, and run
the same model-aware pipeline used for recorded ADC captures.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

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
from pyradar.rsp.doa import steering_vector


def make_radar() -> Radar:
    """Create an eight-channel virtual ULA from two TX and four RX."""

    waveform = FMCW(
        startFrequency=77e9,
        slope=20e12,
        adcStartTime=2e-6,
        rampEndTime=40e-6,
        idleTime=10e-6,
    )
    sampler = Sampler(numSamples=64, numLoops=16, sampleRate=2e6)
    center = waveform.startFrequency + waveform.slope * (
        waveform.adcStartTime + sampler.captureDuration / 2
    )
    wavelength = SPEED_OF_LIGHT / center
    rx = np.column_stack((np.zeros(4), np.arange(4) * wavelength / 2, np.zeros(4)))
    tx = np.array([[0.0, 0.0, 0.0], [0.0, 2 * wavelength, 0.0]], dtype=float)
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
            trainingCells=(3, 2),
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
        retainRangeAngle=True,
    )
    return Radar(
        waveform=waveform,
        sampler=sampler,
        transceivers=Transceivers(txPositions=tx, rxPositions=rx),
        mimo=TDM(numTx=2, numRx=4, txOrder=(1, 0)),
        processing=processing,
        name="custom_tdm_ula",
    )


def synthesize_target(
    radar: Radar,
    *,
    rangeBin: int,
    dopplerBin: int,
    azimuth: float,
) -> np.ndarray:
    """Generate canonical TDM ADC while retaining emission-time phase."""

    sample = np.arange(radar.sampler.numSamples)
    rangeTone = np.exp(2j * np.pi * rangeBin * sample / radar.sampler.numSamples)
    dopplerFrequency = dopplerBin / (radar.sampler.numLoops * radar.slowTimeInterval)
    response = steering_vector(radar.virtualArray, radar.wavelength, azimuth, 0.0)
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
            motion = np.exp(2j * np.pi * dopplerFrequency * time)
            for rxId in range(radar.mimo.numRx):
                channel = txId * radar.mimo.numRx + rxId
                adc[loop, emission, rxId] = 100 * motion * response[channel] * rangeTone
    return adc


radar = make_radar()
adc = synthesize_target(
    radar,
    rangeBin=18,
    dopplerBin=-3,
    azimuth=np.deg2rad(24),
)
result = radar.process_adc(
    adc,
    dims=("loop", "emission", "rx", "sample"),
    frameId=0,
)

rd = result.rangeDopplerCube.power.mean(axis=2)
ra = result.rangeAngleCube.power
points = result.pointCloud

figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
axes[0].imshow(
    10 * np.log10(np.maximum(rd, np.finfo(float).tiny)),
    origin="lower",
    aspect="auto",
    extent=[
        radar.velocityAxis[0],
        radar.velocityAxis[-1],
        radar.rangeAxis[0],
        radar.rangeAxis[-1],
    ],
)
axes[0].set(xlabel="Radial velocity (m/s)", ylabel="Range (m)", title="RD")

axes[1].imshow(
    10 * np.log10(np.maximum(ra, np.finfo(float).tiny)),
    origin="lower",
    aspect="auto",
    extent=[
        np.rad2deg(result.rangeAngleCube.coords["azimuth"][0]),
        np.rad2deg(result.rangeAngleCube.coords["azimuth"][-1]),
        radar.rangeAxis[0],
        radar.rangeAxis[-1],
    ],
)
axes[1].set(xlabel="Azimuth (deg)", ylabel="Range (m)", title="RA")

axes[2].scatter(points.xyz[:, 0], points.xyz[:, 1], c=points.radialVelocity)
axes[2].set(xlabel="Forward x (m)", ylabel="Left y (m)", title="Point cloud")
axes[2].axis("equal")
plt.show()
