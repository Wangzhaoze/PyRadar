from __future__ import annotations

import numpy as np
import pytest

from pyradar.rsp import (
    coherent_integrate,
    micro_doppler_spectrogram,
    noncoherent_integrate,
    remove_static_clutter,
    stft,
    zoom_fft,
)


def test_stft_recovers_two_sided_complex_tone() -> None:
    sampleRate = 1024.0
    frequency = 128.0
    time = np.arange(256) / sampleRate
    tone = np.exp(2j * np.pi * frequency * time)
    result = stft(
        tone,
        sampleRate=sampleRate,
        segmentLength=64,
        overlap=32,
        fftSize=256,
        window="rectangular",
    )
    peak = np.unravel_index(np.argmax(result.power), result.power.shape)[-2]
    assert result.frequency[peak] == pytest.approx(frequency)
    assert result.spectrum.shape == (256, 7)


def test_micro_doppler_uses_radar_slow_time(ula_radar) -> None:
    frequency = 2.0 / (16 * ula_radar.slowTimeInterval)
    time = np.arange(16) * ula_radar.slowTimeInterval
    tone = np.exp(2j * np.pi * frequency * time)
    result = micro_doppler_spectrogram(
        tone,
        radar=ula_radar,
        segmentLength=8,
        overlap=4,
        fftSize=16,
        window="rectangular",
    )
    peak = np.unravel_index(np.argmax(result.power), result.power.shape)[-2]
    assert result.velocity is not None
    assert result.velocity[peak] == pytest.approx(2 * ula_radar.velocityBinSize)


def test_zoom_fft_resolves_narrow_frequency_interval() -> None:
    sampleRate = 1000.0
    frequency = 123.4
    samples = np.arange(200)
    tone = np.exp(2j * np.pi * frequency * samples / sampleRate)
    result = zoom_fft(
        tone,
        frequencyRange=(120.0, 126.0),
        sampleRate=sampleRate,
        fftSize=601,
        axis=0,
    )
    recovered = result.frequency[np.argmax(np.abs(result.spectrum))]
    assert recovered == pytest.approx(frequency, abs=0.02)


def test_preprocessing_integrations() -> None:
    signal = np.asarray([[1.0 + 1.0j, 2.0], [3.0 + 1.0j, 4.0]])
    clutterFree = remove_static_clutter(signal, slowTimeAxis=0)
    np.testing.assert_allclose(np.mean(clutterFree, axis=0), 0.0)
    np.testing.assert_allclose(coherent_integrate(signal, axis=0), (4.0 + 2.0j, 6.0))
    np.testing.assert_allclose(
        noncoherent_integrate(signal, axis=0), np.sum(np.abs(signal) ** 2, axis=0)
    )
    with pytest.raises(ValueError, match="sampleRate"):
        stft(np.ones(8), sampleRate=0.0, segmentLength=4)
