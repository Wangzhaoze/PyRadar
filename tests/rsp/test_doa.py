from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from pyradar.base import (
    FMCW,
    SIMO,
    DoAConfig,
    FFTConfig,
    ProcessingConfig,
    Radar,
    Sampler,
    Transceivers,
)
from pyradar.rsp import (
    combine_duplicate_channels,
    doa_bartlett,
    doa_capon,
    doa_esprit,
    doa_music,
    estimate_doa,
    spatial_covariance,
    spatial_smoothing,
    steering_vector,
)


def test_ula_fft_recovers_single_target(ula_radar) -> None:
    angle = np.deg2rad(18.0)
    signal = steering_vector(ula_radar.virtualArray, ula_radar.wavelength, angle)
    result = estimate_doa(signal, radar=ula_radar)
    assert result.method == "fft"
    assert abs(result.azimuth[0] - angle) <= np.pi / 64
    assert abs(result.elevation[0]) < 1e-12


def test_sparse_bartlett_recovers_azimuth_and_elevation() -> None:
    wavelength = 0.004
    positions = (
        np.asarray(
            [
                [0, 0, 0],
                [0, 0.5, 0],
                [0, 1.5, 0.5],
                [0, 2.5, 1.0],
                [0, 4.0, 0.25],
                [0, 5.5, 1.5],
            ],
            dtype=float,
        )
        * wavelength
    )
    radar = Radar(
        FMCW(75e9, 1e12, 0, 10e-6),
        Sampler(8, 4, 1e6),
        Transceivers([[0, 0, 0]], positions),
        SIMO(numRx=len(positions)),
        processing=ProcessingConfig(
            doa=DoAConfig(
                method="auto",
                azimuthFov=(-0.6, 0.6),
                elevationFov=(-0.4, 0.4),
                azimuthBins=121,
                elevationBins=81,
            )
        ),
    )
    azimuth, elevation = 0.23, -0.14
    signal = steering_vector(radar.virtualArray, radar.wavelength, azimuth, elevation)
    result = estimate_doa(signal, radar=radar)
    assert result.method == "bartlett"
    assert abs(result.azimuth[0] - azimuth) <= 0.011
    assert abs(result.elevation[0] - elevation) <= 0.011


def test_music_resolves_two_targets_and_spatial_smoothing() -> None:
    rng = np.random.default_rng(8)
    wavelength = 0.004
    positions = np.column_stack(
        (np.zeros(10), np.arange(10) * wavelength / 2, np.zeros(10))
    )
    angles = np.asarray([-0.25, 0.31])
    steering = steering_vector(positions, wavelength, angles)
    sources = rng.normal(size=(400, 2)) + 1j * rng.normal(size=(400, 2))
    snapshots = sources @ steering
    axis = np.linspace(-0.7, 0.7, 701)
    spectrum = doa_music(
        snapshots,
        arrayPositions=positions,
        wavelength=wavelength,
        azimuthAxis=axis,
        numSources=2,
    )[0]
    peaks, _ = find_peaks(spectrum, distance=80)
    recovered = axis[peaks[np.argsort(spectrum[peaks])[-2:]]]
    np.testing.assert_allclose(np.sort(recovered), angles, atol=0.01)
    smoothed = spatial_smoothing(snapshots, subarraySize=7)
    assert smoothed.shape == (7, 7)


def test_esprit_and_duplicate_phase_centers() -> None:
    wavelength = 0.004
    spacing = wavelength / 2
    positions = np.column_stack((np.zeros(8), np.arange(8) * spacing, np.zeros(8)))
    angle = -0.2
    signal = steering_vector(positions, wavelength, angle)
    recovered = doa_esprit(signal, numSources=1, spacing=spacing, wavelength=wavelength)
    np.testing.assert_allclose(recovered, [angle], atol=1e-6)

    duplicated = np.concatenate((signal, signal[:2]))
    duplicatePositions = np.vstack((positions, positions[:2]))
    combined, unique = combine_duplicate_channels(
        duplicated, duplicatePositions, policy="coherent"
    )
    assert combined.shape == (8,)
    assert unique.shape == (8, 3)
    np.testing.assert_allclose(combined, signal)


def test_complete_ura_uses_two_dimensional_fft() -> None:
    wavelength = 0.004
    positions = np.asarray(
        [
            [0.0, y * wavelength / 2, z * wavelength / 2]
            for z in range(3)
            for y in range(4)
        ]
    )
    radar = Radar(
        FMCW(75e9, 1e12, 0.0, 16e-6),
        Sampler(8, 4, 1e6),
        Transceivers(np.asarray([[0.0, 0.0, 0.0]]), positions),
        SIMO(len(positions)),
        processing=ProcessingConfig(
            fft=FFTConfig(azimuthFftSize=128, elevationFftSize=128),
            doa=DoAConfig(
                method="auto",
                azimuthFov=(-0.6, 0.6),
                elevationFov=(-0.4, 0.4),
                azimuthBins=121,
                elevationBins=81,
            ),
        ),
    )
    azimuth, elevation = 0.16, -0.11
    signal = steering_vector(positions, radar.wavelength, azimuth, elevation)
    result = estimate_doa(signal, radar=radar)
    assert radar.arrayGeometry == "ura"
    assert result.method == "fft"
    assert abs(result.azimuth[0] - azimuth) < 0.03
    assert abs(result.elevation[0] - elevation) < 0.03


def test_covariance_beamformers_and_explicit_model_methods(ula_radar) -> None:
    angle = 0.21
    steering = steering_vector(ula_radar.virtualArray, ula_radar.wavelength, angle)
    snapshots = np.vstack((steering, 2.0 * steering, 0.5j * steering))
    axis = np.linspace(-0.6, 0.6, 121)
    bartlett = doa_bartlett(
        snapshots,
        arrayPositions=ula_radar.virtualArray,
        wavelength=ula_radar.wavelength,
        azimuthAxis=axis,
    )
    capon = doa_capon(
        snapshots,
        arrayPositions=ula_radar.virtualArray,
        wavelength=ula_radar.wavelength,
        azimuthAxis=axis,
        diagonalLoading=1e-2,
    )
    assert axis[np.argmax(bartlett)] == pytest.approx(angle, abs=0.02)
    assert axis[np.argmax(capon)] == pytest.approx(angle, abs=0.02)
    assert estimate_doa(snapshots, radar=ula_radar, method="capon").method == "capon"
    assert estimate_doa(snapshots, radar=ula_radar, method="music").method == "music"
    assert estimate_doa(snapshots, radar=ula_radar, method="esprit").method == "esprit"


def test_doa_validation_duplicate_policies_and_forward_backward_covariance(
    ula_radar,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        steering_vector(np.ones(4), 0.004, 0.0)
    with pytest.raises(ValueError, match="positive"):
        steering_vector(np.zeros((2, 3)), 0.0, 0.0)
    with pytest.raises(ValueError, match="channel"):
        spatial_covariance(1.0)
    signal = steering_vector(ula_radar.virtualArray, ula_radar.wavelength, 0.1)
    covariance = spatial_covariance(signal, forwardBackward=True)
    assert covariance.shape == (signal.size, signal.size)
    with pytest.raises(ValueError, match="numSources"):
        doa_music(
            signal,
            arrayPositions=ula_radar.virtualArray,
            wavelength=ula_radar.wavelength,
            azimuthAxis=np.linspace(-1, 1, 10),
            numSources=signal.size,
        )
    with pytest.raises(ValueError, match="Invalid ESPRIT"):
        doa_esprit(signal, numSources=0, spacing=1.0, wavelength=1.0)
    with pytest.raises(ValueError, match="shape"):
        spatial_smoothing(np.zeros((2, 2, 2)), subarraySize=2)
    with pytest.raises(ValueError, match="subarraySize"):
        spatial_smoothing(signal, subarraySize=1)

    positions = np.vstack((ula_radar.virtualArray, ula_radar.virtualArray[0]))
    duplicated = np.concatenate((signal, [2j * signal[0]]))
    first, _ = combine_duplicate_channels(duplicated, positions, policy="first")
    noncoherent, _ = combine_duplicate_channels(
        duplicated, positions, policy="noncoherent"
    )
    assert first.shape == noncoherent.shape == signal.shape
    with pytest.raises(ValueError, match="count"):
        combine_duplicate_channels(signal[:-1], ula_radar.virtualArray)
    with pytest.raises(ValueError, match="Unknown"):
        combine_duplicate_channels(duplicated, positions, policy="bad")
