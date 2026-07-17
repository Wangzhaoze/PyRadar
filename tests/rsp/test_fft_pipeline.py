from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyradar.base import (
    ADCFrame,
    ClusteringConfig,
    DetectionSet,
    FrameResult,
    PointCloud,
    RadarCube,
    TrackingConfig,
)
from pyradar.rsp import (
    angle_fft,
    detections_to_pointcloud,
    doppler_fft,
    range_doppler_azimuth_fft,
    range_doppler_fft,
    range_fft,
    window,
)
from tests.conftest import synthesize_tdm_target


def test_parameter_range_fft_recovers_bin() -> None:
    size = 64
    target = 11
    samples = np.exp(2j * np.pi * target * np.arange(size) / size)
    transformed = range_fft(
        samples,
        fftSize=size,
        sampleAxis=0,
        window="rectangular",
    )
    assert isinstance(transformed, np.ndarray)
    assert int(np.argmax(np.abs(transformed))) == target


def test_model_range_doppler_recovers_target_and_compensates_tdm(ula_radar) -> None:
    rangeBin = 12
    signedDoppler = 3
    azimuth = np.deg2rad(20.0)
    adc = synthesize_tdm_target(
        ula_radar,
        rangeBin=rangeBin,
        dopplerBin=signedDoppler,
        azimuth=azimuth,
    )
    cube = range_doppler_fft(
        adc,
        radar=ula_radar,
        dims=("loop", "emission", "rx", "sample"),
    )
    assert isinstance(cube, RadarCube)
    power = np.mean(np.abs(cube.data) ** 2, axis=2)
    recovered = np.unravel_index(np.argmax(power), power.shape)
    assert recovered == (rangeBin, signedDoppler + ula_radar.sampler.numLoops // 2)

    from pyradar.rsp.doa import steering_vector

    measured = cube.data[recovered]
    measured /= measured[0]
    expected = steering_vector(ula_radar.virtualArray, ula_radar.wavelength, azimuth)
    expected /= expected[0]
    np.testing.assert_allclose(measured, expected, atol=1e-6)


def test_full_pipeline_recovers_one_bin_and_returns_typed_pointcloud(ula_radar) -> None:
    rangeBin = 15
    signedDoppler = -2
    angle = np.deg2rad(-25.0)
    adc = synthesize_tdm_target(
        ula_radar,
        rangeBin=rangeBin,
        dopplerBin=signedDoppler,
        azimuth=angle,
        amplitude=100.0,
    )
    result = ula_radar.build_pipeline().process(
        adc, dims=("loop", "emission", "rx", "sample")
    )
    assert isinstance(result, FrameResult)
    assert isinstance(result.pointCloud, PointCloud)
    assert len(result.pointCloud) == 1
    assert result.pointCloud.sourceBins[0, 0] == rangeBin
    assert result.pointCloud.sourceBins[0, 1] == signedDoppler + 8
    assert abs(result.pointCloud.azimuth[0] - angle) <= np.pi / 64


def test_windows_and_parameter_fft_compositions(ula_radar) -> None:
    for name in (
        "rectangular",
        "rect",
        "boxcar",
        "none",
        "hann",
        "hamming",
        "blackman",
        "blackmanharris",
    ):
        assert window(8, name).shape == (8,)
    with pytest.raises(ValueError, match="positive"):
        window(0)
    with pytest.raises(ValueError, match="Unsupported"):
        window(8, "triangle")
    with pytest.raises(ValueError, match="fftSize"):
        range_fft(np.ones(8), fftSize=0)
    np.testing.assert_allclose(
        range_fft(
            np.ones(8),
            fftSize=8,
            sampleAxis=0,
            window="rectangular",
            removeMean=True,
        ),
        0.0,
    )

    loop = np.arange(16)
    tone = np.exp(2j * np.pi * 3 * loop / 16)[:, None]
    doppler = doppler_fft(
        tone,
        fftSize=16,
        loopAxis=0,
        window="rectangular",
        removeMean=False,
    )
    assert np.argmax(np.abs(doppler[:, 0])) == 11
    angle = angle_fft(
        np.arange(4, dtype=float),
        fftSize=8,
        antennaAxis=0,
        window="rectangular",
    )
    assert angle.shape == (8,)
    with pytest.raises(ValueError, match="dimension"):
        angle_fft(np.ones(4), dimension="roll")

    raw = np.ones((8, 16), dtype=complex)
    rd = range_doppler_fft(
        raw,
        rangeFftSize=16,
        dopplerFftSize=8,
        sampleAxis=1,
        loopAxis=0,
        rangeWindow="rectangular",
        dopplerWindow="rectangular",
        removeDopplerMean=False,
    )
    assert isinstance(rd, np.ndarray) and rd.shape == (8, 16)

    adc = synthesize_tdm_target(ula_radar, rangeBin=12, dopplerBin=2, azimuth=0.2)
    rda = range_doppler_azimuth_fft(
        adc,
        radar=ula_radar,
        dims=("loop", "emission", "rx", "sample"),
        rangeWindow="rectangular",
        dopplerWindow="rectangular",
        removeDopplerMean=False,
    )
    assert isinstance(rda, RadarCube)
    assert rda.dims == ("range", "doppler", "azimuth")


def test_angle_fft_preserves_named_cube_coordinates(ula_radar) -> None:
    cube = RadarCube(
        data=np.ones((3, 8), dtype=complex),
        dims=("range", "virtual"),
        stage="range",
        radar=ula_radar,
        coords={"range": np.arange(3) * ula_radar.rangeBinSize},
    )
    transformed = angle_fft(cube, radar=ula_radar, dimension="azimuth")
    assert isinstance(transformed, RadarCube)
    assert transformed.dims == ("range", "azimuth")
    assert transformed.coords["range"].shape == (3,)
    assert transformed.coords["azimuth"].shape == (128,)


def test_pipeline_dense_products_clustering_tracking_and_empty_frame(ula_radar) -> None:
    processing = replace(
        ula_radar.processing,
        clustering=ClusteringConfig(
            enabled=True,
            eps=1.0,
            minSamples=1,
            velocityScale=0.2,
            dimensions=3,
        ),
        tracking=TrackingConfig(
            enabled=True,
            dimensions=3,
            confirmationHits=1,
            deletionMisses=2,
        ),
        retainRangeDoppler=False,
        retainRangeAngle=True,
        retainRangeDopplerAngle=True,
    )
    radar = replace(ula_radar, processing=processing)
    adc = synthesize_tdm_target(
        radar,
        rangeBin=15,
        dopplerBin=2,
        azimuth=-0.2,
        amplitude=100.0,
    )
    frame = ADCFrame(
        adc,
        ("loop", "emission", "rx", "sample"),
        radar=radar,
        timestamp=1.0,
        frameId=4,
    )
    pipeline = radar.build_pipeline()
    result = pipeline.process(frame, timestamp=1.1, frameId=5)
    assert result.adcFrame.timestamp == 1.1 and result.adcFrame.frameId == 5
    assert result.rangeDopplerCube is None
    assert result.rangeAngleCube.shape == (64, 181)
    assert result.rangeDopplerAngleCube.shape == (64, 16, 181)
    assert len(result.clusters) == 1
    assert result.tracks.status == ("confirmed",)
    pipeline.reset()

    empty = ula_radar.process_adc(
        np.zeros((16, 2, 4, 64), dtype=complex),
        dims=("loop", "emission", "rx", "sample"),
    )
    assert len(empty.detections) == len(empty.pointCloud) == 0


def test_detection_to_pointcloud_validation_and_velocity_override(ula_radar) -> None:
    empty = detections_to_pointcloud(
        DetectionSet.empty(), radar=ula_radar, azimuth=[], elevation=[]
    )
    assert len(empty) == 0
    detections = DetectionSet(
        rangeBin=np.asarray([10]),
        dopplerBin=np.asarray([8]),
        power=np.asarray([5.0]),
        noise=np.asarray([1.0]),
        threshold=np.asarray([2.0]),
        snr=np.asarray([7.0]),
    )
    cloud = detections_to_pointcloud(
        detections,
        radar=ula_radar,
        azimuth=np.asarray([0.1]),
        elevation=np.asarray([0.2]),
        radialVelocity=np.asarray([12.0]),
    )
    assert cloud.radialVelocity[0] == 12.0
    with pytest.raises(ValueError, match="azimuth"):
        detections_to_pointcloud(detections, radar=ula_radar, azimuth=[], elevation=[])
    with pytest.raises(ValueError, match="Angle bin"):
        detections_to_pointcloud(
            detections,
            radar=ula_radar,
            azimuth=np.asarray([0.0]),
            elevation=np.asarray([0.0]),
            azimuthBin=np.asarray([1, 2]),
        )
    with pytest.raises(ValueError, match="radialVelocity"):
        detections_to_pointcloud(
            detections,
            radar=ula_radar,
            azimuth=np.asarray([0.0]),
            elevation=np.asarray([0.0]),
            radialVelocity=np.asarray([1.0, 2.0]),
        )
