from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from pyradar.base import (
    DDM,
    FMCW,
    SIMO,
    ADCFrame,
    Calibration,
    CFARConfig,
    ClusteringConfig,
    ClusterSet,
    DetectionSet,
    DoAConfig,
    FFTConfig,
    PointCloud,
    PointCloudConfig,
    Radar,
    RadarCube,
    Sampler,
    TrackingConfig,
    TrackSet,
    Transceivers,
)


def test_waveform_and_sampler_validate_physical_values() -> None:
    waveform = FMCW(77e9, 20e12, 2e-6, 40e-6, 5e-6)
    assert waveform.chirpInterval == pytest.approx(45e-6)
    assert waveform.rampBandwidth == pytest.approx(800e6)
    assert waveform.wavelength > 0.0
    for arguments in (
        (np.nan, 20e12, 2e-6, 40e-6),
        (0.0, 20e12, 2e-6, 40e-6),
        (77e9, 0.0, 2e-6, 40e-6),
        (77e9, 20e12, -1e-6, 40e-6),
        (77e9, 20e12, 40e-6, 40e-6),
    ):
        with pytest.raises(ValueError):
            FMCW(*arguments)

    sampler = Sampler(16, 4, 2e6, framePeriod=0.1)
    assert sampler.samplePeriod == pytest.approx(0.5e-6)
    assert sampler.captureDuration == pytest.approx(8e-6)
    for kwargs in (
        {"numSamples": 0, "numLoops": 1, "sampleRate": 1e6},
        {"numSamples": 1, "numLoops": 0, "sampleRate": 1e6},
        {"numSamples": 1, "numLoops": 1, "sampleRate": np.inf},
        {"numSamples": 1, "numLoops": 1, "sampleRate": 1e6, "bitDepth": 0},
        {"numSamples": 1, "numLoops": 1, "sampleRate": 1e6, "framePeriod": 0},
    ):
        with pytest.raises(ValueError):
            Sampler(**kwargs)


def test_transceiver_geometry_validation_and_groups() -> None:
    geometry = Transceivers(
        txPositions=np.asarray([[0.0, 0.0]]),
        rxPositions=np.asarray([[0.0, 0.0], [0.5, 0.0]]),
    )
    assert geometry.txPositions.shape == (1, 3)
    virtual = geometry.virtual_positions(((0, 0), (0, 1)))
    assert virtual.shape == (2, 3)
    with pytest.raises(ValueError, match="empty"):
        geometry.virtual_positions(())
    with pytest.raises(ValueError, match="Invalid channel"):
        geometry.virtual_positions(((1, 0),))
    with pytest.raises(ValueError, match="tolerance"):
        geometry.phase_center_groups(virtual, tolerance=0)
    groups = geometry.phase_center_groups(np.vstack((virtual, virtual[0])))
    assert groups[0] == (0, 2)
    with pytest.raises(ValueError):
        Transceivers(np.empty((0, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        Transceivers(np.zeros((1, 4)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        Transceivers(np.asarray([[0.0, np.nan, 0.0]]), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        Transceivers(np.zeros((1, 3)), np.zeros((1, 3)), (-1,))


def test_processing_configs_reject_invalid_values() -> None:
    assert FFTConfig(rangeCrop=(2, -1)).crop("range") == slice(2, -1)
    assert FFTConfig().crop("range") == slice(None)
    with pytest.raises(ValueError):
        FFTConfig(rangeFftSize=0)
    for kwargs in (
        {"trainingCells": (-1, 2)},
        {"pfa": 1.0},
        {"rankFraction": 0.0},
        {"maxDetections": 0},
    ):
        with pytest.raises(ValueError):
            CFARConfig(**kwargs)
    for kwargs in (
        {"azimuthBins": 1},
        {"azimuthFov": (0.5, -0.5)},
        {"elevationFov": (-2.0, 0.0)},
        {"diagonalLoading": -1.0},
    ):
        with pytest.raises(ValueError):
            DoAConfig(**kwargs)
    with pytest.raises(ValueError):
        PointCloudConfig(minRange=-1)
    with pytest.raises(ValueError):
        PointCloudConfig(minRange=2, maxRange=1)
    for kwargs in (
        {"eps": 0},
        {"minSamples": 0},
        {"velocityScale": -1},
        {"dimensions": 1},
    ):
        with pytest.raises(ValueError):
            ClusteringConfig(**kwargs)
    for kwargs in (
        {"dimensions": 4},
        {"processNoise": -1},
        {"measurementNoise": 0},
        {"radialVelocityNoise": 0},
        {"gatingThreshold": 0},
        {"confirmationHits": 0},
        {"deletionMisses": 0},
    ):
        with pytest.raises(ValueError):
            TrackingConfig(**kwargs)


def test_radar_validation_config_formats_and_real_sampling(tmp_path, ula_radar) -> None:
    with pytest.raises(ValueError, match="numTx"):
        replace(ula_radar, mimo=SIMO(numRx=4))
    with pytest.raises(ValueError, match="numRx"):
        replace(
            ula_radar,
            transceivers=Transceivers(
                ula_radar.transceivers.txPositions,
                ula_radar.transceivers.rxPositions[:3],
            ),
        )
    with pytest.raises(ValueError, match="extends"):
        replace(ula_radar, sampler=Sampler(1000, 16, 1e6))
    with pytest.raises(ValueError, match="name"):
        replace(ula_radar, name="")
    real = replace(
        ula_radar,
        sampler=replace(ula_radar.sampler, complexSampling=False),
    )
    assert real.maxUnambiguousRange == pytest.approx(ula_radar.maxUnambiguousRange / 2)

    config = {
        "waveform": {
            "startFrequency": 77e9,
            "slope": 1e12,
            "adcStartTime": 0,
            "rampEndTime": 10e-6,
        },
        "sampler": {"numSamples": 4, "numLoops": 4, "sampleRate": 1e6},
        "transceivers": {
            "txPositions": [[0, 0, 0]],
            "rxPositions": [[0, 0, 0]],
        },
        "mimo": {"type": "simo"},
    }
    jsonPath = tmp_path / "radar.json"
    jsonPath.write_text(json.dumps(config), encoding="utf-8")
    assert Radar.from_config(jsonPath).name == "radar"
    with pytest.raises(FileNotFoundError):
        Radar.from_config(tmp_path / "missing.yaml")
    textPath = tmp_path / "radar.txt"
    textPath.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML or JSON"):
        Radar.from_config(textPath)
    config["mimo"] = {"type": "unknown"}
    jsonPath.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid radar config"):
        Radar.from_config(jsonPath)


def test_ddm_derived_slow_time_and_sparse_axes(ula_radar) -> None:
    ddm = DDM(2, 4, (0.0, 0.5), 2)
    radar = replace(
        ula_radar,
        mimo=ddm,
        sampler=replace(ula_radar.sampler, numLoops=16),
    )
    assert radar.numSlowTimeSamples == 8
    assert radar.slowTimeInterval == pytest.approx(2 * radar.waveform.chirpInterval)
    assert radar.velocityAxis.shape == (16,)


def test_calibration_all_domains_and_shape_errors() -> None:
    with pytest.raises(ValueError, match="finite"):
        Calibration(adcGain=np.asarray([np.nan]))
    with pytest.raises(ValueError, match="overlapPairs"):
        Calibration(overlapPairs=np.asarray([1, 2]))
    adc = np.ones((2, 1, 2, 4), dtype=complex)
    calibration = Calibration(
        adcGain=np.asarray([2.0, 3.0]),
        adcPhase=np.asarray([0.0, np.pi]),
        frequencySlope=np.asarray([0.1, -0.2]),
    )
    calibrated = calibration.apply_adc(adc)
    assert calibrated.shape == adc.shape
    assert not np.allclose(calibrated, adc)
    with pytest.raises(ValueError, match="expects"):
        calibration.apply_adc(np.ones((2, 2)))
    with pytest.raises(ValueError, match="frequencySlope"):
        Calibration(frequencySlope=np.ones(3)).apply_adc(adc)
    rangeData = np.ones((2, 2, 4), dtype=complex)
    coupling = np.full((2, 4), 0.25)
    np.testing.assert_allclose(
        Calibration(rangeCoupling=coupling).apply_range(rangeData), 0.75
    )
    with pytest.raises(ValueError, match="expects"):
        Calibration().apply_range(np.ones((2, 2)))
    np.testing.assert_allclose(Calibration().apply_range(rangeData), rangeData)
    np.testing.assert_allclose(Calibration().apply_array(rangeData), rangeData)
    phased = Calibration(arrayPhase=np.asarray([1.0, -1.0])).apply_array(
        rangeData, channelAxis=1
    )
    np.testing.assert_allclose(phased[:, 1], -rangeData[:, 1])
    with pytest.raises(ValueError, match="arrayPhase"):
        Calibration(arrayPhase=np.ones(3)).apply_array(rangeData, channelAxis=1)


def test_named_data_contract_validation_and_helpers(ula_radar) -> None:
    with pytest.raises(ValueError, match="dims length"):
        ADCFrame(np.zeros((2, 2)), ("loop",), radar=ula_radar)
    with pytest.raises(ValueError, match="unique"):
        ADCFrame(np.zeros((2, 2)), ("loop", "loop"), radar=ula_radar)
    with pytest.raises(ValueError, match="Unknown"):
        ADCFrame(np.zeros(2), ("unknown",), radar=ula_radar)
    with pytest.raises(ValueError, match="Radar is required"):
        ADCFrame(
            np.zeros((1, 1, 1, 1)), ("loop", "emission", "rx", "sample")
        ).canonical()
    with pytest.raises(ValueError, match="missing"):
        ADCFrame(
            np.zeros((2, 4, 64)),
            ("emission", "rx", "sample"),
            radar=ula_radar,
        ).canonical()

    simo = Radar(
        FMCW(77e9, 1e12, 0.0, 10e-6),
        Sampler(8, 1, 1e6),
        Transceivers(np.asarray([[0, 0, 0]]), np.asarray([[0, 0, 0]])),
        SIMO(1),
    )
    singleton = ADCFrame(np.zeros((1, 8)), ("rx", "sample"), radar=simo).canonical()
    assert singleton.data.shape == (1, 1, 1, 8)

    with pytest.raises(ValueError, match="dims"):
        RadarCube(np.zeros((2, 2)), ("range", "range"), "bad")
    with pytest.raises(ValueError, match="not a cube"):
        RadarCube(
            np.zeros((2, 2)),
            ("range", "doppler"),
            "bad",
            coords={"angle": np.arange(2)},
        )
    with pytest.raises(ValueError, match="wrong length"):
        RadarCube(
            np.zeros((2, 2)),
            ("range", "doppler"),
            "bad",
            coords={"range": np.arange(3)},
        )
    cube = RadarCube(
        np.asarray([[1 + 1j, 2], [3, 4]]),
        ("range", "doppler"),
        "rd",
        coords={"range": np.arange(2)},
        metadata={"a": 1},
    )
    assert cube.dim_index("doppler") == 1
    with pytest.raises(ValueError, match="not present"):
        cube.dim_index("azimuth")
    np.testing.assert_allclose(cube.magnitude, np.abs(cube.data))
    np.testing.assert_allclose(cube.power, np.abs(cube.data) ** 2)
    updated = cube.with_data(np.ones((2, 2)), stage="new", metadata={"b": 2})
    assert updated.stage == "new" and updated.metadata == {"a": 1, "b": 2}


def test_result_container_validation() -> None:
    detections = DetectionSet(
        np.asarray([1]),
        np.asarray([2]),
        np.asarray([3.0]),
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([4.8]),
        azimuthBin=np.asarray([3]),
        elevationBin=np.asarray([4]),
    )
    assert len(detections) == 1
    assert len(DetectionSet.empty()) == 0
    with pytest.raises(ValueError, match="power"):
        DetectionSet(
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )

    with pytest.raises(ValueError, match="xyz"):
        PointCloud(
            np.ones((2, 2)),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.zeros((2, 4), dtype=int),
        )
    with pytest.raises(ValueError, match="sourceBins"):
        PointCloud(
            np.ones((2, 3)),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.zeros((2, 3), dtype=int),
        )
    cloud = PointCloud(
        np.asarray([[1.0, 1.0, 1.0]]),
        np.ones(1),
        np.ones(1),
        np.ones(1),
        np.zeros((1, 4), dtype=int),
    )
    assert cloud.elevation[0] == pytest.approx(np.arctan2(1.0, np.sqrt(2.0)))
    assert PointCloud.empty(timestamp=1.0, frameId="x").frameId == "x"

    clusters = ClusterSet(
        labels=np.asarray([0]),
        clusterId=np.asarray([0]),
        centroid=np.zeros((1, 3)),
        size=np.ones((1, 3)),
        covariance=np.zeros((1, 3, 3)),
        radialVelocity=np.zeros(1),
        power=np.ones(1),
    )
    assert len(clusters) == 1
    with pytest.raises(ValueError, match="centroid"):
        ClusterSet(
            np.asarray([0]),
            np.asarray([0]),
            np.zeros((1, 4)),
            np.ones((1, 4)),
            np.zeros((1, 4, 4)),
            np.zeros(1),
            np.ones(1),
        )

    tracks = TrackSet(
        trackId=np.asarray([1]),
        status=("confirmed",),
        position=np.zeros((1, 2)),
        velocity=np.zeros((1, 2)),
        covariance=np.eye(4)[None],
        age=np.ones(1, dtype=int),
        hits=np.ones(1, dtype=int),
        misses=np.zeros(1, dtype=int),
    )
    assert len(tracks) == 1
    with pytest.raises(ValueError, match="status"):
        TrackSet(
            np.asarray([1]),
            ("unknown",),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.eye(4)[None],
            np.ones(1, dtype=int),
            np.ones(1, dtype=int),
            np.zeros(1, dtype=int),
        )
