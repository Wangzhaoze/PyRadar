from __future__ import annotations

import numpy as np

from pyradar.base import PointCloud, TrackingConfig
from pyradar.rsp import MultiTargetTracker, dbscan


def _cloud(points, velocities=None) -> PointCloud:
    xyz = np.asarray(points, dtype=float)
    count = xyz.shape[0]
    radial = (
        np.zeros(count) if velocities is None else np.asarray(velocities, dtype=float)
    )
    return PointCloud(
        xyz=xyz,
        power=np.ones(count),
        snr=np.full(count, 20.0),
        radialVelocity=radial,
        sourceBins=np.zeros((count, 4), dtype=int),
    )


def test_velocity_aware_dbscan_is_deterministic() -> None:
    points = [
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [5, 5, 0],
        [5.1, 5, 0],
        [5, 5.1, 0],
        [20, 0, 0],
    ]
    cloud = _cloud(points, [0, 0.1, 0, 2, 2.1, 2, 0])
    first = dbscan(cloud, eps=0.4, minSamples=3, velocityScale=0.2)
    second = dbscan(cloud, eps=0.4, minSamples=3, velocityScale=0.2)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert len(first) == 2
    assert first.labels[-1] == -1
    assert first.centroid.shape == (2, 3)


def test_tracker_birth_confirmation_coast_and_delete() -> None:
    config = TrackingConfig(
        enabled=True,
        dimensions=2,
        confirmationHits=2,
        deletionMisses=2,
        gatingThreshold=20.0,
    )
    tracker = MultiTargetTracker(config)
    first = tracker.update(_cloud([[1, 0, 0]], [1]), deltaTime=1.0)
    assert first.status == ("tentative",)
    second = tracker.update(_cloud([[2, 0, 0]], [1]), deltaTime=1.0)
    assert second.status == ("confirmed",)
    identity = second.trackId[0]
    coast = tracker.update(PointCloud.empty(), deltaTime=1.0)
    assert coast.status == ("coasting",)
    deleted = tracker.update(PointCloud.empty(), deltaTime=1.0)
    assert deleted.status == ("deleted",)
    assert deleted.trackId[0] == identity
    assert len(tracker.update(PointCloud.empty(), deltaTime=1.0)) == 0


def test_tracker_keeps_ids_through_crossing_with_variable_dt() -> None:
    tracker = MultiTargetTracker(
        TrackingConfig(
            enabled=True,
            dimensions=2,
            confirmationHits=1,
            deletionMisses=3,
            gatingThreshold=30.0,
            measurementNoise=0.2,
        )
    )
    initial = tracker.update(
        _cloud([[-2, -0.5, 0], [2, 0.5, 0]], [1, -1]), deltaTime=0.5
    )
    ids = initial.trackId.copy()
    tracker.update(_cloud([[-1, -0.5, 0], [1, 0.5, 0]], [1, -1]), deltaTime=1.0)
    crossed = tracker.update(
        _cloud([[0.2, 0.5, 0], [-0.2, -0.5, 0]], [-1, 1]), deltaTime=0.8
    )
    np.testing.assert_array_equal(np.sort(crossed.trackId), np.sort(ids))
