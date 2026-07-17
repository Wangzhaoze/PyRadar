"""Deterministic velocity-aware DBSCAN for radar point clouds."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from pyradar.base.cube import ClusterSet, PointCloud


def _labels(
    features: NDArray[np.float64], eps: float, minSamples: int
) -> NDArray[np.int64]:
    tree = cKDTree(features)
    neighborhoods = [sorted(tree.query_ball_point(point, eps)) for point in features]
    unvisited = -2
    noise = -1
    labels = np.full(features.shape[0], unvisited, dtype=np.int64)
    clusterId = 0
    for pointId in range(features.shape[0]):
        if labels[pointId] != unvisited:
            continue
        neighbors = neighborhoods[pointId]
        if len(neighbors) < minSamples:
            labels[pointId] = noise
            continue
        labels[pointId] = clusterId
        queue = list(neighbors)
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            neighborId = queue[cursor]
            cursor += 1
            if labels[neighborId] == noise:
                labels[neighborId] = clusterId
            if labels[neighborId] != unvisited:
                continue
            labels[neighborId] = clusterId
            expansion = neighborhoods[neighborId]
            if len(expansion) >= minSamples:
                for candidate in expansion:
                    if candidate not in queued:
                        queued.add(candidate)
                        queue.append(candidate)
        clusterId += 1
    return labels


def dbscan(
    pointCloud: PointCloud,
    *,
    eps: float = 1.0,
    minSamples: int = 3,
    velocityScale: float = 1.0,
    dimensions: int = 3,
) -> ClusterSet:
    """Cluster a radar point cloud in position/radial-velocity space.

    The feature metric is ``[position, velocityScale * radialVelocity]``.
    Iteration and neighborhood ordering are fixed, making labels deterministic.
    """

    if eps <= 0.0 or minSamples < 1 or dimensions not in (2, 3):
        raise ValueError("Invalid DBSCAN eps, minSamples, or dimensions.")
    count = len(pointCloud)
    if count == 0:
        return ClusterSet(
            labels=np.empty(0, dtype=np.int64),
            clusterId=np.empty(0, dtype=np.int64),
            centroid=np.empty((0, dimensions)),
            size=np.empty((0, dimensions)),
            covariance=np.empty((0, dimensions, dimensions)),
            radialVelocity=np.empty(0),
            power=np.empty(0),
        )
    positions = pointCloud.xyz[:, :dimensions]
    features = np.column_stack((positions, velocityScale * pointCloud.radialVelocity))
    labels = _labels(features, eps, minSamples)
    ids = np.unique(labels[labels >= 0])
    centroids = np.empty((ids.size, dimensions), dtype=float)
    sizes = np.empty_like(centroids)
    covariances = np.empty((ids.size, dimensions, dimensions), dtype=float)
    velocities = np.empty(ids.size, dtype=float)
    powers = np.empty(ids.size, dtype=float)
    for outputId, clusterId in enumerate(ids):
        mask = labels == clusterId
        members = positions[mask]
        weights = np.maximum(pointCloud.power[mask], np.finfo(float).tiny)
        centroids[outputId] = np.average(members, axis=0, weights=weights)
        sizes[outputId] = np.ptp(members, axis=0)
        if members.shape[0] > 1:
            covariances[outputId] = np.cov(members, rowvar=False, ddof=1)
        else:
            covariances[outputId] = np.zeros((dimensions, dimensions))
        velocities[outputId] = np.average(
            pointCloud.radialVelocity[mask], weights=weights
        )
        powers[outputId] = np.sum(pointCloud.power[mask])
    return ClusterSet(
        labels=labels,
        clusterId=ids,
        centroid=centroids,
        size=sizes,
        covariance=covariances,
        radialVelocity=velocities,
        power=powers,
    )


__all__ = ["dbscan"]
