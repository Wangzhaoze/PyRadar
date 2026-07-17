"""Conversion of range/Doppler/DoA detections to FLU point clouds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base.cube import DetectionSet, PointCloud

if TYPE_CHECKING:
    from pyradar.base.radar import Radar


def spherical_to_cartesian(
    range: ArrayLike,
    azimuth: ArrayLike,
    elevation: ArrayLike,
) -> NDArray[np.float64]:
    """Convert radar spherical coordinates to right-handed FLU Cartesian."""

    radius, azimuthArray, elevationArray = np.broadcast_arrays(
        np.asarray(range, dtype=float),
        np.asarray(azimuth, dtype=float),
        np.asarray(elevation, dtype=float),
    )
    horizontal = radius * np.cos(elevationArray)
    return np.column_stack(
        (
            horizontal * np.cos(azimuthArray),
            horizontal * np.sin(azimuthArray),
            radius * np.sin(elevationArray),
        )
    )


def detections_to_pointcloud(
    detections: DetectionSet,
    *,
    radar: Radar,
    azimuth: ArrayLike,
    elevation: ArrayLike,
    azimuthBin: ArrayLike | None = None,
    elevationBin: ArrayLike | None = None,
    radialVelocity: ArrayLike | None = None,
    timestamp: float | None = None,
    frameId: int | str | None = None,
) -> PointCloud:
    """Map typed detections and DoA estimates to a typed point cloud."""

    count = len(detections)
    if count == 0:
        return PointCloud.empty(timestamp=timestamp, frameId=frameId)
    azimuthArray = np.asarray(azimuth, dtype=float)
    elevationArray = np.asarray(elevation, dtype=float)
    if azimuthArray.shape != (count,) or elevationArray.shape != (count,):
        raise ValueError("azimuth and elevation must have one value per detection.")
    ranges = detections.rangeBin.astype(float) * radar.rangeBinSize
    xyz = spherical_to_cartesian(ranges, azimuthArray, elevationArray)
    azBins = (
        np.full(count, -1, dtype=np.int64)
        if azimuthBin is None
        else np.asarray(azimuthBin, dtype=np.int64)
    )
    elBins = (
        np.full(count, -1, dtype=np.int64)
        if elevationBin is None
        else np.asarray(elevationBin, dtype=np.int64)
    )
    if azBins.shape != (count,) or elBins.shape != (count,):
        raise ValueError("Angle bin arrays must have one value per detection.")
    sourceBins = np.column_stack(
        (detections.rangeBin, detections.dopplerBin, azBins, elBins)
    )
    velocities = (
        radar.velocityAxis[detections.dopplerBin]
        if radialVelocity is None
        else np.asarray(radialVelocity, dtype=float)
    )
    if velocities.shape != (count,):
        raise ValueError("radialVelocity must have one value per detection.")
    return PointCloud(
        xyz=xyz,
        power=detections.power,
        snr=detections.snr,
        radialVelocity=velocities,
        sourceBins=sourceBins,
        timestamp=timestamp,
        frameId=frameId,
    )


__all__ = ["detections_to_pointcloud", "spherical_to_cartesian"]
