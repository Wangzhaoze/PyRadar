"""Small, optional Matplotlib views for frame products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyradar.base import FrameResult, PointCloud, RadarCube

from .converter import power_to_db


def _axes(ax: Any, *, projection: str | None = None):
    if ax is not None:
        return ax.figure, ax
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(8, 5), constrained_layout=True)
    return figure, figure.add_subplot(111, projection=projection)


def plot_range_doppler(cube: RadarCube, *, ax: Any = None):
    """Plot channel-integrated range-Doppler power."""

    if cube.dims != ("range", "doppler", "virtual"):
        raise ValueError("Expected a range/doppler/virtual RadarCube.")
    figure, axis = _axes(ax)
    power = np.mean(np.abs(cube.data) ** 2, axis=2)
    ranges = np.asarray(cube.coords["range"])
    velocities = np.asarray(cube.coords["doppler"])
    image = axis.imshow(
        power_to_db(power).T,
        origin="lower",
        aspect="auto",
        extent=(ranges[0], ranges[-1], velocities[0], velocities[-1]),
        cmap="turbo",
    )
    axis.set(xlabel="Range (m)", ylabel="Radial velocity (m/s)", title="Range-Doppler")
    figure.colorbar(image, ax=axis, label="Power (dB)")
    return axis


def plot_range_angle(cube: RadarCube, *, ax: Any = None):
    """Plot a range-azimuth power map."""

    if cube.dims != ("range", "azimuth"):
        raise ValueError("Expected a range/azimuth RadarCube.")
    figure, axis = _axes(ax)
    ranges = np.asarray(cube.coords["range"])
    azimuth = np.rad2deg(np.asarray(cube.coords["azimuth"]))
    image = axis.imshow(
        power_to_db(cube.data).T,
        origin="lower",
        aspect="auto",
        extent=(ranges[0], ranges[-1], azimuth[0], azimuth[-1]),
        cmap="turbo",
    )
    axis.set(xlabel="Range (m)", ylabel="Azimuth (deg)", title="Range-Azimuth")
    figure.colorbar(image, ax=axis, label="Power (dB)")
    return axis


def plot_pointcloud(pointCloud: PointCloud, *, ax: Any = None):
    """Plot a 3D FLU point cloud colored by radial velocity."""

    figure, axis = _axes(ax, projection="3d")
    if len(pointCloud):
        image = axis.scatter(
            pointCloud.xyz[:, 0],
            pointCloud.xyz[:, 1],
            pointCloud.xyz[:, 2],
            c=pointCloud.radialVelocity,
            s=np.clip(8.0 + pointCloud.snr, 8.0, 60.0),
            cmap="coolwarm",
            linewidths=0,
        )
        figure.colorbar(image, ax=axis, shrink=0.65, label="Radial velocity (m/s)")
    axis.set(
        xlabel="x forward (m)",
        ylabel="y left (m)",
        zlabel="z up (m)",
        title="Radar point cloud",
    )
    return axis


def save_frame_figures(
    result: FrameResult,
    outputDir: str | Path,
    *,
    prefix: str = "",
    dpi: int = 160,
) -> dict[str, Path]:
    """Save available RD, RA, and point-cloud views for one result."""

    from matplotlib import pyplot as plt

    root = Path(outputDir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if result.rangeDopplerCube is not None:
        axis = plot_range_doppler(result.rangeDopplerCube)
        outputs["rangeDoppler"] = root / f"{prefix}range_doppler.png"
        axis.figure.savefig(outputs["rangeDoppler"], dpi=dpi)
        plt.close(axis.figure)
    if result.rangeAngleCube is not None:
        axis = plot_range_angle(result.rangeAngleCube)
        outputs["rangeAngle"] = root / f"{prefix}range_azimuth.png"
        axis.figure.savefig(outputs["rangeAngle"], dpi=dpi)
        plt.close(axis.figure)
    axis = plot_pointcloud(result.pointCloud)
    outputs["pointCloud"] = root / f"{prefix}point_cloud_3d.png"
    axis.figure.savefig(outputs["pointCloud"], dpi=dpi)
    plt.close(axis.figure)
    return outputs


__all__ = [
    "plot_pointcloud",
    "plot_range_angle",
    "plot_range_doppler",
    "save_frame_figures",
]
