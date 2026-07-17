"""Coordinate, conversion, visualization, and local I/O helpers."""

from .converter import (
    beat_frequency_to_range,
    db_to_power,
    doppler_frequency_to_velocity,
    magnitude_to_db,
    max_unambiguous_range,
    max_unambiguous_velocity,
    power_to_db,
    range_axis,
    range_resolution,
    ula_angle_axis,
    ula_unambiguous_fov,
    velocity_axis,
    velocity_resolution,
)
from .geometry import (
    cartesian_to_spherical,
    compose_transforms,
    inverse_transform,
    make_transform,
    spherical_to_cartesian,
    transform_points,
)
from .pointcloud import (
    ICPResult,
    axis_aligned_bounds,
    icp_register,
    random_sample,
    voxel_downsample,
)

__all__ = [
    "ICPResult",
    "axis_aligned_bounds",
    "beat_frequency_to_range",
    "cartesian_to_spherical",
    "compose_transforms",
    "db_to_power",
    "doppler_frequency_to_velocity",
    "icp_register",
    "inverse_transform",
    "magnitude_to_db",
    "make_transform",
    "max_unambiguous_range",
    "max_unambiguous_velocity",
    "power_to_db",
    "random_sample",
    "range_axis",
    "range_resolution",
    "spherical_to_cartesian",
    "transform_points",
    "ula_angle_axis",
    "ula_unambiguous_fov",
    "velocity_axis",
    "velocity_resolution",
    "voxel_downsample",
]
