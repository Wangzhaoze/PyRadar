"""Model-aware radar simulation.

The core point-target simulator depends only on NumPy. Mesh ray tracing is
loaded lazily and requires the ``simulation`` optional dependency.
"""

from .adc import quantize_adc, simulate_adc, simulate_paths, targets_to_paths
from .raytracing import (
    RayPath,
    RayTraceResult,
    TrimeshRayTracer,
    fresnel_schlick,
    reflect,
    refract,
    sample_cone_directions,
)
from .scene import (
    RangeImage,
    linear_trajectory,
    render_range_image,
    sample_plane,
    sample_sphere,
)
from .targets import PointTarget, PropagationPath, TargetScene

__all__ = [
    "PointTarget",
    "PropagationPath",
    "RangeImage",
    "RayPath",
    "RayTraceResult",
    "TargetScene",
    "TrimeshRayTracer",
    "fresnel_schlick",
    "linear_trajectory",
    "quantize_adc",
    "reflect",
    "refract",
    "render_range_image",
    "sample_cone_directions",
    "sample_plane",
    "sample_sphere",
    "simulate_adc",
    "simulate_paths",
    "targets_to_paths",
]
