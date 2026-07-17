"""Optional mesh ray tracing and shooting-and-bouncing-rays helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .targets import PropagationPath


def _directions(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("directions must have shape (N, 3) with finite values.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("directions cannot contain zero vectors.")
    return array / norms[:, None]


def sample_cone_directions(
    numRays: int,
    coneAngle: float,
    *,
    forward: ArrayLike = (1.0, 0.0, 0.0),
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Uniformly sample unit directions within a cone in radians."""

    if numRays < 1 or not 0.0 <= coneAngle <= np.pi:
        raise ValueError("numRays must be positive and coneAngle in [0, pi].")
    axis = np.asarray(forward, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("forward must contain three finite values.")
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        raise ValueError("forward cannot be zero.")
    axis /= norm
    helper = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(axis, helper))) > 0.99:
        helper = np.asarray((0.0, 1.0, 0.0))
    first = np.cross(helper, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    generator = np.random.default_rng(seed)
    cosTheta = generator.uniform(np.cos(coneAngle), 1.0, numRays)
    sinTheta = np.sqrt(np.maximum(0.0, 1.0 - cosTheta**2))
    phi = generator.uniform(0.0, 2.0 * np.pi, numRays)
    return cosTheta[:, None] * axis + sinTheta[:, None] * (
        np.cos(phi)[:, None] * first + np.sin(phi)[:, None] * second
    )


def reflect(direction: ArrayLike, normal: ArrayLike) -> NDArray[np.float64]:
    """Specularly reflect one direction around a surface normal."""

    directionArray = _directions(np.asarray(direction)[None, :])[0]
    normalArray = _directions(np.asarray(normal)[None, :])[0]
    output = directionArray - 2.0 * np.dot(directionArray, normalArray) * normalArray
    return output / np.linalg.norm(output)


def refract(
    direction: ArrayLike,
    normal: ArrayLike,
    incidentIndex: float,
    transmittedIndex: float,
) -> NDArray[np.float64] | None:
    """Apply Snell's law, returning ``None`` for total internal reflection."""

    if (
        not np.isfinite(incidentIndex)
        or incidentIndex <= 0.0
        or not np.isfinite(transmittedIndex)
        or transmittedIndex <= 0.0
    ):
        raise ValueError("Refractive indices must be finite and positive.")
    ray = _directions(np.asarray(direction)[None, :])[0]
    surfaceNormal = _directions(np.asarray(normal)[None, :])[0]
    cosine = -float(np.dot(ray, surfaceNormal))
    sourceIndex, destinationIndex = incidentIndex, transmittedIndex
    if cosine < 0.0:
        surfaceNormal = -surfaceNormal
        cosine = -float(np.dot(ray, surfaceNormal))
        sourceIndex, destinationIndex = destinationIndex, sourceIndex
    ratio = sourceIndex / destinationIndex
    discriminant = 1.0 - ratio**2 * (1.0 - cosine**2)
    if discriminant < 0.0:
        return None
    output = ratio * ray + (ratio * cosine - np.sqrt(discriminant)) * surfaceNormal
    return output / np.linalg.norm(output)


def fresnel_schlick(
    cosine: float, incidentIndex: float, transmittedIndex: float
) -> float:
    """Return Schlick's unpolarized power-reflection approximation."""

    if not 0.0 <= cosine <= 1.0:
        raise ValueError("cosine must lie in [0, 1].")
    if (
        not np.isfinite(incidentIndex)
        or incidentIndex <= 0.0
        or not np.isfinite(transmittedIndex)
        or transmittedIndex <= 0.0
    ):
        raise ValueError("Refractive indices must be finite and positive.")
    base = (
        (transmittedIndex - incidentIndex) / (transmittedIndex + incidentIndex)
    ) ** 2
    return float(base + (1.0 - base) * (1.0 - cosine) ** 5)


@dataclass(frozen=True, slots=True)
class RayTraceResult:
    """First mesh intersection for each input ray."""

    origins: NDArray[np.float64]
    directions: NDArray[np.float64]
    hitMask: NDArray[np.bool_]
    points: NDArray[np.float64]
    distances: NDArray[np.float64]
    triangleIndices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RayPath:
    """Piecewise-linear SBR path."""

    points: NDArray[np.float64]
    hitReceiver: bool
    gain: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 3:
            raise ValueError("points must contain at least one 3D path vertex.")
        if not np.all(np.isfinite(points)) or not np.isfinite(self.gain):
            raise ValueError("Ray path vertices and gain must be finite.")
        immutable = points.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "points", immutable)

    @property
    def length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def to_propagation_path(
        self,
        *,
        txId: int,
        rxId: int,
        pathRate: float = 0.0,
    ) -> PropagationPath:
        """Convert a receiver-terminated ray into an ADC propagation path."""

        if not self.hitReceiver:
            raise ValueError("Only receiver-terminated rays form propagation paths.")
        return PropagationPath(self.length, txId, rxId, pathRate, self.gain)


def _sphere_distance(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    center: NDArray[np.float64],
    radius: float,
) -> float:
    relative = origin - center
    linear = float(np.dot(direction, relative))
    constant = float(np.dot(relative, relative) - radius**2)
    discriminant = linear**2 - constant
    if discriminant < 0.0:
        return np.inf
    candidates = (-linear - np.sqrt(discriminant), -linear + np.sqrt(discriminant))
    positive = [value for value in candidates if value > 1e-9]
    return min(positive, default=np.inf)


class TrimeshRayTracer:
    """Lazy optional Trimesh first-hit and specular SBR backend."""

    def __init__(self, mesh: Any) -> None:
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "TrimeshRayTracer requires `pip install pyradar[simulation]`."
            ) from exc
        if isinstance(mesh, trimesh.Scene):
            geometries = tuple(mesh.geometry.values())
            if not geometries:
                raise ValueError("The Trimesh scene does not contain geometry.")
            mesh = trimesh.util.concatenate(geometries)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh or trimesh.Scene.")
        self.mesh = mesh
        self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    @classmethod
    def from_file(cls, path: str | Path, *, process: bool = False) -> TrimeshRayTracer:
        """Load a mesh file without importing Trimesh at package import time."""

        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "TrimeshRayTracer requires `pip install pyradar[simulation]`."
            ) from exc
        meshPath = Path(path)
        if not meshPath.is_file():
            raise FileNotFoundError(meshPath)
        return cls(trimesh.load(meshPath, process=process))

    def trace(self, origins: ArrayLike, directions: ArrayLike) -> RayTraceResult:
        """Return the nearest mesh hit for every ray."""

        originArray = np.asarray(origins, dtype=float)
        directionArray = _directions(directions)
        if originArray.shape != directionArray.shape or not np.all(
            np.isfinite(originArray)
        ):
            raise ValueError("origins and directions must be finite (N, 3) arrays.")
        locations, rayIndices, triangleIndices = self._intersector.intersects_location(
            ray_origins=originArray,
            ray_directions=directionArray,
            multiple_hits=True,
        )
        count = originArray.shape[0]
        points = np.full((count, 3), np.nan, dtype=float)
        distances = np.full(count, np.inf, dtype=float)
        triangles = np.full(count, -1, dtype=np.int64)
        for location, rayId, triangleId in zip(
            locations, rayIndices, triangleIndices, strict=True
        ):
            distance = float(np.linalg.norm(location - originArray[rayId]))
            if distance < distances[rayId]:
                points[rayId] = location
                distances[rayId] = distance
                triangles[rayId] = triangleId
        return RayTraceResult(
            origins=originArray,
            directions=directionArray,
            hitMask=np.isfinite(distances),
            points=points,
            distances=distances,
            triangleIndices=triangles,
        )

    def shoot_and_bounce(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        receiverCenter: ArrayLike,
        receiverRadius: float,
        maxBounces: int = 3,
        reflectionCoefficient: complex = 1.0 + 0.0j,
        epsilon: float = 1e-6,
    ) -> tuple[RayPath, ...]:
        """Trace ideal specular rays until they hit a spherical receiver."""

        originArray = np.asarray(origins, dtype=float)
        directionArray = _directions(directions)
        center = np.asarray(receiverCenter, dtype=float)
        if (
            originArray.shape != directionArray.shape
            or not np.all(np.isfinite(originArray))
            or center.shape != (3,)
            or not np.all(np.isfinite(center))
        ):
            raise ValueError("Invalid ray or receiver geometry.")
        if (
            not np.isfinite(receiverRadius)
            or receiverRadius <= 0.0
            or maxBounces < 0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(reflectionCoefficient)
        ):
            raise ValueError("Invalid receiverRadius, maxBounces, or epsilon.")
        paths: list[RayPath] = []
        for origin, direction in zip(originArray, directionArray, strict=True):
            currentOrigin = origin.copy()
            currentDirection = direction.copy()
            points = [origin.copy()]
            gain = 1.0 + 0.0j
            hitReceiver = False
            for _ in range(maxBounces + 1):
                hit = self.trace(currentOrigin[None, :], currentDirection[None, :])
                surfaceDistance = hit.distances[0]
                receiverDistance = _sphere_distance(
                    currentOrigin, currentDirection, center, receiverRadius
                )
                if receiverDistance < surfaceDistance:
                    points.append(currentOrigin + receiverDistance * currentDirection)
                    hitReceiver = True
                    break
                if not hit.hitMask[0]:
                    break
                point = hit.points[0]
                points.append(point)
                normal = np.asarray(
                    self.mesh.face_normals[hit.triangleIndices[0]], dtype=float
                )
                currentDirection = reflect(currentDirection, normal)
                currentOrigin = point + epsilon * currentDirection
                gain *= reflectionCoefficient
            paths.append(RayPath(np.asarray(points), hitReceiver, gain))
        return tuple(paths)


__all__ = [
    "RayPath",
    "RayTraceResult",
    "TrimeshRayTracer",
    "fresnel_schlick",
    "reflect",
    "refract",
    "sample_cone_directions",
]
