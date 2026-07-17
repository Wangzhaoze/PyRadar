"""Named-dimension signal containers and typed processing results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .radar import Radar

AxisOrder = tuple[str, ...]
CANONICAL_ADC_DIMS: AxisOrder = ("loop", "emission", "rx", "sample")


def _array(
    value: ArrayLike, *, ndim: int | None = None, name: str = "array"
) -> NDArray[Any]:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}.")
    return result


def _vector(
    value: ArrayLike, length: int, name: str, dtype: Any = None
) -> NDArray[Any]:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1 or result.size != length:
        raise ValueError(f"{name} must have shape ({length},).")
    return result


@dataclass(frozen=True, slots=True)
class ADCFrame:
    """Raw ADC frame with explicit dimension names.

    Raw ndarrays are never assigned dimensions by guesswork. A singleton
    ``loop`` or ``emission`` dimension may be inserted only when the radar
    model proves that its size is one.
    """

    data: NDArray[Any]
    dims: AxisOrder
    radar: Radar | None = None
    timestamp: float | None = None
    frameId: int | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        dims = tuple(self.dims)
        if len(dims) != array.ndim:
            raise ValueError("dims length must equal data.ndim.")
        if len(set(dims)) != len(dims):
            raise ValueError("ADC dimension names must be unique.")
        unknown = set(dims) - set(CANONICAL_ADC_DIMS)
        if unknown:
            raise ValueError(f"Unknown ADC dimensions: {sorted(unknown)}.")
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        dims: Sequence[str] | None,
        radar: Radar | None = None,
        timestamp: float | None = None,
        frameId: int | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ADCFrame:
        if dims is None:
            raise ValueError(
                "Raw ADC data requires explicit dims, for example "
                "('loop', 'emission', 'rx', 'sample')."
            )
        return cls(
            data=np.asarray(data),
            dims=tuple(dims),
            radar=radar,
            timestamp=timestamp,
            frameId=frameId,
            metadata=metadata or {},
        )

    def canonical(self, radar: Radar | None = None) -> ADCFrame:
        """Return a view ordered as ``loop/emission/rx/sample``."""

        model = radar or self.radar
        if model is None:
            raise ValueError(
                "A Radar is required to validate and canonicalize ADC data."
            )
        data = self.data
        dims = list(self.dims)
        safeSingletons = {
            "loop": model.sampler.numLoops == 1,
            "emission": model.mimo.numEmissions == 1,
        }
        for dim in CANONICAL_ADC_DIMS:
            if dim in dims:
                continue
            if safeSingletons.get(dim, False):
                data = np.expand_dims(data, axis=0)
                dims.insert(0, dim)
                continue
            raise ValueError(
                f"ADC dimension {dim!r} is missing and cannot be inferred safely."
            )
        permutation = tuple(dims.index(dim) for dim in CANONICAL_ADC_DIMS)
        data = np.transpose(data, permutation)
        expected = (
            model.sampler.numLoops,
            model.mimo.numEmissions,
            model.mimo.numRx,
            model.sampler.numSamples,
        )
        if data.shape != expected:
            raise ValueError(
                f"Canonical ADC shape {data.shape} does not match Radar shape {expected}."
            )
        return ADCFrame(
            data=data,
            dims=CANONICAL_ADC_DIMS,
            radar=model,
            timestamp=self.timestamp,
            frameId=self.frameId,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class RadarCube:
    """An ndarray with named dimensions, coordinates, and processing stage."""

    data: NDArray[Any]
    dims: AxisOrder
    stage: str
    radar: Radar | None = None
    coords: Mapping[str, NDArray[Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        dims = tuple(self.dims)
        if len(dims) != array.ndim or len(set(dims)) != len(dims):
            raise ValueError("RadarCube dims must be unique and match data.ndim.")
        coordinates: dict[str, NDArray[Any]] = {}
        for name, values in self.coords.items():
            if name not in dims:
                raise ValueError(f"Coordinate {name!r} is not a cube dimension.")
            coordinate = np.asarray(values)
            if coordinate.ndim != 1 or coordinate.size != array.shape[dims.index(name)]:
                raise ValueError(f"Coordinate {name!r} has the wrong length.")
            coordinates[name] = coordinate
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "coords", coordinates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def magnitude(self) -> NDArray[Any]:
        return np.abs(self.data)

    @property
    def power(self) -> NDArray[Any]:
        return (
            np.abs(self.data) ** 2
            if np.iscomplexobj(self.data)
            else np.asarray(self.data)
        )

    def dim_index(self, name: str) -> int:
        try:
            return self.dims.index(name)
        except ValueError as exc:
            raise ValueError(
                f"Dimension {name!r} is not present in {self.dims}."
            ) from exc

    def transpose(self, *dims: str) -> RadarCube:
        """Transpose by dimension name."""

        if set(dims) != set(self.dims) or len(dims) != len(self.dims):
            raise ValueError("transpose dims must be a permutation of existing dims.")
        axes = tuple(self.dims.index(name) for name in dims)
        return RadarCube(
            data=np.transpose(self.data, axes),
            dims=tuple(dims),
            stage=self.stage,
            radar=self.radar,
            coords={name: self.coords[name] for name in dims if name in self.coords},
            metadata=self.metadata,
        )

    def with_data(
        self,
        data: ArrayLike,
        *,
        dims: Sequence[str] | None = None,
        stage: str | None = None,
        coords: Mapping[str, ArrayLike] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RadarCube:
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return RadarCube(
            data=np.asarray(data),
            dims=tuple(dims or self.dims),
            stage=stage or self.stage,
            radar=self.radar,
            coords=(
                self.coords
                if coords is None
                else {name: np.asarray(values) for name, values in coords.items()}
            ),
            metadata=merged,
        )


@dataclass(frozen=True, slots=True)
class DetectionSet:
    """CFAR detections and their source bins."""

    rangeBin: NDArray[np.int64]
    dopplerBin: NDArray[np.int64]
    power: NDArray[np.float64]
    noise: NDArray[np.float64]
    threshold: NDArray[np.float64]
    snr: NDArray[np.float64]
    azimuthBin: NDArray[np.int64] | None = None
    elevationBin: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        count = np.asarray(self.rangeBin).size
        for name in ("rangeBin", "dopplerBin"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), count, name, np.int64)
            )
        for name in ("power", "noise", "threshold", "snr"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), count, name, float)
            )
        for name in ("azimuthBin", "elevationBin"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(value, count, name, np.int64))

    def __len__(self) -> int:
        return int(self.rangeBin.size)

    @classmethod
    def empty(cls) -> DetectionSet:
        emptyFloat = np.empty(0, dtype=float)
        emptyInt = np.empty(0, dtype=np.int64)
        return cls(emptyInt, emptyInt, emptyFloat, emptyFloat, emptyFloat, emptyFloat)


@dataclass(frozen=True, slots=True)
class PointCloud:
    """Radar point cloud in FLU coordinates with velocity and provenance."""

    xyz: NDArray[np.float64]
    power: NDArray[np.float64]
    snr: NDArray[np.float64]
    radialVelocity: NDArray[np.float64]
    sourceBins: NDArray[np.int64]
    timestamp: float | None = None
    frameId: int | str | None = None

    def __post_init__(self) -> None:
        xyz = _array(self.xyz, ndim=2, name="xyz").astype(float, copy=False)
        if xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3).")
        count = xyz.shape[0]
        bins = _array(self.sourceBins, ndim=2, name="sourceBins").astype(
            np.int64, copy=False
        )
        if bins.shape != (count, 4):
            raise ValueError(
                "sourceBins must have shape (N, 4): range/doppler/azimuth/elevation."
            )
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "sourceBins", bins)
        for name in ("power", "snr", "radialVelocity"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), count, name, float)
            )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def range(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.xyz, axis=1)

    @property
    def azimuth(self) -> NDArray[np.float64]:
        return np.arctan2(self.xyz[:, 1], self.xyz[:, 0])

    @property
    def elevation(self) -> NDArray[np.float64]:
        horizontal = np.hypot(self.xyz[:, 0], self.xyz[:, 1])
        return np.arctan2(self.xyz[:, 2], horizontal)

    def to_numpy(self) -> NDArray[np.float64]:
        """Return columns ``x, y, z, power, snr, radialVelocity``."""

        return np.column_stack((self.xyz, self.power, self.snr, self.radialVelocity))

    @classmethod
    def empty(
        cls, *, timestamp: float | None = None, frameId: int | str | None = None
    ) -> PointCloud:
        return cls(
            xyz=np.empty((0, 3)),
            power=np.empty(0),
            snr=np.empty(0),
            radialVelocity=np.empty(0),
            sourceBins=np.empty((0, 4), dtype=np.int64),
            timestamp=timestamp,
            frameId=frameId,
        )


@dataclass(frozen=True, slots=True)
class ClusterSet:
    """Per-point labels and aggregate cluster statistics."""

    labels: NDArray[np.int64]
    clusterId: NDArray[np.int64]
    centroid: NDArray[np.float64]
    size: NDArray[np.float64]
    covariance: NDArray[np.float64]
    radialVelocity: NDArray[np.float64]
    power: NDArray[np.float64]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64)
        clusterIds = np.asarray(self.clusterId, dtype=np.int64)
        centroid = np.asarray(self.centroid, dtype=float)
        size = np.asarray(self.size, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if centroid.ndim != 2 or centroid.shape[1] not in (2, 3):
            raise ValueError("centroid must have shape (clusters, 2|3).")
        count, dimensions = centroid.shape
        if clusterIds.shape != (count,) or size.shape != centroid.shape:
            raise ValueError(
                "Cluster ids, centroids, and sizes have inconsistent shapes."
            )
        if covariance.shape != (count, dimensions, dimensions):
            raise ValueError("Cluster covariance has an inconsistent shape.")
        velocity = _vector(self.radialVelocity, count, "radialVelocity", float)
        power = _vector(self.power, count, "power", float)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "clusterId", clusterIds)
        object.__setattr__(self, "centroid", centroid)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "radialVelocity", velocity)
        object.__setattr__(self, "power", power)

    def __len__(self) -> int:
        return int(np.asarray(self.clusterId).size)


@dataclass(frozen=True, slots=True)
class TrackSet:
    """Public snapshot of multi-target tracker state."""

    trackId: NDArray[np.int64]
    status: tuple[str, ...]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    covariance: NDArray[np.float64]
    age: NDArray[np.int64]
    hits: NDArray[np.int64]
    misses: NDArray[np.int64]

    def __post_init__(self) -> None:
        trackIds = np.asarray(self.trackId, dtype=np.int64)
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        count = trackIds.size
        if trackIds.shape != (count,) or position.ndim != 2:
            raise ValueError("Track ids and positions have inconsistent shapes.")
        dimensions = position.shape[1]
        if dimensions not in (2, 3) or velocity.shape != position.shape:
            raise ValueError("Track position and velocity must use 2D or 3D shapes.")
        if covariance.shape != (count, 2 * dimensions, 2 * dimensions):
            raise ValueError("Track covariance has an inconsistent shape.")
        if len(self.status) != count or any(
            value not in {"tentative", "confirmed", "coasting", "deleted"}
            for value in self.status
        ):
            raise ValueError("Track status values are invalid or inconsistent.")
        object.__setattr__(self, "trackId", trackIds)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "covariance", covariance)
        for name in ("age", "hits", "misses"):
            object.__setattr__(
                self, name, _vector(getattr(self, name), count, name, np.int64)
            )

    def __len__(self) -> int:
        return int(np.asarray(self.trackId).size)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Products generated while processing one ADC frame."""

    adcFrame: ADCFrame
    detections: DetectionSet
    pointCloud: PointCloud
    rangeCube: RadarCube | None = None
    rangeDopplerCube: RadarCube | None = None
    rangeAngleCube: RadarCube | None = None
    rangeDopplerAngleCube: RadarCube | None = None
    clusters: ClusterSet | None = None
    tracks: TrackSet | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
