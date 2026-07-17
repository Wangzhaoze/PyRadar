"""Simulation targets and propagation-path models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite FLU coordinates.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PointTarget:
    """Ideal isotropic point target in the radar FLU frame.

    Parameters
    ----------
    position:
        Target position at time zero as ``[x, y, z]`` in meters.
    velocity:
        Constant Cartesian velocity in meters per second.
    rcs:
        Monostatic radar cross section in square meters. Its square root is
        used as the complex field amplitude before optional propagation loss.
    phase:
        Additional reflection phase in radians.
    """

    position: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    rcs: float = 1.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, "velocity"))
        if not np.isfinite(self.rcs) or self.rcs < 0.0:
            raise ValueError("rcs must be finite and nonnegative.")
        if not np.isfinite(self.phase):
            raise ValueError("phase must be finite.")

    def position_at(self, time: float) -> NDArray[np.float64]:
        """Return the FLU position at ``time`` seconds."""

        if not np.isfinite(time):
            raise ValueError("time must be finite.")
        return self.position + self.velocity * time

    @property
    def reflection(self) -> complex:
        """Complex field reflection coefficient before path loss."""

        return complex(np.sqrt(self.rcs) * np.exp(1j * self.phase))


@dataclass(frozen=True, slots=True)
class PropagationPath:
    """One bistatic propagation path at the beginning of a frame.

    ``pathRate`` is positive when the total TX-target-RX path length is
    increasing. Consequently, an approaching target produces positive
    Doppler frequency and has a negative path rate.
    """

    pathLength: float
    txId: int
    rxId: int
    pathRate: float = 0.0
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if not np.isfinite(self.pathLength) or self.pathLength <= 0.0:
            raise ValueError("pathLength must be finite and positive.")
        if not np.isfinite(self.pathRate):
            raise ValueError("pathRate must be finite.")
        if self.txId < 0 or self.rxId < 0:
            raise ValueError("txId and rxId cannot be negative.")
        if not np.isfinite(self.amplitude.real) or not np.isfinite(self.amplitude.imag):
            raise ValueError("amplitude must be finite.")


@dataclass(frozen=True, slots=True)
class TargetScene:
    """An immutable collection of point targets."""

    targets: tuple[PointTarget, ...]

    def __post_init__(self) -> None:
        values = tuple(self.targets)
        if any(not isinstance(target, PointTarget) for target in values):
            raise TypeError("TargetScene accepts PointTarget objects only.")
        object.__setattr__(self, "targets", values)

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        *,
        velocities: ArrayLike | None = None,
        rcs: ArrayLike | float = 1.0,
        phase: ArrayLike | float = 0.0,
    ) -> TargetScene:
        """Build one point target per row of an ``(N, 3)`` point cloud."""

        positions = np.asarray(points, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        count = positions.shape[0]
        velocityArray = (
            np.zeros_like(positions)
            if velocities is None
            else np.broadcast_to(np.asarray(velocities, dtype=float), positions.shape)
        )
        rcsArray = np.broadcast_to(np.asarray(rcs, dtype=float), (count,))
        phaseArray = np.broadcast_to(np.asarray(phase, dtype=float), (count,))
        return cls(
            tuple(
                PointTarget(position, velocity, float(targetRcs), float(targetPhase))
                for position, velocity, targetRcs, targetPhase in zip(
                    positions,
                    velocityArray,
                    rcsArray,
                    phaseArray,
                    strict=True,
                )
            )
        )

    @classmethod
    def coerce(
        cls, targets: PointTarget | Sequence[PointTarget] | TargetScene
    ) -> TargetScene:
        """Normalize a target or target sequence into a scene."""

        if isinstance(targets, TargetScene):
            return targets
        if isinstance(targets, PointTarget):
            return cls((targets,))
        return cls(tuple(targets))

    def __len__(self) -> int:
        return len(self.targets)


__all__ = ["PointTarget", "PropagationPath", "TargetScene"]
