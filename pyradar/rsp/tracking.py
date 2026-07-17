"""Constant-velocity Kalman/EKF multi-target tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from pyradar.base.cube import ClusterSet, PointCloud, TrackSet
from pyradar.base.processing import TrackingConfig


@dataclass
class _Track:
    trackId: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    status: Literal["tentative", "confirmed", "coasting", "deleted"] = "tentative"
    age: int = 1
    hits: int = 1
    misses: int = 0


class MultiTargetTracker:
    """GNN tracker with Hungarian assignment and radial-velocity EKF updates."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig(enabled=True)
        self.dimensions = self.config.dimensions
        self._tracks: list[_Track] = []
        self._nextId = 1
        self._lastTimestamp: float | None = None

    def reset(self) -> None:
        """Delete all tracker state and restart track ids."""

        self._tracks.clear()
        self._nextId = 1
        self._lastTimestamp = None

    def _transition(
        self, deltaTime: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dimensions = self.dimensions
        transition = np.eye(2 * dimensions)
        transition[:dimensions, dimensions:] = np.eye(dimensions) * deltaTime
        acceleration = np.vstack(
            (
                0.5 * deltaTime**2 * np.eye(dimensions),
                deltaTime * np.eye(dimensions),
            )
        )
        process = self.config.processNoise**2 * acceleration @ acceleration.T
        return transition, process

    def _predict(self, deltaTime: float) -> None:
        transition, process = self._transition(deltaTime)
        for track in self._tracks:
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + process
            track.age += 1

    def _position_update(self, track: _Track, position: NDArray[np.float64]) -> None:
        dimensions = self.dimensions
        measurement = np.zeros((dimensions, 2 * dimensions))
        measurement[:, :dimensions] = np.eye(dimensions)
        noise = self.config.measurementNoise**2 * np.eye(dimensions)
        innovation = position - measurement @ track.state
        covariance = measurement @ track.covariance @ measurement.T + noise
        gain = track.covariance @ measurement.T @ np.linalg.pinv(covariance)
        track.state = track.state + gain @ innovation
        identity = np.eye(2 * dimensions)
        residual = identity - gain @ measurement
        track.covariance = (
            residual @ track.covariance @ residual.T + gain @ noise @ gain.T
        )

    def _radial_update(self, track: _Track, radialVelocity: float) -> None:
        dimensions = self.dimensions
        position = track.state[:dimensions]
        velocity = track.state[dimensions:]
        radius = np.linalg.norm(position)
        if radius <= 1e-9 or not np.isfinite(radialVelocity):
            return
        predicted = float(position @ velocity / radius)
        jacobian = np.empty(2 * dimensions, dtype=float)
        jacobian[:dimensions] = (
            velocity / radius - position * (position @ velocity) / radius**3
        )
        jacobian[dimensions:] = position / radius
        variance = float(
            jacobian @ track.covariance @ jacobian + self.config.radialVelocityNoise**2
        )
        gain = track.covariance @ jacobian / variance
        track.state = track.state + gain * (radialVelocity - predicted)
        track.covariance = (
            np.eye(2 * dimensions) - np.outer(gain, jacobian)
        ) @ track.covariance

    def _association_cost(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        dimensions = self.dimensions
        measurement = np.zeros((dimensions, 2 * dimensions))
        measurement[:, :dimensions] = np.eye(dimensions)
        noise = self.config.measurementNoise**2 * np.eye(dimensions)
        cost = np.full((len(self._tracks), positions.shape[0]), np.inf)
        for trackId, track in enumerate(self._tracks):
            covariance = measurement @ track.covariance @ measurement.T + noise
            inverse = np.linalg.pinv(covariance)
            residual = positions - track.state[:dimensions]
            distances = np.einsum("ni,ij,nj->n", residual, inverse, residual)
            accepted = distances <= self.config.gatingThreshold
            cost[trackId, accepted] = distances[accepted]
        return cost

    def _new_track(self, position: NDArray[np.float64], radialVelocity: float) -> None:
        dimensions = self.dimensions
        state = np.zeros(2 * dimensions, dtype=float)
        state[:dimensions] = position
        radius = np.linalg.norm(position)
        if radius > 1e-9 and np.isfinite(radialVelocity):
            state[dimensions:] = radialVelocity * position / radius
        covariance = np.diag(
            [self.config.measurementNoise**2] * dimensions
            + [self.config.radialVelocityNoise**2] * dimensions
        )
        status: Literal["tentative", "confirmed"] = (
            "confirmed" if self.config.confirmationHits <= 1 else "tentative"
        )
        self._tracks.append(
            _Track(
                trackId=self._nextId,
                state=state,
                covariance=covariance,
                status=status,
            )
        )
        self._nextId += 1

    def update(
        self,
        pointCloud: PointCloud,
        *,
        clusters: ClusterSet | None = None,
        timestamp: float | None = None,
        deltaTime: float | None = None,
    ) -> TrackSet:
        """Predict, globally associate, and update tracks for one frame."""

        if deltaTime is None:
            if timestamp is not None and self._lastTimestamp is not None:
                deltaTime = timestamp - self._lastTimestamp
            else:
                deltaTime = 0.1
        if not np.isfinite(deltaTime) or deltaTime <= 0.0:
            raise ValueError("deltaTime must be finite and positive.")
        if timestamp is not None:
            self._lastTimestamp = timestamp
        self._predict(float(deltaTime))

        if clusters is not None and len(clusters):
            positions = np.asarray(clusters.centroid, dtype=float)[:, : self.dimensions]
            velocities = np.asarray(clusters.radialVelocity, dtype=float)
        else:
            positions = pointCloud.xyz[:, : self.dimensions]
            velocities = pointCloud.radialVelocity

        matchedTracks: set[int] = set()
        matchedMeasurements: set[int] = set()
        if self._tracks and positions.size:
            cost = self._association_cost(positions)
            finite = np.isfinite(cost)
            if np.any(finite):
                assignmentCost = np.where(finite, cost, 1e12)
                rows, columns = linear_sum_assignment(assignmentCost)
                for row, column in zip(rows, columns, strict=True):
                    if not np.isfinite(cost[row, column]):
                        continue
                    track = self._tracks[row]
                    self._position_update(track, positions[column])
                    self._radial_update(track, float(velocities[column]))
                    track.hits += 1
                    track.misses = 0
                    if track.hits >= self.config.confirmationHits:
                        track.status = "confirmed"
                    matchedTracks.add(row)
                    matchedMeasurements.add(column)

        for index, track in enumerate(self._tracks):
            if index in matchedTracks:
                continue
            track.misses += 1
            if track.misses >= self.config.deletionMisses:
                track.status = "deleted"
            elif track.status == "confirmed":
                track.status = "coasting"

        for index, (position, velocity) in enumerate(
            zip(positions, velocities, strict=True)
        ):
            if index not in matchedMeasurements:
                self._new_track(position, float(velocity))

        snapshotTracks = list(self._tracks)
        self._tracks = [track for track in self._tracks if track.status != "deleted"]
        return self._snapshot(snapshotTracks)

    def _snapshot(self, tracks: list[_Track] | None = None) -> TrackSet:
        current = self._tracks if tracks is None else tracks
        dimensions = self.dimensions
        if not current:
            return TrackSet(
                trackId=np.empty(0, dtype=np.int64),
                status=(),
                position=np.empty((0, dimensions)),
                velocity=np.empty((0, dimensions)),
                covariance=np.empty((0, 2 * dimensions, 2 * dimensions)),
                age=np.empty(0, dtype=np.int64),
                hits=np.empty(0, dtype=np.int64),
                misses=np.empty(0, dtype=np.int64),
            )
        return TrackSet(
            trackId=np.asarray([track.trackId for track in current], dtype=np.int64),
            status=tuple(track.status for track in current),
            position=np.stack([track.state[:dimensions] for track in current]),
            velocity=np.stack([track.state[dimensions:] for track in current]),
            covariance=np.stack([track.covariance for track in current]),
            age=np.asarray([track.age for track in current], dtype=np.int64),
            hits=np.asarray([track.hits for track in current], dtype=np.int64),
            misses=np.asarray([track.misses for track in current], dtype=np.int64),
        )


__all__ = ["MultiTargetTracker"]
