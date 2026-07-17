"""Physical transmit and receive array geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positions(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N, 2) or (N, 3).")
    if array.shape[1] == 2:
        # Two-column input is interpreted as lateral (y) and vertical (z).
        array = np.column_stack((np.zeros(array.shape[0]), array))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite SI coordinates.")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Transceivers:
    """TX/RX coordinates in the radar FLU frame, in meters.

    FLU is right handed: x points forward, y left, and z up. Antenna arrays
    normally lie in the y-z plane. Virtual phase centers are formed from the
    TX/RX channel map supplied by a MIMO strategy.
    """

    txPositions: NDArray[np.float64]
    rxPositions: NDArray[np.float64]
    azimuthOnlyChannels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "txPositions", _positions(self.txPositions, "txPositions")
        )
        object.__setattr__(
            self, "rxPositions", _positions(self.rxPositions, "rxPositions")
        )
        indices = tuple(int(index) for index in self.azimuthOnlyChannels)
        if any(index < 0 for index in indices):
            raise ValueError("azimuthOnlyChannels cannot contain negative indices.")
        object.__setattr__(self, "azimuthOnlyChannels", indices)

    @property
    def numTx(self) -> int:
        return int(self.txPositions.shape[0])

    @property
    def numRx(self) -> int:
        return int(self.rxPositions.shape[0])

    def virtual_positions(
        self, channelMap: Sequence[tuple[int, int]]
    ) -> NDArray[np.float64]:
        """Return monostatic virtual phase-center coordinates.

        The phase of a far-field return is proportional to
        ``(txPosition + rxPosition) dot direction``; therefore no arbitrary
        array offset or dataset-specific topology is required.
        """

        if not channelMap:
            raise ValueError("channelMap cannot be empty.")
        result = np.empty((len(channelMap), 3), dtype=np.float64)
        for channel, (txId, rxId) in enumerate(channelMap):
            if not 0 <= txId < self.numTx or not 0 <= rxId < self.numRx:
                raise ValueError(f"Invalid channel map entry {(txId, rxId)}.")
            result[channel] = self.txPositions[txId] + self.rxPositions[rxId]
        result.setflags(write=False)
        return result

    @staticmethod
    def phase_center_groups(
        positions: NDArray[np.float64], tolerance: float = 1e-9
    ) -> tuple[tuple[int, ...], ...]:
        """Group channels that share a virtual phase center."""

        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        keys = np.rint(np.asarray(positions) / tolerance).astype(np.int64)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, key in enumerate(keys):
            groupKey = (int(key[0]), int(key[1]), int(key[2]))
            groups.setdefault(groupKey, []).append(index)
        return tuple(tuple(group) for group in groups.values())
