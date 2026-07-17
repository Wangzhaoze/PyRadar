"""MIMO multiplexing strategies.

Each strategy consumes canonical ADC data with dimensions
``(loop, emission, rx, ...)`` and returns ``(slow_time, virtual, ...)`` using a
TX-major channel order. No strategy knows about a dataset or radar model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


class MIMOScheme(ABC):
    """Interface implemented by all MIMO strategies."""

    scheme: ClassVar[str]
    numTx: int
    numRx: int

    @property
    @abstractmethod
    def numEmissions(self) -> int:
        """Number of captured emissions in each raw loop."""

    @property
    def channelMap(self) -> tuple[tuple[int, int], ...]:
        """TX/RX id for each decoded virtual channel."""

        return tuple(
            (txId, rxId) for txId in range(self.numTx) for rxId in range(self.numRx)
        )

    @abstractmethod
    def decode(self, adc: NDArray[Any]) -> NDArray[Any]:
        """Decode canonical ADC or range data into virtual channels."""

    @abstractmethod
    def slow_time_interval(self, chirpInterval: float) -> float:
        """Time between adjacent decoded slow-time samples."""

    def phase_correction(
        self,
        dopplerSize: int,
        chirpInterval: float,
    ) -> NDArray[np.complex128]:
        """Doppler phase correction indexed by Doppler bin and channel."""

        return np.ones((dopplerSize, self.numTx * self.numRx), dtype=np.complex128)

    def _validate(self, adc: NDArray[Any]) -> NDArray[Any]:
        array = np.asarray(adc)
        if array.ndim < 4:
            raise ValueError(
                "MIMO input must have dimensions (loop, emission, rx, ...)."
            )
        if array.shape[1] != self.numEmissions:
            raise ValueError(
                f"Expected {self.numEmissions} emissions, got {array.shape[1]}."
            )
        if array.shape[2] != self.numRx:
            raise ValueError(
                f"Expected {self.numRx} RX channels, got {array.shape[2]}."
            )
        return array


@dataclass(frozen=True, slots=True)
class SIMO(MIMOScheme):
    """Single-transmitter acquisition."""

    numRx: int
    numTx: int = field(init=False, default=1)
    scheme: ClassVar[str] = "simo"

    def __post_init__(self) -> None:
        if self.numRx < 1:
            raise ValueError("numRx must be positive.")

    @property
    def numEmissions(self) -> int:
        return 1

    def decode(self, adc: NDArray[Any]) -> NDArray[Any]:
        return self._validate(adc)[:, 0]

    def slow_time_interval(self, chirpInterval: float) -> float:
        return chirpInterval


@dataclass(frozen=True, slots=True)
class TDM(MIMOScheme):
    """Time-division MIMO with arbitrary TX order and emission timing."""

    numTx: int
    numRx: int
    txOrder: tuple[int, ...] = ()
    emissionTimeOffsets: tuple[float, ...] = ()
    cyclePeriod: float | None = None
    scheme: ClassVar[str] = "tdm"

    def __post_init__(self) -> None:
        if self.numTx < 1 or self.numRx < 1:
            raise ValueError("numTx and numRx must be positive.")
        order = self.txOrder or tuple(range(self.numTx))
        if sorted(order) != list(range(self.numTx)):
            raise ValueError("txOrder must be a permutation of 0..numTx-1.")
        object.__setattr__(self, "txOrder", tuple(int(item) for item in order))
        if self.emissionTimeOffsets:
            if len(self.emissionTimeOffsets) != self.numTx:
                raise ValueError("emissionTimeOffsets must match numTx.")
            offsets = tuple(float(value) for value in self.emissionTimeOffsets)
            if offsets[0] < 0.0 or any(
                b <= a for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            ):
                raise ValueError(
                    "emissionTimeOffsets must be nonnegative and increasing."
                )
            object.__setattr__(self, "emissionTimeOffsets", offsets)
        if self.cyclePeriod is not None and self.cyclePeriod <= 0.0:
            raise ValueError("cyclePeriod must be positive.")

    @property
    def numEmissions(self) -> int:
        return self.numTx

    def decode(self, adc: NDArray[Any]) -> NDArray[Any]:
        array = self._validate(adc)
        ordered = np.empty(
            (array.shape[0], self.numTx, self.numRx, *array.shape[3:]),
            dtype=array.dtype,
        )
        for emissionId, txId in enumerate(self.txOrder):
            ordered[:, txId] = array[:, emissionId]
        return ordered.reshape(
            array.shape[0], self.numTx * self.numRx, *array.shape[3:]
        )

    def slow_time_interval(self, chirpInterval: float) -> float:
        return self.cyclePeriod or self.numEmissions * chirpInterval

    def tx_time_offsets(self, chirpInterval: float) -> NDArray[np.float64]:
        offsets = (
            np.asarray(self.emissionTimeOffsets, dtype=float)
            if self.emissionTimeOffsets
            else np.arange(self.numTx, dtype=float) * chirpInterval
        )
        result = np.empty(self.numTx, dtype=float)
        for emissionId, txId in enumerate(self.txOrder):
            result[txId] = offsets[emissionId]
        return result

    def phase_correction(
        self,
        dopplerSize: int,
        chirpInterval: float,
    ) -> NDArray[np.complex128]:
        interval = self.slow_time_interval(chirpInterval)
        frequencies = np.fft.fftshift(np.fft.fftfreq(dopplerSize, d=interval))
        txOffsets = self.tx_time_offsets(chirpInterval)
        channelOffsets = np.repeat(txOffsets, self.numRx)
        return np.exp(-2j * np.pi * frequencies[:, None] * channelOffsets[None, :])


@dataclass(frozen=True, slots=True)
class BPM(MIMOScheme):
    """Two-transmitter binary phase modulation with Hadamard decoding."""

    numRx: int
    codeMatrix: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 1.0),
        (1.0, -1.0),
    )
    numTx: int = field(init=False, default=2)
    scheme: ClassVar[str] = "bpm"

    def __post_init__(self) -> None:
        if self.numRx < 1:
            raise ValueError("numRx must be positive.")
        matrix = np.asarray(self.codeMatrix, dtype=float)
        if matrix.shape != (2, 2) or np.linalg.matrix_rank(matrix) != 2:
            raise ValueError("BPM codeMatrix must be an invertible 2x2 matrix.")

    @property
    def numEmissions(self) -> int:
        return 2

    def decode(self, adc: NDArray[Any]) -> NDArray[Any]:
        array = self._validate(adc)
        decoder = np.linalg.inv(np.asarray(self.codeMatrix, dtype=float))
        decoded = np.einsum("te,ler...->ltr...", decoder, array, optimize=True)
        return decoded.reshape(
            array.shape[0], self.numTx * self.numRx, *array.shape[3:]
        )

    def slow_time_interval(self, chirpInterval: float) -> float:
        return 2.0 * chirpInterval


@dataclass(frozen=True, slots=True)
class DDM(MIMOScheme):
    """Doppler-division MIMO using orthogonal linear slow-time phase codes.

    ``dopplerOffsets`` are normalized cycles per raw chirp. Decoding correlates
    one complete code period, so the input loop count must be divisible by
    ``codeLength``. The target is assumed stationary over one code period.
    """

    numTx: int
    numRx: int
    dopplerOffsets: tuple[float, ...]
    codeLength: int
    scheme: ClassVar[str] = "ddm"

    def __post_init__(self) -> None:
        if self.numTx < 1 or self.numRx < 1 or self.codeLength < self.numTx:
            raise ValueError(
                "DDM requires positive dimensions and codeLength >= numTx."
            )
        if len(self.dopplerOffsets) != self.numTx:
            raise ValueError("dopplerOffsets must match numTx.")
        codes = self.codes
        gram = codes @ codes.conj().T / self.codeLength
        if not np.allclose(gram, np.eye(self.numTx), atol=1e-10):
            raise ValueError("DDM phase codes must be orthogonal over codeLength.")

    @property
    def numEmissions(self) -> int:
        return 1

    @property
    def codes(self) -> NDArray[np.complex128]:
        indices = np.arange(self.codeLength, dtype=float)
        offsets = np.asarray(self.dopplerOffsets, dtype=float)
        return np.exp(2j * np.pi * offsets[:, None] * indices[None, :])

    def decode(self, adc: NDArray[Any]) -> NDArray[Any]:
        array = self._validate(adc)[:, 0]
        if array.shape[0] % self.codeLength:
            raise ValueError("DDM loop count must be divisible by codeLength.")
        blocks = array.reshape(
            array.shape[0] // self.codeLength,
            self.codeLength,
            self.numRx,
            *array.shape[2:],
        )
        decoded = np.einsum(
            "tl,blr...->btr...",
            self.codes.conj() / self.codeLength,
            blocks,
            optimize=True,
        )
        return decoded.reshape(
            blocks.shape[0], self.numTx * self.numRx, *array.shape[2:]
        )

    def slow_time_interval(self, chirpInterval: float) -> float:
        return self.codeLength * chirpInterval
