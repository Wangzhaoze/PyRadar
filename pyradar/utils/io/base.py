"""Reader contracts and format-independent array helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.io import loadmat

from pyradar.base import ADCFrame, Radar

PathLike = str | Path


@runtime_checkable
class FrameReader(Protocol):
    """Protocol implemented by all dataset ADC readers."""

    @property
    def radar(self) -> Radar:
        """Physical radar and capture profile represented by this reader."""

    def __len__(self) -> int:
        """Number of readable frames."""

    def read(self, frameIndex: int) -> ADCFrame:
        """Decode one frame without running signal processing."""


def load_npy(path: PathLike) -> NDArray[Any]:
    """Load an ndarray without enabling pickle deserialization."""

    return np.load(Path(path), allow_pickle=False)


def save_npy(path: PathLike, array: ArrayLike) -> None:
    """Save an ndarray in NumPy's portable ``.npy`` format."""

    np.save(Path(path), np.asarray(array), allow_pickle=False)


def load_mat_array(path: PathLike, key: str | None = None) -> NDArray[Any]:
    """Load one public array from a MATLAB file."""

    values = loadmat(Path(path))
    if key is not None:
        if key not in values:
            raise KeyError(f"MAT key {key!r} not found in {path}.")
        return np.asarray(values[key])
    keys = [name for name in values if not name.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"MAT file contains {len(keys)} public arrays: {keys}.")
    return np.asarray(values[keys[0]])


def read_complex_iq_bin(
    path: PathLike, *, dtype: Any = np.int16
) -> NDArray[np.complex64]:
    """Read interleaved scalar I/Q values as a complex vector."""

    raw = np.fromfile(Path(path), dtype=dtype)
    if raw.size % 2:
        raise ValueError("Interleaved I/Q file contains an odd scalar count.")
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)).astype(
        np.complex64, copy=False
    )
