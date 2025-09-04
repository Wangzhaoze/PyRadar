import numpy as np
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import sliding_window_view
from functools import wraps


def _make_kernel(win: int, guard: int) -> np.ndarray:
    """
    Create CFAR convolution kernel.
    1 = training cells, 0 = guard + CUT.
    """
    full_win = 2 * win + 1
    kernel = np.ones((full_win, full_win), dtype=np.float32)
    kernel[win-guard:win+guard+1, win-guard:win+guard+1] = 0
    return kernel


# =============================
# Decorator for unified CFAR API
# =============================

def cfar_2d(func):
    """
    Decorator to provide a unified interface for all 2D CFAR detectors.
    Ensures consistent argument handling and allows easy switching between methods.
    """
    @wraps(func)
    def wrapper(spectrum: np.ndarray, *args, **kwargs) -> np.ndarray:
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("Input spectrum must be a NumPy array")
        if spectrum.ndim != 2:
            raise ValueError("Spectrum must be a 2D range-Doppler map")
        return func(spectrum, *args, **kwargs)
    return wrapper


# =============================
# CFAR Variants
# =============================

@cfar_2d
def ca_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
               scale: float = 2.0) -> np.ndarray:
    """Cell Averaging CFAR (CA-CFAR)."""
    kernel = _make_kernel(win, guard)
    n_train = kernel.sum()

    sum_train = convolve2d(spectrum, kernel, mode='same', boundary='symm')
    mean_train = sum_train / (n_train + 1e-10)

    mask = (spectrum > mean_train * scale).astype(np.uint8)
    return mask


@cfar_2d
def vi_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
               scale: float = 3.0, ratio_threshold: float = 2.0) -> np.ndarray:
    """Variability Index CFAR (VI-CFAR)."""
    kernel = _make_kernel(win, guard)
    n_train = kernel.sum()

    sum_train = convolve2d(spectrum, kernel, mode='same', boundary='symm')
    mean_val = sum_train / (n_train + 1e-10)

    sum_sq = convolve2d(spectrum**2, kernel, mode='same', boundary='symm')
    mean_sq = sum_sq / (n_train + 1e-10)
    std_val = np.sqrt(mean_sq - mean_val**2 + 1e-10)

    variability_ratio = std_val / (mean_val + 1e-10)
    adaptive_scale = scale * (1 + variability_ratio * ratio_threshold)

    noise_level = mean_val * adaptive_scale
    mask = (spectrum > noise_level).astype(np.uint8)
    return mask


@cfar_2d
def oa_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
               scale: float = 2.5) -> np.ndarray:
    """Ordered Averaging CFAR (OA-CFAR)."""
    full_win = 2 * win + 1
    kernel = _make_kernel(win, guard)
    n_train = int(kernel.sum())

    windows = sliding_window_view(np.pad(spectrum, win, mode="symmetric"),
                                  (full_win, full_win))
    tcells = windows[..., kernel.astype(bool)]
    sorted_cells = np.sort(tcells, axis=-1)

    k = np.maximum(1, n_train // 4)  # top 25%
    noise_level = sorted_cells[..., -k:].mean(axis=-1)

    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def os_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
               scale: float = 3.0, k: int = 12) -> np.ndarray:
    """Ordered Statistics CFAR (OS-CFAR)."""
    full_win = 2 * win + 1
    kernel = _make_kernel(win, guard)

    windows = sliding_window_view(np.pad(spectrum, win, mode="symmetric"),
                                  (full_win, full_win))
    tcells = windows[..., kernel.astype(bool)]
    sorted_cells = np.sort(tcells, axis=-1)

    k_idx = np.minimum(k, sorted_cells.shape[-1] - 1)
    noise_level = np.take_along_axis(
        sorted_cells,
        np.full(sorted_cells.shape[:-1], k_idx)[..., None],
        axis=-1
    )[..., 0]

    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def trimmed_mean_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
                         scale: float = 2.5, trim_ratio: float = 0.1) -> np.ndarray:
    """Trimmed Mean CFAR (TM-CFAR)."""
    full_win = 2 * win + 1
    kernel = _make_kernel(win, guard)

    windows = sliding_window_view(np.pad(spectrum, win, mode="symmetric"),
                                  (full_win, full_win))
    tcells = windows[..., kernel.astype(bool)]
    sorted_cells = np.sort(tcells, axis=-1)

    n = sorted_cells.shape[-1]
    t = int(n * trim_ratio)
    noise_level = sorted_cells[..., t:n-t].mean(axis=-1)

    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def cml_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
                scale: float = 2.5) -> np.ndarray:
    """Censored Mean Level CFAR (CML-CFAR)."""
    full_win = 2 * win + 1
    kernel = _make_kernel(win, guard)

    windows = sliding_window_view(np.pad(spectrum, win, mode="symmetric"),
                                  (full_win, full_win))
    tcells = windows[..., kernel.astype(bool)]

    censored = np.where(tcells > spectrum[..., None], np.nan, tcells)
    noise_level = np.nanmean(censored, axis=-1)

    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def goca_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
                 scale: float = 2.0) -> np.ndarray:
    """Greatest-Of CA-CFAR (GOCA-CFAR)."""
    ns, nc = spectrum.shape
    kernel = np.ones((2*win+1, 2*win+1), dtype=float)
    kernel[win-guard:win+guard+1, :] = 0

    left_kernel = kernel.copy()
    left_kernel[:, win+1:] = 0
    right_kernel = kernel.copy()
    right_kernel[:, :win] = 0

    left_mean = convolve2d(spectrum, left_kernel, mode='same', boundary='symm') / left_kernel.sum()
    right_mean = convolve2d(spectrum, right_kernel, mode='same', boundary='symm') / right_kernel.sum()

    noise_level = np.maximum(left_mean, right_mean)
    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def soca_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
                 scale: float = 2.0) -> np.ndarray:
    """Smallest-Of CA-CFAR (SOCA-CFAR)."""
    ns, nc = spectrum.shape
    kernel = np.ones((2*win+1, 2*win+1), dtype=float)
    kernel[win-guard:win+guard+1, :] = 0

    left_kernel = kernel.copy()
    left_kernel[:, win+1:] = 0
    right_kernel = kernel.copy()
    right_kernel[:, :win] = 0

    left_mean = convolve2d(spectrum, left_kernel, mode='same', boundary='symm') / left_kernel.sum()
    right_mean = convolve2d(spectrum, right_kernel, mode='same', boundary='symm') / right_kernel.sum()

    noise_level = np.minimum(left_mean, right_mean)
    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask


@cfar_2d
def median_cfar_2d(spectrum: np.ndarray, win: int = 8, guard: int = 2,
                   scale: float = 2.0) -> np.ndarray:
    """Median CFAR."""
    full_win = 2 * win + 1
    kernel = _make_kernel(win, guard)

    windows = sliding_window_view(np.pad(spectrum, win, mode="symmetric"),
                                  (full_win, full_win))
    tcells = windows[..., kernel.astype(bool)]

    noise_level = np.median(tcells, axis=-1)
    mask = (spectrum > noise_level * scale).astype(np.uint8)
    return mask
