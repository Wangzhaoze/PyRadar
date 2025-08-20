#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : fft.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
import numpy as np
from typing import Literal
from scipy.fft import fft, fftshift
from typing import Optional, Union

# ######################################################################
# FFT Functions
# ######################################################################


def windowing(len: int, window_type: Literal['Hamming', 'Blackman', 'Hann']) -> np.ndarray:
    """
    Generate a window function based on the specified type.

    Args:
        len (int): Length of the window.
        window_type (Literal['Hamming', 'Blackman', 'Hann']): Type of the window function.
            - 'Hamming': Hamming window.
            - 'Blackman': Blackman window.
            - 'Hann': Hann window.

    Returns:
        np.ndarray: The generated window function as a 1D numpy array.

    Raises:
        ValueError: If an unsupported window type is provided.
    """
    if window_type == 'Hamming':
        window = np.hamming(len)
    elif window_type == 'Blackman':
        window = np.blackman(len)
    elif window_type == 'Hann':
        window = np.hanning(len)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    return window


def range_fft(
    signal: np.ndarray, 
    numRangeBins: Optional[int] = None,
    IdxSamples: int = 1, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the range dimension of the ADC cube, with optional windowing.

    Args:
        signal (np.ndarray): Input ADC data cube.
        numRangeBins (Optional[int]): Number of range bins for zero-padding. Defaults to None.
        IdxSamples (int): Index of the range dimension in the input array. Defaults to 1.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The range FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input signal is not a numpy array.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError('Input must be a numpy array.')

    # Apply window if specified
    if window_type is not None:
        numADCSamples = signal.shape[IdxSamples]
        window = windowing(numADCSamples, window_type)
        shape = [1] * signal.ndim
        shape[IdxSamples] = numADCSamples
        window = window.reshape(shape)
        signal = signal * window

    range_spectrum = fft(signal, n=numRangeBins, axis=IdxSamples, workers=num_workers)
    return range_spectrum


def doppler_fft(
    signal: np.ndarray, 
    numDopplerBins: Optional[int] = None,
    IdxChirps: int = 2, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the Doppler (chirps) dimension of the ADC cube, with optional windowing.

    Args:
        signal (np.ndarray): Input ADC data cube.
        numDopplerBins (Optional[int]): Number of Doppler bins for zero-padding. Defaults to None.
        IdxChirps (int): Index of the Doppler dimension in the input array. Defaults to 2.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The Doppler FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input signal is not a numpy array.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError('Input must be a numpy array.')


    # Apply window if specified
    if window_type is not None:
        num_chirps = signal.shape[IdxChirps]
        window = windowing(num_chirps, window_type)
        shape = [1] * signal.ndim
        shape[IdxChirps] = num_chirps
        window = window.reshape(shape)
        signal = signal * window

    doppler_spectrum = fftshift(
        fft(signal, n=numDopplerBins, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return doppler_spectrum


def angle_fft(
    adc_cube: np.ndarray, 
    numAngleBins: Optional[int] = None,
    IdxVirtualAntennas: int = 0, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the virtual antennas (azimuth) dimension of the ADC cube, with optional windowing.

    Args:
        adc_cube (np.ndarray): Input ADC data cube.
        numAngleBins (Optional[int]): Number of angle bins for zero-padding. Defaults to None.
        IdxVirtualAntennas (int): Index of the virtual antennas dimension in the input array. Defaults to 0.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The angle FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input adc_cube is not a numpy array or does not have three dimensions.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Apply window if specified
    if window_type is not None:
        num_antenna = adc_cube.shape[IdxVirtualAntennas]
        window = windowing(num_antenna, window_type)
        shape = [1, 1, 1]
        shape[IdxVirtualAntennas] = num_antenna
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    azimuth_spectrum = fft(adc_cube, n=numAngleBins, axis=IdxVirtualAntennas, workers=num_workers)
    azimuth_spectrum = fftshift(azimuth_spectrum, axes=IdxVirtualAntennas)
    return azimuth_spectrum


def range_doppler_fft(
    adc_cube: np.ndarray,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    num_workers: Optional[int] = None,
    range_window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
    doppler_window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
) -> np.ndarray:
    """
    Perform Range FFT (with optional windowing) followed by Doppler FFT (with optional windowing) to generate a Range-Doppler map.

    Args:
        adc_cube (np.ndarray): Input ADC data cube with three dimensions (virtual antennas, range, chirps).
        IdxSamples (int): Index of the range dimension in the input array. Defaults to 1.
        IdxChirps (int): Index of the Doppler dimension in the input array. Defaults to 2.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        range_window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function for the range dimension. Defaults to None.
        doppler_window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function for the Doppler dimension. Defaults to None.

    Returns:
        np.ndarray: The Range-Doppler map as a numpy array.

    Raises:
        ValueError: If the input ADC cube is not a numpy array or does not have three dimensions.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    range_fft_spectrum = range_fft(
        adc_cube, 
        IdxSamples=IdxSamples, 
        num_workers=num_workers, 
        window_type=range_window_type
    )
    doppler_fft_spectrum = doppler_fft(
        range_fft_spectrum, 
        IdxChirps=IdxChirps, 
        num_workers=num_workers,
        window_type=doppler_window_type
    )
    return doppler_fft_spectrum

