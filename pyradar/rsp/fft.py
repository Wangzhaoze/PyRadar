#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : fft.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
from scipy.ndimage import convolve
import numpy as np
from typing import Literal
from scipy.fft import fft, fftshift
from typing import Optional, Union
from ..utils import *

# ######################################################################
# FFT Functions
# ######################################################################


def range_fft(
    adc_cube: np.ndarray, 
    IdxSamples: int = 1, 
    num_workers: Optional[int] = None,
    windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the range dimension of the ADC cube, with optional windowing.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Apply window if specified
    if windowing is not None:
        num_samples = adc_cube.shape[IdxSamples]
        if windowing == 'Hamming':
            window = np.hamming(num_samples)
        elif windowing == 'Blackman':
            window = np.blackman(num_samples)
        elif windowing == 'Hann':
            window = np.hanning(num_samples)
        else:
            raise ValueError(f"Unsupported window type: {windowing}")
        shape = [1, 1, 1]
        shape[IdxSamples] = num_samples
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    range_spectrum = fft(adc_cube, axis=IdxSamples, workers=num_workers)
    return range_spectrum


def doppler_fft(
    adc_cube: np.ndarray, 
    IdxChirps: int = 2, 
    num_workers: Optional[int] = None,
    windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the Doppler (chirps) dimension of the ADC cube, with optional windowing.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Apply window if specified
    if windowing is not None:
        num_chirps = adc_cube.shape[IdxChirps]
        if windowing == 'Hamming':
            window = np.hamming(num_chirps)
        elif windowing == 'Blackman':
            window = np.blackman(num_chirps)
        elif windowing == 'Hann':
            window = np.hanning(num_chirps)
        else:
            raise ValueError(f"Unsupported window type: {windowing}")
        shape = [1, 1, 1]
        shape[IdxChirps] = num_chirps
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    doppler_spectrum = fftshift(
        fft(adc_cube, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return doppler_spectrum


def angle_fft(
    adc_cube: np.ndarray, 
    IdxVirtualAntennas: int = 0, 
    num_workers: Optional[int] = None,
    windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the virtual antennas (azimuth) dimension of the ADC cube, with optional windowing.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Apply window if specified
    if windowing is not None:
        num_ant = adc_cube.shape[IdxVirtualAntennas]
        if windowing == 'Hamming':
            window = np.hamming(num_ant)
        elif windowing == 'Blackman':
            window = np.blackman(num_ant)
        elif windowing == 'Hann':
            window = np.hanning(num_ant)
        else:
            raise ValueError(f"Unsupported window type: {windowing}")
        shape = [1, 1, 1]
        shape[IdxVirtualAntennas] = num_ant
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    azimuth_spectrum = fft(adc_cube, axis=IdxVirtualAntennas, workers=num_workers)
    azimuth_spectrum = fftshift(azimuth_spectrum, axes=IdxVirtualAntennas)
    return azimuth_spectrum


def range_doppler_fft(
    adc_cube: np.ndarray,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    num_workers: Optional[int] = None,
    range_windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
    doppler_windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
) -> np.ndarray:
    """
    Perform Range FFT (with optional windowing) followed by Doppler FFT (with optional windowing) to generate a Range-Doppler map.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Range window
    if range_windowing is not None:
        num_samples = adc_cube.shape[IdxSamples]
        if range_windowing == 'Hamming':
            window = np.hamming(num_samples)
        elif range_windowing == 'Blackman':
            window = np.blackman(num_samples)
        elif range_windowing == 'Hann':
            window = np.hanning(num_samples)
        else:
            raise ValueError(f"Unsupported window type: {range_windowing}")
        shape = [1, 1, 1]
        shape[IdxSamples] = num_samples
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    range_spectrum = fft(adc_cube, axis=IdxSamples, workers=num_workers)

    # Doppler window
    if doppler_windowing is not None:
        num_chirps = range_spectrum.shape[IdxChirps]
        if doppler_windowing == 'Hamming':
            window = np.hamming(num_chirps)
        elif doppler_windowing == 'Blackman':
            window = np.blackman(num_chirps)
        elif doppler_windowing == 'Hann':
            window = np.hanning(num_chirps)
        else:
            raise ValueError(f"Unsupported window type: {doppler_windowing}")
        shape = [1, 1, 1]
        shape[IdxChirps] = num_chirps
        window = window.reshape(shape)
        range_spectrum = range_spectrum * window

    range_doppler_spectrum = fftshift(
        fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return range_doppler_spectrum


def range_doppler_azimuth_fft(
    adc_cube: np.ndarray,
    IdxVirtualAntennas: int = 0,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    numAngleBins: int = 180,
    num_workers: Optional[int] = None,
    range_windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
    doppler_windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
    azimuth_windowing: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
) -> np.ndarray:
    """
    Perform a series of FFTs (Range FFT, Doppler FFT, and Azimuth FFT) to transform ADC data
    into a Range-Doppler-Azimuth representation, with optional windowing for each dimension.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Calculate padding for the virtual antennas (azimuth) dimension
    azimuth_padding = [(0, 0), (0, 0), (0, 0)]
    azimuth_padding[IdxVirtualAntennas] = (
        0,
        numAngleBins - adc_cube.shape[IdxVirtualAntennas],
    )
    azimuth_padding = tuple(azimuth_padding)

    padded_adc_cube = np.pad(adc_cube, pad_width=azimuth_padding, mode='constant')

    # Range window
    if range_windowing is not None:
        num_samples = padded_adc_cube.shape[IdxSamples]
        if range_windowing == 'Hamming':
            window = np.hamming(num_samples)
        elif range_windowing == 'Blackman':
            window = np.blackman(num_samples)
        elif range_windowing == 'Hann':
            window = np.hanning(num_samples)
        else:
            raise ValueError(f"Unsupported window type: {range_windowing}")
        shape = [1, 1, 1]
        shape[IdxSamples] = num_samples
        window = window.reshape(shape)
        padded_adc_cube = padded_adc_cube * window

    range_spectrum = fft(padded_adc_cube, axis=IdxSamples, workers=num_workers)

    # Doppler window
    if doppler_windowing is not None:
        num_chirps = range_spectrum.shape[IdxChirps]
        if doppler_windowing == 'Hamming':
            window = np.hamming(num_chirps)
        elif doppler_windowing == 'Blackman':
            window = np.blackman(num_chirps)
        elif doppler_windowing == 'Hann':
            window = np.hanning(num_chirps)
        else:
            raise ValueError(f"Unsupported window type: {doppler_windowing}")
        shape = [1, 1, 1]
        shape[IdxChirps] = num_chirps
        window = window.reshape(shape)
        range_spectrum = range_spectrum * window

    range_doppler_spectrum = fftshift(
        fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )

    # Azimuth window
    if azimuth_windowing is not None:
        num_ant = range_doppler_spectrum.shape[IdxVirtualAntennas]
        if azimuth_windowing == 'Hamming':
            window = np.hamming(num_ant)
        elif azimuth_windowing == 'Blackman':
            window = np.blackman(num_ant)
        elif azimuth_windowing == 'Hann':
            window = np.hanning(num_ant)
        else:
            raise ValueError(f"Unsupported window type: {azimuth_windowing}")
        shape = [1, 1, 1]
        shape[IdxVirtualAntennas] = num_ant
        window = window.reshape(shape)
        range_doppler_spectrum = range_doppler_spectrum * window

    range_doppler_azimuth_spectrum = fftshift(
        fft(range_doppler_spectrum, axis=IdxVirtualAntennas, workers=num_workers),
        axes=IdxVirtualAntennas,
    )

    return range_doppler_azimuth_spectrum
