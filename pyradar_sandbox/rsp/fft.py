#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : fft.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
from matplotlib.image import BLACKMAN
from scipy.ndimage import convolve
import numpy as np
from scipy.fft import fft, fftshift
from typing import Optional, Union
from ..utils import *

# ######################################################################
# FFT Functions
# ######################################################################

def windowing(input, window_type, axis=0):
    """Window the input based on given window type.

    Args:
        input: input numpy array to be windowed.

        window_type: enum chosen between Bartlett, Blackman, Hamming, Hanning and Kaiser.

        axis: the axis along which the windowing will be applied.
    
    Returns:

    """
    window_length = input.shape[axis]
    if window_type == 'BARTLETT':
        window = np.bartlett(window_length)
    elif window_type == 'BLACKMAN':
        window = np.blackman(window_length)
    elif window_type == 'HAMMING':
        window = np.hamming(window_length)
    elif window_type == 'HANNING':
        window = np.hanning(window_length)
    else:
        raise ValueError("The specified window is not supported!!!")

    output = input * window

    return output

def range_fft(
    adc_cube: np.ndarray, IdxSamples: int = 1, num_workers: Optional[int] = None
) -> np.ndarray:
    """
    Perform FFT along the range dimension of the ADC cube.

    Args:
        adc_cube (np.ndarray): (numAntennas, numSamples, numChirps) shape 3D array with complex ADC (Analog-to-Digital Converter) data.
        IdxSamples (int): Axis index for the range dimension (default: 1).

    Returns:
        np.ndarray: The transformed data with the range dimension in the frequency domain.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Perform FFT along the range axis
    range_spectrum = fft(adc_cube, axis=IdxSamples, workers=num_workers)
    return range_spectrum


def doppler_fft(
    adc_cube: np.ndarray, IdxChirps: int = 2, num_workers: Optional[int] = None
) -> np.ndarray:
    """
    Perform FFT along the Doppler (chirps) dimension of the ADC cube.

    Args:
        adc_cube (np.ndarray): (numSamples, numChirps, numAntennas) shape 3D array with complex ADC (Analog-to-Digital Converter) data.
        IdxChirps (int): Axis index for the Doppler dimension (default: 2).

    Returns:
        np.ndarray: The transformed data with the Doppler dimension in the frequency domain.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Perform FFT along the Doppler axis and shift the zero frequency
    # component to the center
    doppler_spectrum = fftshift(
        fft(adc_cube, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return doppler_spectrum


def angle_fft(
    adc_cube: np.ndarray, IdxVirtualAntennas: int = 0, num_workers: Optional[int] = None
) -> np.ndarray:
    """
    Perform FFT along the range dimension of the ADC cube.

    Args:
        adc_cube (np.ndarray): (numSamples, numChirps, numAntennas) shape 3D array with complex ADC (Analog-to-Digital Converter) data.
        IdxVirtualAntennas (int): Axis index for the range dimension (default: 0).

    Returns:
        np.ndarray: The transformed data with the range dimension in the frequency domain.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Perform FFT along the range axis
    azimuth_spectrum = fft(adc_cube, axis=IdxVirtualAntennas, workers=num_workers)
    azimuth_spectrum = fftshift(azimuth_spectrum, axes=IdxVirtualAntennas)
    return azimuth_spectrum


def range_doppler_fft(
    adc_cube: np.ndarray,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Perform Range FFT followed by Doppler FFT to generate a Range-Doppler map.

    Args:
        adc_cube (np.ndarray): (numSamples, numChirps, numAntennas) shape 3D array with complex ADC (Analog-to-Digital Converter) data.
        IdxSamples (int): Axis index for the range dimension (default: 1).
        IdxChirps (int): Axis index for the Doppler dimension (default: 2).

    Returns:
        np.ndarray: The Range-Doppler map with both range and Doppler dimensions transformed to the frequency domain.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Perform Range FFT along the range axis
    range_spectrum = fft(adc_cube, axis=IdxSamples, workers=num_workers)

    # Perform Doppler FFT along the Doppler axis and shift the zero frequency
    # component to the center
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
) -> np.ndarray:
    """
    Perform a series of FFTs (Range FFT, Doppler FFT, and Azimuth FFT) to transform ADC data
    into a Range-Doppler-Azimuth representation.

    This function processes 3D ADC data to extract spatial, temporal, and angular frequency information.
    The function first applies FFT along the range axis, then along the Doppler axis, and finally
    performs an azimuth FFT to resolve angular information. Padding is applied along the virtual antennas
    dimension to achieve the desired angular resolution.

    Args:
        adc_cube (np.ndarray): 3D array with shape (numSamples, numChirps, numAntennas),
        IdxVirtualAntennas (int): Axis index for the virtual antennas/azimuth dimension (default: 0).
                               representing complex ADC data from the radar sensor.
        IdxSamples (int): Axis index for the range dimension (default: 1).
        IdxChirps (int): Axis index for the Doppler dimension (default: 2).
        numAngleBins (int): Number of angular bins for azimuth  If this value is greater than
                            the size of the virtual antennas dimension, zero-padding will be applied.

    Returns:
        np.ndarray: A 3D array with Range-Doppler-Azimuth data where:
                    - The range dimension is transformed into the frequency domain,
                    - The Doppler dimension is transformed into the frequency domain with zero frequency centered,
                    - The azimuth dimension is transformed into the angular domain with zero frequency centered.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Calculate padding for the virtual antennas (azimuth) dimension
    azimuth_padding = [(0, 0), (0, 0), (0, 0)]  # Default no padding
    azimuth_padding[IdxVirtualAntennas] = (
        0,
        numAngleBins - adc_cube.shape[IdxVirtualAntennas],
    )  # Pad the azimuth axis
    azimuth_padding = tuple(azimuth_padding)

    # Apply zero-padding to the azimuth axis
    padded_adc_cube = np.pad(adc_cube, pad_width=azimuth_padding, mode='constant')

    # Perform Range FFT along the range axis
    range_spectrum = fft(padded_adc_cube, axis=IdxSamples, workers=num_workers)

    # Perform Doppler FFT along the Doppler axis and shift the zero frequency
    # component to the center
    range_doppler_spectrum = fftshift(
        fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )

    # Perform Azimuth FFT along the virtual antennas axis and shift the zero
    # frequency component to the center
    range_doppler_azimuth_spectrum = fftshift(
        fft(range_doppler_spectrum, axis=IdxVirtualAntennas, workers=num_workers),
        axes=IdxVirtualAntennas,
    )

    return range_doppler_azimuth_spectrum
