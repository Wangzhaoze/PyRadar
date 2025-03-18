#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : rsp.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
import numpy as np
from scipy.fft import fft, fftshift
from typing import Optional, Union
from .utils import *

# ######################################################################
# FFT Functions
# ######################################################################

def range_fft(adc_cube: np.ndarray, IdxSamples: int = 1, num_workers: Optional[int] = None) -> np.ndarray:
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

def doppler_fft(adc_cube: np.ndarray, IdxChirps: int = 2, num_workers: Optional[int] = None) -> np.ndarray:
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

    # Perform FFT along the Doppler axis and shift the zero frequency component to the center
    doppler_spectrum = fftshift(fft(adc_cube, axis=IdxChirps, workers=num_workers), axes=IdxChirps)
    return doppler_spectrum

def angle_fft(adc_cube: np.ndarray, IdxVirtualAntennas: int = 0, num_workers: Optional[int] = None) -> np.ndarray:
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

def range_doppler_fft(adc_cube: np.ndarray, IdxSamples: int = 1, IdxChirps: int = 2, num_workers: Optional[int] = None) -> np.ndarray:
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

    # Perform Doppler FFT along the Doppler axis and shift the zero frequency component to the center
    range_doppler_spectrum = fftshift(fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps)
    return range_doppler_spectrum

def range_doppler_azimuth_fft(adc_cube: np.ndarray,
                              IdxVirtualAntennas: int = 0,
                              IdxSamples: int = 1,
                              IdxChirps: int = 2,
                              numAngleBins: int = 180,
                              num_workers: Optional[int] = None) -> np.ndarray:
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
    azimuth_padding[IdxVirtualAntennas] = (0, numAngleBins - adc_cube.shape[IdxVirtualAntennas])  # Pad the azimuth axis
    azimuth_padding = tuple(azimuth_padding)

    # Apply zero-padding to the azimuth axis
    padded_adc_cube = np.pad(adc_cube, pad_width=azimuth_padding, mode='constant')

    # Perform Range FFT along the range axis
    range_spectrum = fft(padded_adc_cube, axis=IdxSamples, workers=num_workers)

    # Perform Doppler FFT along the Doppler axis and shift the zero frequency component to the center
    range_doppler_spectrum = fftshift(fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps)

    # Perform Azimuth FFT along the virtual antennas axis and shift the zero frequency component to the center
    range_doppler_azimuth_spectrum = fftshift(fft(range_doppler_spectrum, axis=IdxVirtualAntennas, workers=num_workers), axes=IdxVirtualAntennas)

    return range_doppler_azimuth_spectrum

import numpy as np
from scipy.ndimage import convolve

# ######################################################################
# CFAR Functions
# ######################################################################

def cfar_2d(rdm, guard_len=4, noise_len=8, mode='wrap', l_bound=20000):
    """
    Performs CFAR detection on a 2D Range-Doppler Map.

    Args:
        rdm (numpy.ndarray): 2D Range-Doppler Map (Matrix).
        guard_len (int): Number of cells adjacent to the CUT (Cell Under Test) that are ignored.
        noise_len (int): Number of cells adjacent to the guard that are considered for noise calculation.
        mode (str): Mode for edge handling ('wrap' or 'constant').
        l_bound (float or int): Lower bound threshold to add to the calculated noise floor.

    Returns:
        numpy.ndarray: Boolean mask of detected targets in the Range-Doppler Map.
    """
    
    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=rdm.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve(rdm, kernel, mode=mode)
    threshold = noise_floor + l_bound
    
    # Compare the RDM with the threshold to detect targets
    target_mask = rdm > threshold
    
    return target_mask

# ######################################################################
# DoA Functions
# ######################################################################

def compute_steering_vector(
    numVirtualAntennas: int, 
    angles: Union[float, np.ndarray, list]
) -> np.ndarray:
    """
    Compute the steering vector for a Uniform Linear Array (ULA).

    The steering vector is a matrix where each column corresponds to a virtual antenna,
    and each row represents the phase shift for a given angle of arrival (AoA). 
    It is typically used in beamforming and array signal processing.

    Parameters:
    ----------
    numVirtualAntennas : int
        The number of virtual antennas in the array. This determines the number of columns
        in the output matrix.

    angles : Union[float, np.ndarray, list]
        The angles of arrival (AoA) in degrees. Can be a single float value, a list, 
        or a numpy array. If a single value is provided, it will be converted to a 1D array.

    Returns:
    -------
    np.ndarray
        A 2D numpy array (shape: [numVirtualAntennas, numAngleBins]) where each element 
        represents the complex steering vector value for a given angle and antenna index.
        - The rows correspond to angles.
        - The columns correspond to virtual antenna indices.

    """
    angles = np.deg2rad(angles).reshape(-1)

    steering_vector = np.zeros((numVirtualAntennas, angles.shape[0]), dtype=np.complex64)

    for IdxAntenna in range(numVirtualAntennas):
        steering_vector[IdxAntenna] = np.exp(-1j * np.pi * IdxAntenna * np.sin(angles))

    return steering_vector

def compute_spatial_covariance(signal: np.ndarray, fb_avg: bool = False) -> np.ndarray:
    """
    Compute the spatial covariance matrix of a signal.

    This function calculates the spatial covariance matrix for an input signal,
    which is typically used in array signal processing to analyze the spatial
    characteristics of received signals. An optional forward-backward averaging
    step can be performed to enhance the covariance matrix's properties, 
    especially in scenarios with uniform linear arrays (ULA).

    Parameters:
    ----------
    signal : np.ndarray
        A 2D numpy array with dimensions (numVirtualAntennas, numSamplesPerChirp), 
        where:
        - `numSamplesPerChirp` is the number of signal samples per chirp.
        - `numVirtualAntennas` is the number of virtual antennas.

    fb_avg : bool, optional (default: False)
        If True, forward-backward averaging is applied to the covariance matrix.
        This can improve performance for uniform linear arrays (ULA) in scenarios 
        with spatial symmetry.

    Returns:
    -------
    np.ndarray
        A 2D numpy array (numVirtualAntennas x numVirtualAntennas) representing 
        the spatial covariance matrix. If forward-backward averaging is enabled, 
        the returned matrix will incorporate the averaging.

    Raises:
    -------
    ValueError
        If the input `signal` is not a 2D array.

    """
    # Check dimensions
    if signal.ndim != 2:
        raise ValueError("Input signal must be a 2D array (numVirtualAntennas x numSamplesPerChirp).")
    
    # Compute spatial covariance matrix
    numVirtualAntennas, numSamplesPerChirp = signal.shape
    # np.einsum('ij,ik->ijk', x, x)
    Rxx = signal @ signal.T.conj()
    Rxx = np.divide(Rxx, numSamplesPerChirp)

    if fb_avg:
        # Perform forward-backward averaging
        # Create exchange matrix
        J = np.fliplr(np.eye(numVirtualAntennas))  # Flip identity matrix to form exchange matrix
        # Compute forward-backward averaged covariance matrix
        Rxx = 0.5 * (Rxx + J @ np.conjugate(Rxx) @ J)

    return Rxx


# ######################################################################
# Beamforming 
# ######################################################################

def doa_bartlett(signal: np.ndarray, steering_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Direction of Arrival (DoA) using the Bartlett method.

    Parameters:
        signal (np.ndarray): The received signal matrix with dimensions (numVirtualAntennas, numChirpsPerFrame).
        steering_vector (np.ndarray): The steering vector matrix with dimensions 
                                       (numAngleBins, numVirtualAntennas).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - power_spectrum (np.ndarray): The Bartlett power spectrum with shape (numAngleBins,numChirpsPerFrame),
                                           representing the signal power for each angle bin.
            - weight (np.ndarray): The normalized steering weights used for beamforming,
                                   with shape (numVirtualAntennas, numAngleBins).

    Raises:
        ValueError: If input dimensions are invalid or mismatched.
    """
    # Validate signal dimensions
    if signal.ndim != 2:
        signal = signal.reshape((-1, 1))
        # raise ValueError("The input 'signal' must be a 2D array with shape (numVirtualAntennas, numChirpsPerFrame).")

    # Validate steering vector dimensions
    if steering_vector.ndim != 2:
        raise ValueError("The input 'steering_vector' must be a 2D array with shape (numAngleBins, numVirtualAntennas).")

    numVirtualAntennas, numAngleBins = steering_vector.shape
    if signal.shape[0] != numVirtualAntennas:
        raise ValueError("The number of antennas in 'signal' and 'steering_vector' must match.")
    
    weight = steering_vector / numVirtualAntennas

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

    # Compute Bartlett Power Spectrum
    # Option 1: Power = np.sum((steering_vector.T.conj() @ Rxx_inv) * steering_vector, axis=0)
    # Option 2: Power = np.abs(steering_vector.T.conj() @ signal)**2
    power = np.einsum('ij,ij->i', steering_vector.T.conj(), (Rxx @ steering_vector).T)

    power_spectrum = np.abs(power)

    return power_spectrum, weight

def doa_capon(signal: np.ndarray, steering_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Direction of Arrival (DoA) using the Bartlett method.

    Parameters:
        signal (np.ndarray): The received signal matrix with dimensions (numVirtualAntennas, numChirpsPerFrame).
        steering_vector (np.ndarray): The steering vector matrix with dimensions 
                                       (numAngleBins, numVirtualAntennas).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - power_spectrum (np.ndarray): The Bartlett power spectrum with shape (numAngleBins,numChirpsPerFrame),
                                           representing the signal power for each angle bin.
            - weight (np.ndarray): The normalized steering weights used for beamforming,
                                   with shape (numVirtualAntennas, numAngleBins).

    Raises:
        ValueError: If input dimensions are invalid or mismatched.
    """
    # Validate signal dimensions
    if signal.ndim != 2:
        signal = signal.reshape((-1, 1))
        # raise ValueError("The input 'signal' must be a 2D array with shape (numVirtualAntennas, numChirpsPerFrame).")

    # Validate steering vector dimensions
    if steering_vector.ndim != 2:
        raise ValueError("The input 'steering_vector' must be a 2D array with shape (numAngleBins, numVirtualAntennas).")

    numVirtualAntennas, numAngleBins = steering_vector.shape
    if signal.shape[0] != numVirtualAntennas:
        raise ValueError("The number of antennas in 'signal' and 'steering_vector' must match.")

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

    try:
        Rxx_inv = np.linalg.inv(Rxx)
    except np.linalg.LinAlgError:
        # Raise an error if the covariance matrix is singular or not invertible
        raise ValueError("Covariance matrix is singular or not invertible.")

    # Compute Capon Power Spectrum
    power = np.reciprocal(np.einsum('ij,ij->i', steering_vector.T.conj(), (Rxx_inv @ steering_vector).T))

    weight = np.matmul((Rxx_inv @ steering_vector), power)

    # response = watt2db(np.abs(power_spectrum))
    power_spectrum = np.abs(power)

    return power_spectrum, weight



# ######################################################################
# MUSIC
# ######################################################################
