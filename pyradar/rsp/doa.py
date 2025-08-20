#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : doa.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
from scipy.ndimage import convolve
import numpy as np
from scipy.fft import fft, fftshift
from typing import Optional, Union
from ..utils import *
from pyradar.base.transceivers import Transceivers

# ######################################################################
# DoA Functions
# ######################################################################

def compute_steering_vector(
    transceivers: Transceivers,
    longitude: Optional[Union[float, np.ndarray, list]] = None,
    latitude: Optional[Union[float, np.ndarray, list]] = None,
    azimuth: Optional[Union[float, np.ndarray, list]] = None,
    elevation: Optional[Union[float, np.ndarray, list]] = None
) -> np.ndarray:

    if longitude is not None and latitude is not None:
        longitude, latitude = np.meshgrid(longitude, latitude)
        longitude = longitude.reshape(1, -1)
        latitude = latitude.reshape(1, -1)

        
        unit_vector = np.array([
            np.sin(longitude) * np.cos(latitude),
            np.sin(longitude) * np.sin(latitude),
            np.cos(longitude)
        ])
    
        return np.exp(
            -1j * np.pi * transceivers.virtualAntennaArray @ unit_vector
        )

    elif azimuth is not None and elevation is not None:

        azimuth, elevation = np.meshgrid(azimuth, elevation)
        azimuth = azimuth.flatten()
        elevation = elevation.flatten()

        unit_vector = np.array([
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
            np.cos(azimuth) * np.cos(elevation)
        ])

        return np.exp(
            -1j * np.pi * transceivers.virtualAntennaArray @ unit_vector
        )

    else:
        raise ValueError("At least one of longitude/latitude or azimuth/elevation must be provided.")



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
        A 2D numpy array with dimensions (numVirtualAntennas, numSamples),
        where:
        - `numSamples` is the total number of signal samples.
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
        raise ValueError(
            'Input signal must be a 2D array (numVirtualAntennas x numSamplesPerChirp).'
        )

    # Compute spatial covariance matrix
    numVirtualAntennas, _ = signal.shape

    Rxx = np.cov(signal)

    if fb_avg:
        # Perform forward-backward averaging
        # Create exchange matrix
        J = np.fliplr(
            np.eye(numVirtualAntennas)
        )  # Flip identity matrix to form exchange matrix
        # Compute forward-backward averaged covariance matrix
        Rxx = 0.5 * (Rxx + J @ np.conjugate(Rxx) @ J)

    return Rxx


# ######################################################################
# Beamforming
# ######################################################################


def doa_bartlett(
    signal: np.ndarray, steering_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(
            "The input 'steering_vector' must be a 2D array with shape (numAngleBins, numVirtualAntennas)."
        )

    numVirtualAntennas, numAngleBins = steering_vector.shape
    if signal.shape[0] != numVirtualAntennas:
        raise ValueError(
            "The number of antennas in 'signal' and 'steering_vector' must match."
        )

    weight = steering_vector / numVirtualAntennas

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=False)

    # Compute Bartlett Power Spectrum
    # Option 1: Power = np.sum((steering_vector.T.conj() @ Rxx_inv) * steering_vector, axis=0)
    # Option 2: Power = np.abs(steering_vector.T.conj() @ signal)**2
    power = np.einsum('ij,ij->i', steering_vector.T.conj(), (Rxx @ steering_vector).T)

    power_spectrum = np.abs(power)

    return power_spectrum, weight


def doa_capon(
    signal: np.ndarray, steering_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Direction of Arrival (DoA) using the Bartlett method.

    Parameters:
        signal (np.ndarray): The received signal matrix with dimensions (numVirtualAntennas, numSamplesPerChirp).
        steering_vector (np.ndarray): The steering vector matrix with dimensions
                                       (numAngleBins, numVirtualAntennas).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - power_spectrum (np.ndarray): The Bartlett power spectrum with shape (numAngleBins,numSamplesPerChirp),
                                           representing the signal power for each angle bin.
            - weight (np.ndarray): The normalized steering weights used for beamforming,
                                   with shape (numVirtualAntennas, numAngleBins).

    Raises:
        ValueError: If input dimensions are invalid or mismatched.
    """
    # Validate signal dimensions
    if signal.ndim != 2:
        signal = signal.reshape((-1, 1))
        # raise ValueError("The input 'signal' must be a 2D array with shape (numVirtualAntennas, numSamplesPerChirp).")

    # Validate steering vector dimensions
    if steering_vector.ndim != 2:
        raise ValueError(
            "The input 'steering_vector' must be a 2D array with shape (numAngleBins, numVirtualAntennas)."
        )

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

    try:
        Rxx_inv = np.linalg.pinv(Rxx)
    except np.linalg.LinAlgError:
        # Raise an error if the covariance matrix is singular or not invertible
        raise ValueError('Covariance matrix is singular or not invertible.')

    first = Rxx_inv @ steering_vector
    power = np.reciprocal(np.einsum('ij,ij->i', steering_vector.T.conj(), first.T))
    weight = np.matmul(first, power)

    # response = watt2db(np.abs(power_spectrum))
    power_spectrum = np.abs(power)

    return power_spectrum, weight


# ######################################################################
# MUSIC
# ######################################################################
def doa_music(
    signal: np.ndarray, steering_vector: np.ndarray, num_targets: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Direction of Arrival (DoA) using the MUSIC algorithm.

    Parameters:
        signal (np.ndarray): The received signal matrix with dimensions (numVirtualAntennas, numSamplesPerChirp).
        steering_vector (np.ndarray): The steering vector matrix with dimensions
                                      (numAngleBins, numVirtualAntennas).
        num_targets (int): Number of expected targets (default: 1).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - power_spectrum (np.ndarray): The MUSIC power spectrum with shape (numAngleBins,),
                                           representing the signal power for each angle bin.
            - noise_subspace (np.ndarray): The noise subspace eigenvectors.

    Raises:
        ValueError: If input dimensions are invalid or mismatched.
    """
    # Validate signal dimensions
    if signal.ndim != 2:
        signal = signal.reshape((-1, 1))

    if steering_vector.ndim != 2:
        raise ValueError(
            "The input 'steering_vector' must be a 2D array with shape (numAngleBins, numVirtualAntennas)."
        )

    numVirtualAntennas = signal.shape[0]
    numAngleBins = steering_vector.shape[0]
    if steering_vector.shape[1] != numVirtualAntennas:
        raise ValueError(
            "The number of antennas in 'signal' and 'steering_vector' must match."
        )

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

    # Eigen decomposition
    eigval, eigvec = np.linalg.eigh(Rxx)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    # Signal and noise subspaces
    signal_subspace = eigvec[:, :num_targets]
    noise_subspace = eigvec[:, num_targets:]

    # MUSIC spectrum calculation
    power_spectrum = np.zeros(numAngleBins, dtype=np.float64)
    for i in range(numAngleBins):
        sv = steering_vector[i, :].reshape(-1, 1)
        denom = np.linalg.norm(noise_subspace.conj().T @ sv) ** 2
        power_spectrum[i] = 1.0 / denom if denom > 0 else 0.0

    return power_spectrum, noise_subspace

# def doa_esprit(
#     signal: np.ndarray, order: int, num_targets: int = 1
# ) -> np.ndarray:
#     """
#     Estimate Direction of Arrival (DoA) using the ESPRIT algorithm.

#     Parameters:
#         signal (np.ndarray): The received signal matrix with dimensions (numVirtualAntennas, numSamplesPerChirp).
#         order (int): Subarray order (number of rows in each subarray, typically numVirtualAntennas).
#         num_targets (int): Number of sources/targets to estimate.

#     Returns:
#         np.ndarray: Estimated normalized angular frequencies (DoA roots).

#     Raises:
#         ValueError: If input dimensions are invalid or mismatched.
#     """
#     # Validate signal dimensions
#     if signal.ndim != 2:
#         signal = signal.reshape((-1, 1))

#     numVirtualAntennas, numSamples = signal.shape
#     if order > numVirtualAntennas or order < 2:
#         raise ValueError("Order must be between 2 and numVirtualAntennas.")

#     # Compute spatial covariance matrix
#     Rxx = compute_spatial_covariance(signal, fb_avg=True)

#     # Eigen decomposition
#     eigval, eigvec = np.linalg.eigh(Rxx)
#     idx = np.argsort(eigval)[::-1]
#     eigvec = eigvec[:, idx]

#     # Signal subspace
#     signal_subspace = eigvec[:, :num_targets]

#     # Form subarrays
#     s1 = signal_subspace[0:order-1, :]
#     s2 = signal_subspace[1:order, :]

#     # Solve for rotational invariance
#     # s1 * Psi ≈ s2
#     Psi, residuals, rank, s = np.linalg.lstsq(s1, s2, rcond=None)
#     roots, _ = np.linalg.eig(Psi)

#     return roots