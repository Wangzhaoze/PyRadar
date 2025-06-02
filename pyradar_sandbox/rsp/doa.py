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


# ######################################################################
# DoA Functions
# ######################################################################


def compute_steering_vector(
    numVirtualAntennas: int, angles: Union[float, np.ndarray, list]
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

    steering_vector = np.zeros(
        (numVirtualAntennas, angles.shape[0]), dtype=np.complex64
    )

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
        raise ValueError(
            'Input signal must be a 2D array (numVirtualAntennas x numSamplesPerChirp).'
        )

    # Compute spatial covariance matrix
    numVirtualAntennas, numSamplesPerChirp = signal.shape
    # np.einsum('ij,ik->ijk', x, x)
    Rxx = signal @ signal.T.conj()
    Rxx = np.divide(Rxx, numSamplesPerChirp)

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
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

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

    # Compute spatial covariance matrix
    Rxx = compute_spatial_covariance(signal, fb_avg=True)

    try:
        Rxx_inv = np.linalg.inv(Rxx)
    except np.linalg.LinAlgError:
        # Raise an error if the covariance matrix is singular or not invertible
        raise ValueError('Covariance matrix is singular or not invertible.')

    # Compute Capon Power Spectrum
    power = np.reciprocal(
        np.einsum('ij,ij->i', steering_vector.T.conj(), (Rxx_inv @ steering_vector).T)
    )

    weight = np.matmul((Rxx_inv @ steering_vector), power)

    # response = watt2db(np.abs(power_spectrum))
    power_spectrum = np.abs(power)

    return power_spectrum, weight


# ######################################################################
# MUSIC
# ######################################################################
