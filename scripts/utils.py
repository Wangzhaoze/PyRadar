#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : utils.py
# @IDE     : vscode

"""
Radar Signal Processing Module

This module provides functions for radar signal processing and simulation applications.
It includes computations for radar parameters such as:

- **Range resolution**: Determines the ability of the radar to distinguish two targets close in range.
- **Maximum range**: The farthest distance a radar can detect an object.
- **Velocity resolution**: The ability of the radar to distinguish two targets moving at similar speeds.
- **Maximum velocity**: The highest detectable velocity without ambiguity.
- **Azimuth resolution**: The angular resolution of the radar in the horizontal plane.
- **Maximum azimuth**: The maximum detectable azimuth angle without ambiguity.

Constants:
    SPEED_OF_LIGHT: The speed of light in a vacuum (299,792,458 m/s), used in radar calculations.

Functions:
    - `get_range_resolution`: Computes the range resolution based on the chirp bandwidth.
    - `get_max_range`: Determines the maximum range based on ADC sampling rate and chirp slope.
    - `get_velocity_resolution`: Calculates the velocity resolution using carrier frequency, number of chirps per frame, and chirp duration.
    - `get_max_velocity`: Computes the maximum detectable velocity for a given carrier frequency and chirp duration.
    - `get_azimuth_resolution`: Determines the azimuth resolution based on carrier frequency, number of virtual antennas, and antenna spacing.
    - `get_max_azimuth`: Computes the maximum azimuth angle detectable without ambiguity.

All functions use the speed of light as a default constant, ensuring accuracy in radar signal processing calculations.
"""

import numpy as np
from typing import Optional

# Constants
speedOfLight = 299792458  # Speed of light in meters/second (default constant)

def range_resolution(chirpBandwidth: float) -> float:
    """
    Calculate the range resolution of the radar.

    Parameters:
        chirpBandwidth (float): Bandwidth of the radar chirp in Hz.

    Returns:
        float: Range resolution in meters.

    Formula:
        Range resolution = c / (2 * B)
        Where:
            c = speed of light (299,792,458 m/s)
            B = Chirp bandwidth in Hz
    """
    return speedOfLight / (2 * chirpBandwidth)

def range_maximum(adcSampleRate: float, chirpSlope: float) -> float:
    """
    Calculate the maximum detectable range of the radar.

    Parameters:
        adcSampleRate (float): Analog-to-Digital Converter (ADC) sampling rate in Hz.
        chirpSlope (float): Chirp slope in Hz/s.

    Returns:
        float: Maximum range in meters.

    Formula:
        Maximum range = (c * Fs) / (2 * S)
        Where:
            c = speed of light (299,792,458 m/s)
            Fs = ADC sample rate (samples per second)
            S = Chirp slope (Hz/s)
    """
    return speedOfLight * adcSampleRate / (2 * chirpSlope)

def range_bins(numSamplesPerChirp: int,
               adcSampleRate: float,
               chirpSlope: float) -> np.ndarray:
    """
    Calculate the range bins for the radar based on maximum range.

    Parameters:
        numSamplesPerChirp (int): Number of samples per chirp.
        adcSampleRate (float): ADC sampling rate in Hz.
        chirpSlope (float): Chirp slope in Hz/s.

    Returns:
        np.ndarray: Array of range bins in meters.

    Formula:
        Range axis = linspace(0, Maximum range, numSamplesPerChirp)
    """
    # Calculate maximum range
    range_max = range_maximum(adcSampleRate, chirpSlope)

    return np.linspace(0, range_max, numSamplesPerChirp)

def velocity_resolution(carrierFrequency: float, numChirpsPerFrame: int, chirpTime: float) -> float:
    """
    Calculate the velocity resolution of the radar.

    Parameters:
        carrierFrequency (float): Carrier frequency in Hz.
        numChirpsPerFrame (int): Number of chirps in one frame.
        chirpTime (float): Duration of one chirp in seconds.

    Returns:
        float: Velocity resolution in m/s.

    Formula:
        Velocity resolution = λ / (2 * N * T)
        Where:
            λ = wavelength = c / fc (speed of light / carrier frequency)
            N = number of chirps per frame
            T = chirp time in seconds
    """
    waveLength = speedOfLight / carrierFrequency
    return waveLength / (2 * numChirpsPerFrame * chirpTime)

def velocity_maximum(carrierFrequency: float, chirpTime: float) -> float:
    """
    Calculate the maximum detectable velocity of the radar.

    Parameters:
        carrierFrequency (float): Carrier frequency in Hz.
        chirpTime (float): Duration of one chirp in seconds.

    Returns:
        float: Maximum velocity in m/s.

    Formula:
        Maximum velocity = λ / (4 * T)
        Where:
            λ = wavelength = c / fc (speed of light / carrier frequency)
            T = chirp time in seconds
    """
    waveLength = speedOfLight / carrierFrequency
    return waveLength / (4 * chirpTime)

def velocity_bins(numChirpsPerFrame: int,
                  carrierFrequency: float,
                  chirpTime: float) -> np.ndarray:
    """
    Calculate the velocity bins for the radar.

    Parameters:
        numChirpsPerFrame (int): Number of chirps in one frame.
        carrierFrequency (float, optional): Carrier frequency in Hz.
        chirpTime (float, optional): Duration of one chirp in seconds.

    Returns:
        np.ndarray: Array of velocity bins in m/s.

    Formula:
        Velocity axis = linspace(-V_max, V_max, numChirpsPerFrame)
        Where:
            V_max = Maximum velocity
    """
    # Calculate maximum velocity
    vel_max = velocity_maximum(carrierFrequency, chirpTime)

    return np.linspace(-vel_max, vel_max, numChirpsPerFrame)


def azimuth_resolution(carrierFrequency: float, numVirtualAntennas: int, antennaSpacing: float) -> float:
    """
    Calculate the azimuth resolution of the radar.

    Parameters:
        carrierFrequency (float): Carrier frequency in Hz.
        numVirtualAntennas (int): Number of virtual antennas in the radar array.
        antennaSpacing (float): Spacing between antennas in meters.

    Returns:
        float: Azimuth resolution in radians.

    Formula:
        Azimuth resolution = λ / (N * d)
        Where:
            λ = wavelength = c / fc (speed of light / carrier frequency)
            N = number of virtual antennas
            d = antenna spacing in meters
    """
    waveLength = speedOfLight / carrierFrequency
    return waveLength / (numVirtualAntennas * antennaSpacing)

def azimuth_maximum(carrierFrequency: float, antennaSpacing: float) -> float:
    """
    Calculate the maximum detectable azimuth angle of the radar.

    Parameters:
        carrierFrequency (float): Carrier frequency in Hz.
        antennaSpacing (float): Spacing between antennas in meters.

    Returns:
        float: Maximum azimuth angle in radians.

    Formula:
        Maximum azimuth angle = arcsin(λ / (2 * d))
        Where:
            λ = wavelength = c / fc (speed of light / carrier frequency)
            d = antenna spacing in meters
    """
    waveLength = speedOfLight / carrierFrequency
    return np.arcsin(waveLength / (2 * antennaSpacing))

def azimuth_bins(numAzimuthBins: int,
                 carrierFrequency: float,
                 antennaSpacing: float) -> np.ndarray:
    """
    Calculate the azimuth bins for the radar.

    Parameters:
        numAzimuthBins (int): Number of samples for azimuth.
        carrierFrequency (float, optional): Carrier frequency in Hz.
        antennaSpacing (float, optional): Spacing between antennas in meters.

    Returns:
        np.ndarray: Array of azimuth bins in Degrees.

    Formula:
        Azimuth axis = linspace(-Azimuth_max, Azimuth_max, numAzimuthBins)
        Where:
            Azimuth_max = Maximum azimuth angle
    """
    # Calculate maximum azimuth angle
    azimuth_max = azimuth_maximum(carrierFrequency, antennaSpacing)

    azimuth_max = np.rad2deg(azimuth_max)

    # Generate azimuth bins from -azimuth_max to azimuth_max
    return np.linspace(-azimuth_max, azimuth_max, numAzimuthBins)


# from radar_sensor_configs import *
# def range_FFT_freq():
#     return np.arange(0, numSamplesPerChirp)*(adcSampleRate)/numSamplesPerChirp

# def freq2range():
#     return f  * speedOfLight/(2*chirpSlope)

# def doppler_FFT_freq():
#     return

# #doppler bins to frequencies
# f = fftshift(fftfreq(255, chirpTime))

# omega = 2 * np.pi * f

# # doppler frequencies to velocity
# omega * speedOfLight / (4 * np.pi * carrierFrequency)
