#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : vis.py
# @IDE     : vscode

"""Visualization Tools of Radar Signal."""

import matplotlib.pyplot as plt
from typing import Optional, Union
import open3d as o3d

import numpy as np
from rsp.fft import range_doppler_fft, range_doppler_azimuth_fft


def show_2D_heat_map(
    spectrum: np.ndarray,
    figsize: tuple = (8, 6),
    xaxis: Union[np.ndarray, list] = None,
    yaxis: Union[np.ndarray, list] = None,
    cmap: str = 'jet',
    xlabel: str = 'X-axis',
    ylabel: str = 'Y-axis',
    title: str = 'Heat Map',
    colorbar_label: str = 'Magnitude',
):
    """
    Display a heat map of a given spectrum.

    Parameters:
    - spectrum (np.ndarray): 2D numpy array of complex values representing the spectrum.
    - xaxis (np.ndarray, optional): Values for the x-axis. Defaults to index-based values.
    - yaxis (np.ndarray, optional): Values for the y-axis. Defaults to index-based values.
    - cmap (str, optional): Colormap for the heat map. Default is 'jet'.
    - xlabel (str, optional): Label for the x-axis. Default is 'X-axis'.
    - ylabel (str, optional): Label for the y-axis. Default is 'Y-axis'.
    - title (str, optional): Title of the plot. Default is 'Heat Map'.
    - colorbar_label (str, optional): Label for the colorbar. Default is 'Magnitude'.

    Raises:
    - ValueError: If input spectrum is not a 2D numpy array of complex values.
    """
    # Validate inputs
    if not isinstance(spectrum, np.ndarray):
        raise ValueError('Input spectrum must be a numpy array.')
    # if np.issubdtype(spectrum.dtype, np.complexfloating) or isinstance(spectrum, complex):
    #     raise ValueError(f'Input spectrum must contain complex numbers, but it is {spectrum.dtype}.')
    if spectrum.ndim != 2:
        raise ValueError('Input spectrum must be a 2D array.')

    # Set x and y axis extents
    xaxis = xaxis if xaxis is not None else np.arange(spectrum.shape[1])
    yaxis = yaxis if yaxis is not None else np.arange(spectrum.shape[0])

    # Plot the heat map
    plt.figure(figsize=figsize)
    plt.imshow(
        np.abs(spectrum),
        # 20.0 * np.log10( np.abs(spectrum) ),
        cmap=cmap,
        extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
        aspect='auto',
    )
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def range_doppler_map(adc_cube: np.ndarray,
                      IdxSamples: int = 0,
                      IdxChirps: int = 1,
                      IdxVirtualAntennas: int = 2):
    """
    Computes the Range-Doppler Map from the given ADC data cube by performing a Range-Doppler FFT.

    The function calculates the Range-Doppler Map from the 3D ADC data cube, which typically represents
    samples, chirps, and virtual antennas. The map shows the range on the horizontal axis (0 to Rmax)
    and velocity on the vertical axis (from -Vmin to Vmax).

    Parameters:
    - adc_cube (np.ndarray): 3D numpy array containing the ADC data cube. The dimensions represent samples, chirps, and virtual antennas.
    - IdxSamples (int): Index of the samples dimension in the `adc_cube`. Default is 0.
    - IdxChirps (int): Index of the chirps dimension in the `adc_cube`. Default is 1.
    - IdxVirtualAntennas (int): Index of the virtual antennas dimension in the `adc_cube`. Default is 2.

    Returns:
    - range_doppler_map (np.ndarray): 2D Range-Doppler map where:
      - The x-axis represents range from 0 to Rmax.
      - The y-axis represents velocity from -Vmin to Vmax.

      Visual representation:

                        #################################################
                        #               Range-Doppler Map               #
                        #################################################
                        #                                               #
                        #   Velocity                                    #
                        #                                               #
                        #   +Vmax    ▲                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #   0 Hz     |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #   -Vmin     ————————————————————————▷         #
                        #                                               #
                        #           0m         Range           Rmax     #
                        #################################################
    """
    # Reorder the input `adc_cube` dimensions so chirps, samples, and virtual antennas are in (0, 1, 2) order.
    adc_cube = np.transpose(adc_cube, (IdxChirps, IdxSamples, IdxVirtualAntennas))

    # Perform Range-Doppler FFT on the ADC cube. Compute the spectrum along samples and chirps dimensions.
    range_doppler_spectrum = range_doppler_fft(adc_cube, IdxSamples=1, IdxChirps=0)

    # Extract the first virtual antenna slice (index 0) along the virtual antenna axis.
    range_doppler_map = range_doppler_spectrum.take(indices=0, axis=2)

    # Flip the velocity axis (axis 0) so the maximum velocity appears at the top.
    range_doppler_map = np.flip(range_doppler_map, axis=0)

    return range_doppler_map

def range_azimuth_map(adc_cube: np.ndarray,
                      IdxSamples: int = 0,
                      IdxChirps: int = 1,
                      IdxVirtualAntennas: int = 2):
    """
    Computes the Range-Azimuth Map from the given ADC data cube.

    This function performs a Range-Azimuth FFT on the ADC data cube and calculates the Range-Azimuth Map
    by summing the Doppler information across the Doppler axis. The map displays the range along the
    vertical axis and azimuth along the horizontal axis.

    Parameters:
    - adc_cube (np.ndarray): 3D numpy array containing the ADC data cube. The dimensions represent samples, chirps, and virtual antennas.
    - IdxSamples (int): Index of the samples dimension in the `adc_cube`. Default is 0.
    - IdxChirps (int): Index of the chirps dimension in the `adc_cube`. Default is 1.
    - IdxVirtualAntennas (int): Index of the virtual antennas dimension in the `adc_cube`. Default is 2.

    Returns:
    - range_azimuth_map (np.ndarray): 2D Range-Azimuth map where:
      - The x-axis represents range from 0 to Rmax.
      - The y-axis represents azimuth angle from -Amax to +Amax.

      Visual representation:

                        #################################################
                        #               Range-Azimuth Map               #
                        #################################################
                        #                                               #
                        #   Range                                       #
                        #                                               #
                        #   +Rmax    ▲                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #   0        ————————————————————————▷          #
                        #                                               #
                        #          -Amax      Azimuth       +Amax       #
                        #################################################
    """
    # Reorder the input `adc_cube` dimensions so chirps, samples, and virtual antennas are in (0, 1, 2) order.
    adc_cube = np.transpose(adc_cube, (IdxChirps, IdxSamples, IdxVirtualAntennas))

    # Perform Range-Doppler-Azimuth FFT on the ADC cube. Compute the spectrum along samples, chirps, and antennas.
    range_doppler_azimuth_spectrum = range_doppler_azimuth_fft(
        adc_cube, IdxSamples=1, IdxChirps=0, IdxVirtualAntennas=2
    )

    # Flip the azimuth axis (axis 1) and then sum the data along the Doppler axis (axis 0).
    range_azimuth_map = np.flip(range_doppler_azimuth_spectrum, axis=1).sum(axis=0)

    return range_azimuth_map
