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


