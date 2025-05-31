#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : cfar.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
from scipy.ndimage import convolve
import numpy as np
from scipy.fft import fft, fftshift
from typing import Optional, Union
from ..utils import *


# ######################################################################
# CFAR Functions
# ######################################################################

# TODO
# 1. determine input/output format
# 2. implement 2D and 3D CFAR
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

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=rdm.dtype) / (
        2 * noise_len
    )
    kernel[noise_len : noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve(rdm, kernel, mode=mode)
    threshold = noise_floor + l_bound

    # Compare the RDM with the threshold to detect targets
    target_mask = rdm > threshold

    return target_mask
