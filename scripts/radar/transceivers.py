#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : transceiver.py
# @IDE     : vscode

from dataclasses import dataclass


@dataclass
class Transceivers:
    """
    Antenna array configuration, including antenna positions and array details.
    """
    numTX: int = 2  # Number of transmit antennas
    numRX: int = 4  # Number of receive antennas
    numVirtualAntennas: int = 8  # Number of antennas in the array
