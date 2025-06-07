#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : transceiver.py
# @IDE     : vscode

from dataclasses import dataclass
from typing import List, Tuple, Literal

@dataclass
class Antenna:
    type: Literal['TX', 'RX']
    delta_xyz: Tuple[float]  # Antenna position in meters (x, y)

    def __init__(
            self, 
            type: Literal['TX', 'RX'], 
            delta_y: float,
            delta_z: float
            ):
        self.type = type
        self.delta_xyz = (0, delta_y, delta_z)


@dataclass
class Transceivers:
    """
    Antenna array configuration, including antenna positions and array details.
    """

    transmiters: List[Antenna] = None
    receivers: List[Antenna] = None

    @property
    def numTX(self) -> int:
        """Number of transmitters."""
        return len(self.transmiters) if self.transmiters else 0
    
    @property
    def numRX(self) -> int:
        """Number of receivers."""
        return len(self.receivers) if self.receivers else 0


