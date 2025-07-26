#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : transceiver.py
# @IDE     : vscode

from dataclasses import dataclass
from typing import List, Tuple, Literal
from matplotlib import pyplot as plt
import numpy as np

@dataclass
class Transceivers:
    """
    Antenna array configuration, including antenna positions and array details.
    """
    def __init__(
            self, 
            TX: List[List[float]] = None,
            RX: List[List[float]] = None
    ):
        self.TX = np.array(TX)
        self.RX = np.array(RX)

    @property
    def numTX(self) -> int:
        """Number of transmitters."""
        return len(self.TX)
    
    @property
    def numRX(self) -> int:
        """Number of RX."""
        return len(self.RX)
    
    @property
    def numVirtualAntennas(self) -> int:
        """Number of virtual antennas."""
        return self.numTX * self.numRX
    
    @property
    def antennaGain(self) -> float:
        """Antenna gain in dB."""
        return NotImplemented


    @property
    def virtualAntennaArray(self) -> np.ndarray:
        """Virtual antenna array positions."""
        TX_X = np.array(self.TX[:, 0] - np.min(self.TX[:, 0]))
        TX_Y = np.array(self.TX[:, 1] - np.min(self.TX[:, 1]))
        RX_X = np.array(self.RX[:, 0])
        RX_Y = np.array(self.RX[:, 1])

        # va_x = np.kron(TX_X - np.min(TX_X), np.ones(len(RX_X))).reshape(-1, len(RX_X)) + RX_X
        # va_y = np.kron(TX_Y - np.min(TX_Y), np.ones(len(RX_Y))).reshape(-1, len(RX_Y)) + RX_Y

        va_x = TX_X.reshape(-1, 1) + RX_X.reshape(1, -1)
        va_y = TX_Y.reshape(-1, 1) + RX_Y.reshape(1, -1)

        return np.column_stack((va_x.flatten(), va_y.flatten(), np.zeros(self.numVirtualAntennas)))

    def show(self) -> None:
        """Visualize the antenna positions."""

        plt.scatter(self.TX[:, 0], self.TX[:, 1], label='TX Antennas', color='blue', marker='^')
        plt.scatter(self.RX[:, 0], self.RX[:, 1], label='RX Antennas', color='red', marker='o')
        plt.scatter(self.virtualAntennaArray[:, 0], self.virtualAntennaArray[:, 1], label='Virtual Antenna Array', color='green', marker='x')

        plt.xlabel('Azimuth (half-wavelength)')
        plt.ylabel('Elevation (half-wavelength)')
        plt.title('Antenna Layout')
        plt.legend()
        plt.grid()
        plt.show()

    def show3D(self):
        return NotImplemented
    
    def steeringMatrix(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        Calculate the steering matrix for the antenna array based on azimuth and elevation angles.
        
        Args:
            azimuth (float): Azimuth angle in radians.
            elevation (float): Elevation angle in radians.
        
        Returns:
            np.ndarray: Steering matrix for the antenna array.
        """
        # Placeholder for steering matrix calculation
        # r_az_el = np.sqrt(laz**2 + lel**2)
        # phi_az_el = np.arctan2(lel, laz)
        # smat = np.exp(
        #     -1j * np.pi * r_az_el * np.cos(alpha - phi_az_el) * np.cos(beta)
        # )
        return NotImplemented
    

if __name__ == "__main__":
    # Example usage
    tx = [[6, 0, 0], [8, 1, 0], [10, 0, 0]]
    rx = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    transceivers = Transceivers(TX=tx, RX=rx)
    transceivers.show()