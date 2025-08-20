#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : transceiver.py
# @IDE     : vscode

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Union
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        self.dTX = self.TX - np.min(self.TX, axis=0)
        self.dRX = self.RX - np.min(self.RX, axis=0)

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

    def mask(self, maskTX: np.ndarray, maskRX: np.ndarray) -> None:
        """Apply a mask to isolate specific antennas."""
        try:
            self.maskTX = maskTX
            self.maskRX = maskRX
            self.__init__(self.TX[maskTX], self.RX[maskRX])
        except Exception as e:
            print(f"Error applying mask: {e}")

    @property
    def virtualAntennaArray(self) -> np.ndarray:
        """Virtual antenna array positions."""
        TX_X = np.array(self.TX[:, 0] - np.min(self.TX[:, 0])).reshape(-1, 1)
        TX_Y = np.array(self.TX[:, 1] - np.min(self.TX[:, 1])).reshape(-1, 1)
        RX_X = np.array(self.RX[:, 0] - np.min(self.RX[:, 0])).reshape(1, -1)
        RX_Y = np.array(self.RX[:, 1] - np.min(self.RX[:, 1])).reshape(1, -1)

        virtualArray_X = (TX_X + RX_X).flatten()
        virtualArray_Y = (TX_Y + RX_Y).flatten()

        return np.column_stack((virtualArray_X, virtualArray_Y, np.zeros(self.numVirtualAntennas)))

    def show(self) -> None:
        """Visualize the antenna positions."""

        plt.scatter(self.TX[:, 0], self.TX[:, 1], label='TX Antennas', color='blue', marker='^', s=80)
        plt.scatter(self.RX[:, 0], self.RX[:, 1], label='RX Antennas', color='red', marker='o', s=80)
        plt.scatter(self.virtualAntennaArray[:, 0], self.virtualAntennaArray[:, 1], label='Virtual Antenna Array', color='green', marker='x', s=80)

        plt.xlabel('Azimuth (half-wavelength)')
        plt.ylabel('Elevation (half-wavelength)')
        plt.title('Antenna Layout')
        plt.legend()
        plt.grid()
        plt.show()

    def show3D(self) -> None:
        """Visualize the antenna positions in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot TX antennas
        ax.scatter(self.TX[:, 0], self.TX[:, 1], self.TX[:, 2], 
                  label='TX Antennas', color='blue', marker='^', s=120)
        
        # Plot RX antennas
        ax.scatter(self.RX[:, 0], self.RX[:, 1], self.RX[:, 2], 
                  label='RX Antennas', color='red', marker='o', s=100)
        
        # Plot virtual antenna array
        ax.scatter(self.virtualAntennaArray[:, 0], self.virtualAntennaArray[:, 1], self.virtualAntennaArray[:, 2], 
                  label='Virtual Antenna Array', color='green', marker='x', s=80)
        
        # Add coordinate frame at first RX antenna
        origin = self.RX[0]  # First RX antenna position
        axis_length = 2.0    # Length of coordinate axes
        
        # X-axis (red)
        ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, 
                 color='red', arrow_length_ratio=0.1, linewidth=3, label='X-axis')
        
        # Y-axis (green) 
        ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, 
                 color='lime', arrow_length_ratio=0.1, linewidth=3, label='Y-axis')
        
        # Z-axis (blue)
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, 
                 color='cyan', arrow_length_ratio=0.1, linewidth=3, label='Z-axis')
        
        # Set labels and title
        ax.set_xlabel('X (half-wavelength)')
        ax.set_ylabel('Y (half-wavelength)')
        ax.set_zlabel('Z (half-wavelength)')
        ax.set_title('3D Antenna Layout with Coordinate Frame')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Set equal aspect ratio for better visualization
        max_range = np.array([self.virtualAntennaArray[:, 0].max() - self.virtualAntennaArray[:, 0].min(),
                             self.virtualAntennaArray[:, 1].max() - self.virtualAntennaArray[:, 1].min(),
                             self.virtualAntennaArray[:, 2].max() - self.virtualAntennaArray[:, 2].min()]).max() / 2.0
        
        mid_x = (self.virtualAntennaArray[:, 0].max() + self.virtualAntennaArray[:, 0].min()) * 0.5
        mid_y = (self.virtualAntennaArray[:, 1].max() + self.virtualAntennaArray[:, 1].min()) * 0.5
        mid_z = (self.virtualAntennaArray[:, 2].max() + self.virtualAntennaArray[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()



if __name__ == "__main__":
    # Example usage
    tx = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
    rx = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
    transceivers = Transceivers(TX=tx, RX=rx)
    
    # Show 2D visualization
    print("Displaying 2D antenna layout...")
    transceivers.show()
    
    # Show 3D visualization
    print("Displaying 3D antenna layout...")
    transceivers.show3D()

    transceivers.mask(np.array([True, False, True]), np.array([True, True, False, True]))
    transceivers.show()
    transceivers.show3D()