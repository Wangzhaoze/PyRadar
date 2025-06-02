
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import glob
from functools import cached_property
import math

import matplotlib.pyplot as plt

import struct
from typing import Tuple
import open3d as o3d


@dataclass
class ColoradarTI1843BoostConfig:
    numTX: int = 3
    numRX: int = 4
    numChirpsPerFrame: int = 128
    numADCSamples: int = 128
    numVirtualAntennas: int = 12
    adcSampleRate: int = 10666000
    startFrequency: float = 76999999488.0
    idleTime: float = 0.000110000000859
    adcStartTime: float = 7.00000009601e-06
    rampDuration: float = 1.99999994948e-05
    chirpSlope: float = 1.00000000377e+14
    
    @cached_property
    def adc_scale_factor(self) -> float:
        return 0.01

    @cached_property
    def waveLength(self) -> float:
        return 3e8 / self.startFrequency
    
    @cached_property
    def sampleTimeArray(self) -> np.ndarray:
        return np.arange(0, self.numADCSamples) * 1 / self.adcSampleRate

    @cached_property
    def frameShape(self) -> tuple:
        return (self.numTX, self.numRX, self.numChirpsPerFrame, self.numADCSamples, 2)


    def get_antenna_pattern(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compute single antenna gain for the TI AWR1843BOOST with numerical stability."""
        # Convert angles to degrees and normalize
        _theta = np.clip(theta / np.pi * 180 / 56, -100, 100)  # Clip extreme values
        _phi = np.clip(phi / np.pi * 180 / 56, -100, 100)
        
        # Calculate gain components with numerical stability
        phi_component = 0.14 * _phi**6 + 0.13 * _phi**4 - 8.2 * _phi**2
        theta_component = 3.1 * _theta**8 - 22 * _theta**6 + 54 * _theta**4 - 55 * _theta**2
        
        # Calculate gain in dB with protection against extreme values
        gain_db = np.clip(phi_component + theta_component, -100, 100)  # Clip to reasonable dB range
        
        # Convert to linear scale safely
        gain = np.power(10, gain_db / 20)
        
        # Replace any remaining inf/nan with small finite value
        gain = np.nan_to_num(gain, nan=1e-10, posinf=1e-10, neginf=1e-10)
        
        return gain
    
    @cached_property
    def tx_rel_poses(self) -> np.ndarray:
        tx_rel_trans = -np.array([
            [0, 6, 0],
            [0, 10, 0],
            [0, 8, 2],
        ], dtype=np.float64) * 0.5 * self.waveLength

        tx_rel_poses = np.tile(np.eye(4, dtype=np.float64), (self.numTX, 1, 1))
        tx_rel_poses[:, 0:3, 3:4] = tx_rel_trans[..., np.newaxis]
        return tx_rel_poses
        

    @cached_property
    def rx_rel_poses(self) -> np.ndarray:
        rx_rel_trans = -np.array([
            [0, 0, 0],
            [0, 2, 0],
            [0, 4, 0],
            [0, 6, 0],
        ], dtype=np.float64) * 0.5 * self.waveLength
    
        rx_rel_poses = np.tile(np.eye(4, dtype=np.float64), (self.numRX, 1, 1))
        rx_rel_poses[:, 0:3, 3:4] = rx_rel_trans[..., np.newaxis]
        return rx_rel_poses
        

@dataclass
class ColoradarTI2243Config:
    frameShape=(12, 16, 16, 256, 2),
    numTX=12,
    numRX=16
    numChirpsPerFrame=16,
    numADCSamples=256,
    numVirtualAntennas=192,


