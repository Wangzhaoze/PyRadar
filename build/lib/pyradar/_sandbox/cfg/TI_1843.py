from torch.utils.data import Dataset
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import glob
from functools import cached_property
import math
from radar_gs_utils import interpolate_quaternion_poses
import matplotlib.pyplot as plt
import torch
import struct
from typing import Tuple
import open3d as o3d
from tqdm import trange


# rampDuration = adcStartTime + numADCSamples * 1 / adcSampleRate
# FrameDuration = numChirpsPerFrame * (rampDuration + idleTime)

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

    @cached_property
    def antennaGain(self) -> torch.Tensor:
        self.spectrum_resolution = 180
        return torch.from_numpy(
            self.get_antenna_pattern(
                theta=np.linspace(-np.pi / 2, np.pi / 2, self.spectrum_resolution),
                phi=np.linspace(-np.pi / 2, np.pi / 2, self.spectrum_resolution),
            )
        )
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




@dataclass
class Record(ABC):
    def __init__(self, files: list,
        time_stamps: List[float],
        calibration: np.ndarray,
        ):

        self.files = files
        self.time_stamps = time_stamps
        self.calibration = calibration

    @abstractmethod
    def __len__(self) -> int:
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> np.ndarray:
        return NotImplementedError

    def _load_data(self, filename: str):
        return None
    

class EgoTrajectory(Record):
    def __init__(self, files: list,
        time_stamps: List[float],
        calibration: np.ndarray,
        ):
        super().__init__(files, time_stamps, calibration)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.files[idx]
    
class Lidar(Record):
    def __init__(self, files: List[str],
        time_stamps: List[float],
        calibration: np.ndarray,
        ):
        super().__init__(files, time_stamps, calibration)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.load_point_cloud(self.files[idx])

    def load_point_cloud(self, filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            print('File ' + filename + ' not found')
            return None

        with open(filename, mode='rb') as file:
            cloud_bytes = file.read()

        cloud_vals = struct.unpack(str(len(cloud_bytes) // 4)+'f', cloud_bytes)
        pcd_array = np.array(cloud_vals).reshape((-1, 4))

        return pcd_array
    
    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        self.raw_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.time_stamps,
        )

        # self.lidar_poses = self.calibration[np.newaxis, ...] @ interpolated_poses 
        self.lidar_calib_poses = self.raw_poses @ self.calibration[np.newaxis, ...]

        return self.lidar_calib_poses
    

class Radar(Record):
    def __init__(self, files: list,
        time_stamps: List[float],
        calibration: np.ndarray,
        radar_config: Union[ColoradarTI1843BoostConfig, ColoradarTI2243Config],
        ):
        super().__init__(files, time_stamps, calibration)
        self.radar_config = radar_config

        self.drop_last_n_frames = 20
        if self.drop_last_n_frames:
            self.files = self.files[:-self.drop_last_n_frames]
            self.time_stamps = self.time_stamps[:-self.drop_last_n_frames]

    @cached_property
    def num_frames(self) -> int:
        return len(self.files)

    def __len__(self) -> int:
        return len(self.time_stamps)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.files[idx]
    
    def _load_data(self, filename) -> np.ndarray:
        adc_samples = np.fromfile(filename, dtype=np.int16).reshape(self.radar_config.frameShape)
        
        I = np.float16(adc_samples[:, :, :, :, 0])
        Q = np.float16(adc_samples[:, :, :, :, 1])
        # adc_samples = I + 1j * Q
        adc_samples = np.concatenate((I[:, :, :, :, np.newaxis], Q[:, :, :, :, np.newaxis]), axis=-1)

        return adc_samples.astype(np.float32)
    
    @cached_property
    def chirp_timestamps(self) -> np.ndarray:
        total_num_chirps = self.num_frames * self.radar_config.numChirpsPerFrame
        chirp_timestamps = np.zeros(total_num_chirps, dtype=np.float64)
        chirp_indices = np.arange(self.radar_config.numChirpsPerFrame, dtype=np.int32)
        for idxFrameTimeStamp in range(self.num_frames):
            start = idxFrameTimeStamp * self.radar_config.numChirpsPerFrame
            stop = start + self.radar_config.numChirpsPerFrame
            chirp_timestamps[start:stop] = self.time_stamps[idxFrameTimeStamp] \
                + chirp_indices * (self.radar_config.rampDuration + self.radar_config.idleTime) \
                + self.radar_config.adcStartTime
        return chirp_timestamps
    
    def interpolate_poses(self, traj: EgoTrajectory):
        """
        Interpolate poses for each chirp.
        """
        interpolated_poses = interpolate_quaternion_poses(
            src_poses=traj.files,
            src_stamps=traj.time_stamps,
            tgt_stamps=self.chirp_timestamps,
        )

        self.chirp_calib_poses = interpolated_poses @ self.calibration[np.newaxis, ...]

        return self.chirp_calib_poses

