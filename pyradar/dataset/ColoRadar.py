#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025-03-18
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : ColoRadar.py
# @IDE     : vscode

import os
import glob
from typing import Optional, Union
import numpy as np


Dim_Lidar: int = 4
Dim_SCRadar: int = 2


class ColoRadarDataset:
    def __init__(
        self,
        data_dir: str = 'data\\ColoRadar',
        scene_id: str = '2_28_2021_outdoors_run0',
    ):
        self.data_dir = data_dir
        self.scene_id = scene_id

    @property
    def num_frames(self):
        return NotImplementedError

    def load_pose(self, frame_idx: int = 0) -> np.ndarray:
        self.pose_dir = os.path.join(
            self.data_dir, self.scene_id, 'groundtruth', 'groundtruth_poses.txt'
        )
        poses = np.loadtxt(self.pose_dir)[frame_idx]
        # x, y, z, qx, qy, qz, qw = poses[0]
        return poses

    def load_lidar(self, frame_idx: int = 0):
        self.lidar_dir = os.path.join(
            self.data_dir, self.scene_id, 'lidar', 'pointclouds'
        )
        lidar_files = glob.glob(os.path.join(self.lidar_dir, '*.bin'))

        cld = np.fromfile(lidar_files[frame_idx], np.float32)
        cld = np.reshape(cld, (-1, Dim_Lidar))

        return cld

    def load_scradar(self, frame_idx: int = 0) -> np.ndarray:
        scradar_dir = os.path.join(
            self.data_dir, self.scene_id, 'single_chip', 'adc_samples', 'data'
        )
        scradar_files = glob.glob(os.path.join(scradar_dir, '*.bin'))
        data = np.fromfile(scradar_files[frame_idx], np.int16)
        data = np.reshape(
            data,
            (
                3,  # coloradar.calibration.scradar.waveform.num_tx,
                4,  # coloradar.calibration.scradar.waveform.num_rx,
                128,  # coloradar.calibration.scradar.waveform.num_chirps_per_frame,
                128,  # coloradar.calibration.scradar.waveform.num_adc_samples_per_chirp,
                2,  # I and Q signal measurements
            ),
        )
        I = np.float16(data[:, :, :, :, 0])
        Q = np.float16(data[:, :, :, :, 1])
        adc_samples = I + 1j * Q

        return np.array(adc_samples, dtype=np.complex64)

    def load_ccradar(self, frame_idx: int = 0) -> np.ndarray:
        ccradar_dir = os.path.join(
            self.data_dir, self.scene_id, 'cascade', 'adc_samples', 'data'
        )
        scradar_files = glob.glob(os.path.join(ccradar_dir, '*.bin'))
        data = np.fromfile(scradar_files[frame_idx], np.int16)
        data = np.reshape(
            data,
            (
                12,  # coloradar.calibration.ccradar.waveform.num_tx,
                16,  # coloradar.calibration.ccradar.waveform.num_rx,
                16,  # coloradar.calibration.ccradar.waveform.num_chirps_per_frame,
                256,  # coloradar.calibration.ccradar.waveform.num_adc_samples_per_chirp,
                2,  # I and Q signal measurements
            ),
        )
        I = np.float16(data[:, :, :, :, 0])
        Q = np.float16(data[:, :, :, :, 1])
        adc_samples = I + 1j * Q

        adc_samples = np.transpose(adc_samples, (0, 1, 3, 2))

        return adc_samples

    def load_ccradar_heatmap(self, frame_idx: int = 0):
        ccradar_heatmap_dir = os.path.join(
            self.data_dir, self.scene_id, 'cascade', 'heatmaps', 'data'
        )
        scradar_heatmap_files = glob.glob(os.path.join(ccradar_heatmap_dir, '*.bin'))
        heatmap = np.fromfile(scradar_heatmap_files[frame_idx], np.float32)
        return heatmap
