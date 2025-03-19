#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
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
    def __init__(self, data_dir: str = "data\ColoRadar", scene_id: str = "2_28_2021_outdoors_run0"):
        self.data_dir = data_dir
        self.scene_id = scene_id


    @property
    def num_frames(self):
        return NotImplementedError
    

    def load_pose(self, frame_idx: int = 0) -> np.ndarray:
        self.pose_dir = os.path.join(self.data_dir, self.scene_id, 'groundtruth', 'groundtruth_poses.txt')
        poses = np.loadtxt(self.pose_dir)[frame_idx]
        # x, y, z, qx, qy, qz, qw = poses[0]
        return poses
    
    def load_lidar(self, frame_idx: int = 0):
        self.lidar_dir = os.path.join(self.data_dir, self.scene_id, 'lidar', 'pointclouds')
        lidar_files = glob.glob(os.path.join(self.lidar_dir, '*.bin'))

        cld = np.fromfile(lidar_files[frame_idx], np.float32)
        cld = np.reshape(cld, (-1, Dim_Lidar))

        return cld
    
    def load_scradar(self, frame_idx: int = 0) -> np.ndarray:
        scradar_dir = os.path.join(self.data_dir, self.scene_id, 'single_chip', 'adc_samples', 'data')
        scradar_files = glob.glob(os.path.join(scradar_dir, '*.bin'))
        data = np.fromfile(scradar_files[frame_idx], np.int16)
        data = np.reshape(data,             (
                    3, #coloradar.calibration.scradar.waveform.num_tx,
                    4, #coloradar.calibration.scradar.waveform.num_rx,
                    128, #coloradar.calibration.scradar.waveform.num_chirps_per_frame,
                    128, #coloradar.calibration.scradar.waveform.num_adc_samples_per_chirp,
                    2 # I and Q signal measurements
                ))
        I = np.float16(data[:, :, :, :, 0])
        Q = np.float16(data[:, :, :, :, 1])
        adc_samples = I + 1j * Q

        return np.array(adc_samples, dtype=np.complex64)
    
    def load_ccradar(self, frame_idx: int = 0) -> np.ndarray:
        ccradar_dir = os.path.join(self.data_dir, self.scene_id, 'cascade', 'adc_samples', 'data')
        scradar_files = glob.glob(os.path.join(ccradar_dir, '*.bin'))
        data = np.fromfile(scradar_files[frame_idx], np.int16)
        data = np.reshape(data,             (
                    12, #coloradar.calibration.ccradar.waveform.num_tx,
                    16, #coloradar.calibration.ccradar.waveform.num_rx,
                    16, #coloradar.calibration.ccradar.waveform.num_chirps_per_frame,
                    256, #coloradar.calibration.ccradar.waveform.num_adc_samples_per_chirp,
                    2 # I and Q signal measurements
                ))
        I = np.float16(data[:, :, :, :, 0])
        Q = np.float16(data[:, :, :, :, 1])
        adc_samples = I + 1j * Q

        adc_samples = np.transpose(adc_samples, (0, 1, 3, 2))

        return adc_samples
    
    def load_ccradar_heatmap(self, frame_idx: int = 0):
        ccradar_heatmap_dir = os.path.join(self.data_dir, self.scene_id, 'cascade', 'heatmaps', 'data')
        scradar_heatmap_files = glob.glob(os.path.join(ccradar_heatmap_dir, '*.bin'))
        heatmap = np.fromfile(scradar_heatmap_files[frame_idx], np.float32)
        return heatmap




if __name__ == "__main__":
    coloradar = ColoRadarDataset()
    print(coloradar.load_pose(0))
    print(coloradar.load_lidar(0))
    print(coloradar.load_scradar(0))
    print(coloradar.load_ccradar(0))



# from typing import Optional
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# from core.config import *
# from core.calibration import Calibration
# from core.radar import SCRadar, CCRadar
# from core.dataset import Coloradar
# from core.utils import radardsp as rdsp

# coloradar = Coloradar()


# folderpath = "C:/Users/wangzh179/Documents/Datasets/ColorRadar/2_28_2021_outdoors_run0/cascade/adc_samples/data"
# filepath = os.path.join(folderpath, "frame_0.bin")
# try:
#     data = np.fromfile(filepath, np.int16)
#     data = np.reshape(data,             (
#                 coloradar.calibration.ccradar.waveform.num_tx, #12
#                 coloradar.calibration.ccradar.waveform.num_rx, #16
#                 coloradar.calibration.ccradar.waveform.num_chirps_per_frame,#16
#                 coloradar.calibration.ccradar.waveform.num_adc_samples_per_chirp,#256
#                 2 # I and Q signal measurements
#             ))
#     I = np.float16(data[:, :, :, :, 0])
#     Q = np.float16(data[:, :, :, :, 1])
#     adc_samples = I + 1j * Q
# except FileNotFoundError:
#     # error(f"File '{filepath}' not found.")
#     data = None

# # ADC sampling frequency
# fs: float = coloradar.calibration.ccradar.waveform.adc_sample_frequency

# # Frequency slope
# fslope: float = coloradar.calibration.ccradar.waveform.frequency_slope

# # Start frequency
# fstart: float = coloradar.calibration.ccradar.waveform.start_frequency

# # Ramp end time
# te: float = coloradar.calibration.ccradar.waveform.ramp_end_time

# # Chirp time
# tc: float = coloradar.calibration.ccradar.waveform.idle_time + te

# Na = 64
# Ne = 64

# # if coloradar.calibration.sensor != "scradar":
# #     adc_samples *= coloradar.calibration.ccradar.get_frequency_calibration()
# #     adc_samples *= coloradar.calibration.ccradar.get_phase_calibration()

# ntx, nrx, nc, ns = adc_samples.shape

# rfft = np.fft.fft(adc_samples, ns, -1) - coloradar.calibration.ccradar.get_coupling_calibration()
# dfft = np.fft.fft(rfft, nc, -2)
# dfft = np.fft.fftshift(dfft, -2)
# dfft = dfft.reshape(ntx * nrx, nc, ns)

# # signal = np.sum(dfft, (1, 2))
# # print("signal shape: ", signal.shape)

# vbins = rdsp.get_velocity_bins(ntx, nc, fstart, tc)
# rbins = rdsp.get_range_bins(ns, fs, fslope)

# # Azimuth bins
# ares = np.pi / Na
# abins = np.arange(-np.pi/2, np.pi/2, ares)
# # Elevation
# eres = np.pi / Ne
# ebins = np.arange(-np.pi/2, np.pi/2, eres)

# spectrum = np.zeros((ns, Ne, Na, 1), dtype=np.complex128)

# signal = np.sum(dfft, (1, 2))

# spectrum = rdsp.music(
#     signal, coloradar.calibration.ccradar.antenna.txl, coloradar.calibration.ccradar.antenna.rxl, abins, ebins
# )
# '''
# hmap = np.zeros((Na * Ne, 3))

# for eidx in range(Ne):
#     for aidx in range(Na):
#         hmap_idx: int = aidx + Na * eidx
#         hmap[hmap_idx] = np.array([
#             abins[aidx],
#             ebins[eidx],
#             spectrum[hmap_idx],
#         ])
# '''

# # ax = plt.axes(projection="3d")
# fig = plt.figure()
# # ax = fig.gca(projection="3d")

# ax = fig.add_subplot(111, projection="3d")
# # _, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_title("Test MUSIC")
# ax.set_xlabel("Azimuth")
# ax.set_ylabel("Elevation")
# ax.set_zlabel("Gain")
# el, az = np.meshgrid(ebins, abins)
# '''
# map = ax.scatter(
#     hmap[:, 0],
#     hmap[:, 1],
#     hmap[:, 2],
#     c=hmap[:, 2],
#     cmap=plt.cm.get_cmap()
# )
# '''
# surf = ax.plot_surface(
#     el, az, spectrum.reshape(Ne, Na),
#     cmap="coolwarm",
#     rstride=1,
#     cstride=1,
#     alpha=None,
#     # linewidth=0,
#     # antialiased=False
# )
# plt.colorbar(surf, shrink=0.5, aspect=1)
# plt.show()
# print()