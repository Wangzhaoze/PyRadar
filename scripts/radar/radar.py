#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : radar.py
# @IDE     : vscode


"""Radar Sensor Model."""

from dataclasses import dataclass, asdict
from typing import Any
from tabulate import tabulate
import numpy as np
from .waveform import WaveForm
from .sampler import Sampler
from .transceivers import Transceivers

from omegaconf import DictConfig

class RadarConfig:
    _target_: str = 'scripts.radar.Radar'
    WaveFormConfig: DictConfig
    SamplerConfig: DictConfig
    TransceiversConfig: DictConfig


@dataclass
class Radar:
    """Radar system configuration, including antenna and receiver details."""

    waveform: WaveForm
    sampler: Sampler
    antenna_array: Transceivers

    def info(self):
        """
        Print the information of the WaveForm, Sampler, and AntennaArray instances as a combined table.
        """
        # Prepare the data for each component
        waveform_info = asdict(self.waveform)
        sampler_info = asdict(self.sampler)
        antenna_info = asdict(self.antenna_array)

        # Add derived properties
        waveform_info['Bandwidth'] = self.waveform.Bandwidth
        waveform_info['waveLength'] = self.waveform.waveLength

        # Combine all data into one dictionary
        combined_info = {
            **{f'WaveForm - {key}': value for key, value in waveform_info.items()},
            **{f'Sampler - {key}': value for key, value in sampler_info.items()},
            **{f'AntennaArray - {key}': value for key, value in antenna_info.items()},
        }

        # Format as table
        table = [[key, value] for key, value in combined_info.items()]
        print(tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid'))

# class TI_AWR1843BOOST(Radar):
#     def __init__(self):
#         waveform = WaveForm.load()
#         sampler = Sampler()
#         antenna_array = Transceivers()
#         super().__init__(waveform, sampler, antenna_array)

#     def __str__(self):
#         return 'TI-AWR1843BOOST Radar Sensor'
    
# class TI_MMWCAS_RF_EVM(Radar):
#     def __init__(self):
#         waveform = WaveForm.load('configs/radar_cfg/TI-MMWCAS-RF-EVM.yaml')
#         sampler = Sampler()
#         antenna_array = Transceivers()
#         super().__init__(waveform, sampler, antenna_array)

#     def __str__(self):
#         return 'TI-MMWCAS-RF-EVM Radar Sensor'
    
