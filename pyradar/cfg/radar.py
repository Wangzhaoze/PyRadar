#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : radar.py
# @IDE     : vscode


"""Radar Sensor Model."""

from dataclasses import dataclass, field, asdict
from typing import Optional
from typing import Any
from tabulate import tabulate
import numpy as np
import os
from .waveform import WaveForm, FMCW
from .sampler import Sampler
from .transceivers import Transceivers
import yaml

from omegaconf import DictConfig


class RadarConfig(DictConfig):
    _target_: str = 'pyradar.radar.Radar'
    name = 'Radar'
    waveform: DictConfig
    sampler: DictConfig
    transceivers: DictConfig

    @classmethod
    def load(
        self, cfg_path: str = 'configs/radar_cfg/TI-MMWCAS-RF-EVM.yaml'
    ) -> 'RadarConfig':
        self.name = os.path.splitext(os.path.basename(cfg_path))[0]
        try:
            with open(cfg_path, 'r') as file:
                cfg = yaml.safe_load(file)
            return self(cfg)
        except FileNotFoundError:
            raise FileNotFoundError(f'Config file not found at {cfg_path}')


@dataclass
class Radar:
    """Radar system configuration, including antenna and receiver details."""

    name: Optional[str] = field(default=None)
    waveform: Optional[WaveForm] = field(default=None)
    sampler: Optional[Sampler] = field(default=None)
    transceivers: Optional[Transceivers] = field(default=None)

    def __str__(self):
        return self.name


class TI_MMWCAS_RF_EVM(Radar):
    def __init__(self):
        waveform = FMCW(
            chirpDuration=4.99999987369e-06,
            chirpSlope=7.90000010527e13,
            startFrequency=76999999488.0,
        )
        sampler = Sampler(
            numSamplesPerChirps=256,
            numChirpsperFrame=16,
            adcSampleRate=8000000,
            chirpDuration=4.99999987369e-06,
        )
        transceivers = Transceivers(numTX=12, numRX=16)
        super().__init__(waveform, sampler, transceivers)

    def __str__(self):
        return 'TI-MMWCAS-RF-EVM Radar Sensor'

    def info(self):
        """
        Print the information of the WaveForm, Sampler, and AntennaArray instances as a combined table.
        """
        # Prepare the data for each component
        waveform_info = asdict(self.waveform)
        sampler_info = asdict(self.sampler)
        antenna_info = asdict(self.transceivers)

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
