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
from .waveform import FMCW
from .sampler import Sampler
from .transceivers import Transceivers
from omegaconf import DictConfig
import yaml
from hydra_zen import instantiate

class Radar:
    """Radar system configuration, including antenna and receiver details."""

    def __init__(self, waveform: FMCW, sampler: Sampler, transceivers: Transceivers):
        self.waveform = waveform
        self.sampler = sampler
        self.transceivers = transceivers

    def __post_init__(self):
        pass
    
    @staticmethod
    def load(cfg_path: str) -> 'Radar':

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f'Config file not found at {cfg_path}')
        with open(cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)
            cfg = DictConfig(cfg)
        try:
            radar: Radar = instantiate(cfg)
        except Exception as e:
            raise ValueError(f"Config file error: {e}")
        return radar
    

    def info(self):
        """
        Print the information of the WaveForm, Sampler, and AntennaArray instances as a combined table.
        """
        # Prepare the data for each component
        waveform_info = asdict(self.waveform)
        sampler_info = asdict(self.sampler)
        antenna_info = asdict(self.transceivers)
        

        # Combine all data into one dictionary
        combined_info = {
            **{f'WaveForm - {key}': value for key, value in waveform_info.items()},
            **{f'Sampler - {key}': value for key, value in sampler_info.items()},
            **{f'AntennaArray - {key}': value for key, value in antenna_info.items()},
        }

        # Format as table
        table = [[key, value] for key, value in combined_info.items()]
        print(tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid'))


