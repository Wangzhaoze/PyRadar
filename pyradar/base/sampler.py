#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : sampler.py
# @IDE     : vscode

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Sampler:
    """
    Parameters related to ADC sampling, including sample rate and sample details.
    """

    numChirpsPerFrame: Optional[int] = field(default=None)  # Number of chirps per frame
    numSamplesPerChirps: Optional[int] = field(default=None)  # Number of samples per chirp
    adcSampleRate: Optional[float] = field(default=None)  # ADC sample rate (samples per second)
    bitDepth: int = 16  # ADC resolution (bits)

    def __post_init__(self):


        # Optionally infer adcSampleRate if numSamplesPerChirps and chirp duration are known
        # NOTE: You may want to pass in a reference to the FMCW waveform to get chirpDuration
        # This implementation assumes you know chirpDuration externally
        # e.g., self.adcSampleRate = numSamplesPerChirps / chirpDuration

        # Add consistency checks or warnings
        if self.numChirpsPerFrame is None:
            print("Warning: numChirpsperFrame is not set.")
        if self.numSamplesPerChirps is None:
            print("Warning: numSamplesPerChirps is not set.")
        if self.adcSampleRate is None:
            print("Warning: adcSampleRate is not set.")
