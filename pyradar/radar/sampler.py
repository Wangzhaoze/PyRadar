#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : sampler.py
# @IDE     : vscode

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sampler:
    """
    Parameters related to ADC sampling, including sample rate and sample details.
    """

    numChirpsperFrame: Optional[int] = field(default=None)  # Number of chirps per frame
    numSamplesPerChirps: Optional[int] = field(
        default=None
    )  # Number of samples per chirp
    frameDuration: Optional[float] = field(
        default=None
    )  # Duration of one frame (in seconds)
    chirpDuration: Optional[float] = field(
        default=None
    )  # Duration of each chirp (in seconds)
    adcSampleRate: Optional[float] = field(default=None)  # ADC sample rate

    def __post_init__(self):
        if self.numChirpsperFrame is None:
            self.numChirpsperFrame = self.frameDuration / self.chirpDuration
        if self.numSamplesPerChirps is None:
            self.numSamplesPerChirps = self.adcSampleRate * self.chirpDuration
        if self.frameDuration is None:
            self.frameDuration = self.numChirpsperFrame * self.chirpDuration
        if self.chirpDuration is None:
            self.chirpDuration = self.frameDuration / self.numChirpsperFrame
        if self.adcSampleRate is None:
            self.adcSampleRate = self.numSamplesPerChirps / self.chirpDuration
