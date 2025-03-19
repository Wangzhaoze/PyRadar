#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : sampler.py
# @IDE     : vscode

from dataclasses import dataclass


@dataclass
class Sampler:
    """
    Parameters related to ADC sampling, including sample rate and sample details.
    """

    numChirpsperFrame: int = 255  # Number of chirps per frame
    numSamplesPerChirps: int = 255  # Number of samples per chirp
    frameDuration: float = 0.005  # Duration of one frame (in seconds)

    @property
    def chirpDuration(self) -> float:
        """Calculate the duration of each chirp (in seconds)."""
        return self.frameDuration / self.numChirpsperFrame

    @property
    def adcSampleRate(self) -> float:
        """Calculate and return the ADC sample rate."""
        return self.numSamplesPerChirps / self.chirpDuration
