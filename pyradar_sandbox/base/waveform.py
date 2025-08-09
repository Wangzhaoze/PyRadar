#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : waveform.py
# @IDE     : vscode

from dataclasses import dataclass, field, asdict
from typing import Optional
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

C: float = 299792458  # Speed of light in m/s

@dataclass
class WaveForm:
    """
    Base class for waveform configuration settings for transmitted chirps.
    Includes methods for displaying, saving, and loading waveform parameters.
    """

    def info(self):
        """
        Print the information of the WaveForm instance as a formatted table.

        Uses the tabulate library to display all dataclass fields and their values.
        """
        data = asdict(self)
        table = [[key, value] for key, value in data.items()]
        print(tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid'))

    def save(self, config_path: str) -> None:
        """
        Save the current waveform configuration to a YAML file.

        Args:
            config_path (str): The path to save the YAML configuration file.
        """
        save_path = os.path.join(config_path)
        with open(save_path, 'w') as file:
            yaml.dump(asdict(self), file, default_flow_style=False)

    @staticmethod
    def load(filename: str) -> 'WaveForm':
        """
        Load a WaveForm dataclass instance from a YAML file.

        Args:
            filename (str): Path to the YAML file to load.

        Returns:
            WaveForm: An instance of the WaveForm dataclass created from the YAML file contents.

        Raises:
            ValueError: If the file cannot be read or the content cannot be loaded into a WaveForm object.
        """
        try:
            with open(filename, 'r') as file:
                radar_cfg = yaml.safe_load(file)
            return WaveForm(**radar_cfg.waveform)
        except Exception as e:
            raise ValueError(
                f'Cannot load chirp configurations from file {filename}: {e}'
            )

    def show(self) -> None:
        """
        Visualize the waveform. Should be implemented by subclasses.
        """
        return NotImplementedError

@dataclass
class FMCW(WaveForm):
    """
    FMCW (Frequency-Modulated Continuous Wave) waveform configuration settings for transmitted chirps.
    Includes bandwidth, duration, frequency sweep parameters, and timing details.
    """

    chirpDuration: Optional[float] = field(default=None)       # Duration of one chirp (in seconds)
    chirpSlope: Optional[float] = field(default=None)          # Chirp slope (Hz/s)
    startFrequency: Optional[float] = field(default=77e9)      # Start frequency (Hz)
    bandwidth: Optional[float] = field(default=None)           # Bandwidth (Hz)
    waveLength: Optional[float] = field(default=None)          # Wavelength (meters)
    adcStartTime: Optional[float] = field(default=0)           # ADC start time (s)
    chirpIdleTime: Optional[float] = field(default=0)          # Chirp idle time (s)
    rampEndTime: Optional[float] = field(default=None)         # Frequency ramp duration (s)

    def __post_init__(self):
        """
        Automatically calculate missing parameters after initialization.
        Ensures all required parameters are set and consistent.
        """
        if self.chirpDuration is None:
            self.chirpDuration = self.rampEndTime - self.adcStartTime
            if np.allclose(self.chirpDuration, 16 / 8000000):
                raise ValueError("Invalid chirpDuration.")
        if self.bandwidth is None:
            self.bandwidth = self.chirpSlope * self.chirpDuration
        if self.chirpSlope is None:
            self.chirpSlope = self.bandwidth / self.chirpDuration

        if self.waveLength is not None and self.startFrequency is None:
            self.startFrequency = C / self.waveLength
        elif self.startFrequency is not None and self.waveLength is None:
            self.waveLength = C / self.startFrequency

        required_params = [self.chirpDuration, self.chirpSlope, self.startFrequency]
        if any(p is None for p in required_params):
            raise ValueError("Missing required chirp parameters. Please set at least bandwidth & slope or duration.")

    def show(self) -> None:
        """
        Plot the frequency-time and signal-time curves for one FMCW chirp.

        Visualizes the frequency sweep and the corresponding time-domain signal,
        including ADC start, chirp duration, and idle time.
        """
        t_total = self.adcStartTime + self.chirpDuration + self.chirpIdleTime
        t = np.linspace(0, t_total, 2000)
        freq = np.zeros_like(t)
        signal = np.zeros_like(t)

        chirp_start = self.adcStartTime
        chirp_end = chirp_start + self.chirpDuration
        idle_end = chirp_end + self.chirpIdleTime

        idx = (t >= chirp_start) & (t < chirp_end)
        k = t[idx] - chirp_start
        freq[idx] = self.startFrequency + self.chirpSlope * k
        signal[idx] = np.sin(
            2 * np.pi * (self.startFrequency * k + 0.5 * self.chirpSlope * k**2)
        )

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

        # Frequency vs Time plot
        ax_f_t: plt.Axes = axs[0]
        ax_f_t.plot(t, freq, label='Frequency (f(t))', color='blue')
        ax_f_t.set_title('Frequency vs Time (1 Chirp)')
        ax_f_t.set_xlabel('Time (s)')
        ax_f_t.set_ylabel('Frequency (Hz)')
        ax_f_t.grid(True)
        ax_f_t.legend()

        ylim = ax_f_t.get_ylim()
        y_top = ylim[1]
        y_dim = y_top + 0.2 * (ylim[1] - ylim[0])

        # Draw vertical lines for timing events
        ax_f_t.vlines([0, chirp_start, chirp_end, idle_end], ylim[0], y_dim, colors=['k','g','r','c'], linestyles='dashed')
        # Annotate chirp duration
        ax_f_t.annotate('', xy=(chirp_start, y_dim), xytext=(chirp_end, y_dim),
                    arrowprops=dict(arrowstyle='<->', color='m', lw=2))
        ax_f_t.text((chirp_start+chirp_end)/2, y_dim*1.01, 'chirpDuration', color='m', ha='center', va='bottom', fontsize=10)

        # Annotate ADC start time
        ax_f_t.annotate('', xy=(0, y_dim*0.98), xytext=(chirp_start, y_dim*0.98),
                    arrowprops=dict(arrowstyle='<->', color='g', lw=2))
        ax_f_t.text(chirp_start/2, y_dim*0.97, 'adcStartTime', color='g', ha='center', va='top', fontsize=10)

        # Annotate chirp idle time
        ax_f_t.annotate('', xy=(chirp_end, y_dim*0.96), xytext=(idle_end, y_dim*0.96),
                    arrowprops=dict(arrowstyle='<->', color='c', lw=2))
        ax_f_t.text((chirp_end+idle_end)/2, y_dim*0.95, 'chirpIdleTime', color='c', ha='center', va='top', fontsize=10)

        # Signal vs Time plot
        ax_s_t: plt.Axes = axs[1]
        ax_s_t.plot(t, signal, label='Signal s(t)', color='orange')
        ax_s_t.set_title('Signal s(t) (1 Chirp)')
        ax_s_t.set_xlabel('Time (s)')
        ax_s_t.set_ylabel('Amplitude')
        ax_s_t.grid(True)
        ax_s_t.legend()

        fig.suptitle(
            f'WaveForm Visualization',
            fontsize=14
        )
        plt.show()
