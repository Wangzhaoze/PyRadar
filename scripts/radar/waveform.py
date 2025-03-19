#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : waveform.py
# @IDE     : vscode

from dataclasses import dataclass, field, asdict
from typing import Optional
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

C: float = 299792458


@dataclass
class WaveForm:
    """
    Configuration settings for transmitted chirps, including parameters such as
    bandwidth and chirp details.
    """

    def info(self):
        """
        Print the information of the WaveForm instance as a table.
        """
        # Convert the dataclass fields to a dictionary
        data = asdict(self)

        # Prepare data for the table
        table = [[key, value] for key, value in data.items()]

        # Print the table using tabulate
        print(tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid'))

    def save(self, config_path: str = 'chirp_config.yaml') -> None:
        # Open the specified file in write mode and save the dataclass as YAML
        save_path = os.path.join(config_path)
        with open(save_path, 'w') as file:
            yaml.dump(asdict(self), file, default_flow_style=False)

    @staticmethod
    def load(filename: str = 'configs/radar_cfg/TI-AWR1843BOOST.yaml') -> 'WaveForm':
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
            # Open the YAML file in read mode
            with open(filename, 'r') as file:
                radar_cfg = yaml.safe_load(file)  # Parse the YAML content

            # Return a WaveForm instance created from the parsed data
            return WaveForm(**radar_cfg.waveform)

        except Exception as e:
            # Raise a ValueError with details if the file cannot be loaded
            raise ValueError(
                f'Cannot load chirp configurations from file {filename}: {e}'
            )

    def show(self) -> None:
        return NotImplementedError

    @property
    def Bandwidth(self) -> float:
        """Calculate and return the bandwidth of the signal in Hz."""
        return NotImplementedError

    @property
    def waveLength(self) -> float:
        """Calculate and return the wavelength of the signal."""
        return NotImplementedError


@dataclass
class FMCW(WaveForm):
    """
    Configuration settings for transmitted chirps, including parameters such as
    bandwidth and chirp details.
    """

    chirpDuration: Optional[float] = field(
        default=None
    )  # Duration of one chirp (in seconds)
    chirpSlope: Optional[float] = field(default=None)  # Chirp slope (Hz/s)
    startFrequency: Optional[float] = field(default=None)  # Start frequency (Hz)
    bandwidth: Optional[float] = field(default=None)  # Bandwidth of the signal
    waveLength: Optional[float] = field(default=None)  # Wavelength of the signal

    def __post_init__(self):
        if self.chirpDuration is None:
            self.chirpDuration = self.bandwidth / self.chirpSlope
        if self.chirpSlope is None:
            self.chirpSlope = self.bandwidth / self.chirpDuration
        if self.startFrequency is None:
            self.startFrequency = C / self.waveLength
        if self.bandwidth is None:
            self.bandwidth = self.chirpSlope * self.chirpDuration
        if self.waveLength is None:
            self.waveLength = C / self.startFrequency

    # @property
    # def Bandwidth(self) -> float:
    #     """Calculate and return the bandwidth of the signal in Hz."""
    #     return self.chirpSlope * self.chirpDuration

    # @property
    # def waveLength(self) -> float:
    #     """Calculate and return the wavelength of the signal."""
    #     return C / self.startFrequency

    def show(self) -> None:
        numChirps = 5
        t = np.linspace(0, self.chirpDuration * numChirps, 100 * numChirps)
        k = np.tile(np.linspace(0, self.chirpDuration, 100), reps=numChirps)

        f_t = self.startFrequency + self.chirpSlope * k

        s_t = np.sin(
            2 * np.pi * (self.startFrequency * k + 0.5 * self.chirpSlope * k**2)
            + self.phaseShift
        )

        # Create the figure and axes
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

        # Plot frequency vs time for numChirps chirps
        axs[0].plot(t, f_t, label='Frequency (f(t))', color='blue')
        axs[0].set_title(f'Frequency vs Time ({numChirps} Chirps)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].grid(True)
        axs[0].legend()

        # Plot signal s(t) for  chirps
        axs[1].plot(t, s_t, label='Signal s(t)', color='orange')
        axs[1].set_title(f'Signal s(t) ({numChirps} Chirps)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        axs[1].legend()

        # Add a title for the entire figure
        fig.suptitle(f'WaveForm Visualization for {numChirps} Chirps', fontsize=16)

        # Show the plot
        plt.show()
