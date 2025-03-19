import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import yaml

from scripts.radar.radar import RadarConfig, TI_MMWCAS_RF_EVM


radar_cfg = RadarConfig.load(cfg_path='configs/radar_cfg/TI-MMWCAS-RF-EVM.yaml')
radar = instantiate(radar_cfg)

raadr = TI_MMWCAS_RF_EVM()
print(radar_cfg)
