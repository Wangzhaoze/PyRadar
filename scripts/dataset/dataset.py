#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-11-28
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : ColoRadar.py
# @IDE     : vscode

'Define Base Dataset Class'

import os
import glob
from typing import Optional, Union
import numpy as np
from torch.utils.data import Dataset, IterableDataset

class BaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        scene_id: str,
    ):
        self.data_dir = data_dir
        self.scene_id = scene_id

    def _num_frames(self):
        return NotImplementedError

    def __len__(self):
        return self._num_frames()

    def __getitem__(self, idx: int):
        return NotImplementedError


class KittiDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        scene_id: str,
    ):
        self.data_dir = data_dir
        self.scene_id = scene_id

    def _num_frames(self):
        return NotImplementedError

    def __len__(self):
        return self._num_frames()

    def __getitem__(self, idx: int):
        return NotImplementedError


class nuScenesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        scene_id: str,
    ):
        self.data_dir = data_dir
        self.scene_id = scene_id

    def _num_frames(self):
        return NotImplementedError

    def __len__(self):
        return self._num_frames()

    def __getitem__(self, idx: int):
        return NotImplementedError
