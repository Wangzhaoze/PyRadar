#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : test_rsp.py
# @IDE     : vscode

import unittest
import numpy as np
from pyradar.rsp.fft import (
    windowing,
    range_fft,
    doppler_fft,
    angle_fft,
    range_doppler_fft,
)

class TestWindowing(unittest.TestCase):
    def test_windowing_types(self):
        for wtype in ['Hamming', 'Blackman', 'Hann']:
            win = windowing(8, wtype)
            self.assertEqual(win.shape, (8,))
        with self.assertRaises(ValueError):
            windowing(8, 'Unknown')

class TestRangeFFT(unittest.TestCase):
    def setUp(self):
        self.adc_cube = np.random.rand(4, 8, 16)

    def test_range_fft_shape(self):
        out = range_fft(self.adc_cube)
        self.assertEqual(out.shape, self.adc_cube.shape)

    def test_range_fft_window(self):
        out = range_fft(self.adc_cube, window_type='Hamming')
        self.assertEqual(out.shape, self.adc_cube.shape)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError):
            range_fft([[1, 2], [3, 4]])

class TestDopplerFFT(unittest.TestCase):
    def setUp(self):
        self.adc_cube = np.random.rand(4, 8, 16)

    def test_doppler_fft_shape(self):
        out = doppler_fft(self.adc_cube)
        self.assertEqual(out.shape, self.adc_cube.shape)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError):
            doppler_fft([[1, 2], [3, 4]])

class TestAngleFFT(unittest.TestCase):
    def setUp(self):
        self.adc_cube = np.random.rand(4, 8, 16)

    def test_angle_fft_shape(self):
        out = angle_fft(self.adc_cube)
        self.assertEqual(out.shape, self.adc_cube.shape)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError):
            angle_fft([[1, 2], [3, 4]])

    def test_invalid_input_dimension(self):
        with self.assertRaises(ValueError):
            angle_fft(np.random.rand(4, 8))  # 2D array

class TestRangeDopplerFFT(unittest.TestCase):
    def setUp(self):
        self.adc_cube = np.random.rand(4, 8, 16)

    def test_range_doppler_fft_shape(self):
        out = range_doppler_fft(self.adc_cube)
        self.assertEqual(out.shape, self.adc_cube.shape)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError):
            range_doppler_fft([[1, 2], [3, 4]])

    def test_invalid_input_dimension(self):
        with self.assertRaises(ValueError):
            range_doppler_fft(np.random.rand(4, 8))  # 2D array

if __name__ == '__main__':
    unittest.main()