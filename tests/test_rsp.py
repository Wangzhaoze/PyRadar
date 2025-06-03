#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : test_rsp.py
# @IDE     : vscode

import unittest
import numpy as np
from scipy.fft import fft
from pyradar.rsp.fft import (
    range_fft,
)  # Update this import based on your actual module name


class TestRangeFFT(unittest.TestCase):

    def test_range_fft_invalid_type(self):
        """
        Test case to check if ValueError is raised when input is not a numpy array.
        """
        with self.assertRaises(ValueError) as context:
            # Passing a list instead of a numpy array
            range_fft([[1, 2], [3, 4]])

        self.assertEqual(str(context.exception), 'Input must be a numpy array.')

    def test_range_fft_invalid_dimension(self):
        """
        Test case to check if ValueError is raised when input does not have three dimensions.
        """
        invalid_shapes = [
            np.random.rand(4, 5),  # 2D array
            np.random.rand(2, 3, 4, 5),  # 4D array
            np.random.rand(2),  # 1D array
        ]

        for invalid_input in invalid_shapes:
            with self.assertRaises(ValueError) as context:
                range_fft(invalid_input)
            self.assertEqual(
                str(context.exception),
                'Input array must have exactly three dimensions.',
            )


if __name__ == '__main__':
    unittest.main()
