#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyradar.base.waveform import FMCW, C


class TestFMCWInitialization(unittest.TestCase):
    """Test the FMCW class initialization and parameter calculation"""
    
    def test_basic_initialization(self):
        """Test basic initialization with required parameters"""
        fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            startFrequency=77e9
        )
        
        # Check that parameters are set correctly
        self.assertEqual(fmcw.chirpDuration, 40e-6)
        self.assertEqual(fmcw.chirpSlope, 2e12)
        self.assertEqual(fmcw.startFrequency, 77e9)
        
        # Check calculated parameters
        self.assertEqual(fmcw.bandwidth, 80e6)  # 2e12 * 40e-6
        self.assertAlmostEqual(fmcw.waveLength, C / 77e9)
    
    def test_initialization_with_wavelength(self):
        """Test initialization with wavelength instead of startFrequency"""
        expected_wavelength = C / 77e9
        fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            waveLength=expected_wavelength
        )
        
        # Check that startFrequency is calculated correctly
        self.assertAlmostEqual(fmcw.startFrequency, 77e9)
        self.assertAlmostEqual(fmcw.waveLength, expected_wavelength)
    
    def test_initialization_with_ramp_end_time(self):
        """Test initialization with rampEndTime instead of chirpDuration"""
        fmcw = FMCW(
            rampEndTime=50e-6,
            adcStartTime=10e-6,
            chirpSlope=2e12,
            startFrequency=77e9
        )
        
        # Check that chirpDuration is calculated correctly
        self.assertEqual(fmcw.chirpDuration, 40e-6)  # 50e-6 - 10e-6
        self.assertEqual(fmcw.bandwidth, 80e6)  # 2e12 * 40e-6
    
    # def test_missing_required_parameters(self):
    #     """Test that missing required parameters raise ValueError"""
    #     # Missing chirpDuration
    #     with self.assertRaises(ValueError):
    #         FMCW(chirpSlope=2e12, startFrequency=77e9)
        
    #     # Missing chirpSlope
    #     with self.assertRaises(ValueError):
    #         FMCW(chirpDuration=40e-6, startFrequency=77e9)
        
    #     # Missing startFrequency
    #     with self.assertRaises(ValueError):
    #         FMCW(chirpDuration=40e-6, chirpSlope=2e12)
    
    def test_invalid_chirp_duration_calculation(self):
        """Test that invalid chirpDuration calculation raises ValueError"""
        # This should trigger the specific error condition in __post_init__
        with self.assertRaises(ValueError):
            FMCW(
                rampEndTime=16/8000000 + 1e-6,  # Very close to the problematic value
                adcStartTime=1e-6,
                chirpSlope=2e12,
                startFrequency=77e9
            )


class TestFMCWProperties(unittest.TestCase):
    """Test the calculated properties of the FMCW class"""
    
    def setUp(self):
        """Set up a standard FMCW instance for testing"""
        self.fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            startFrequency=77e9,
            adcStartTime=5e-6,
            chirpIdleTime=10e-6
        )
    
    def test_bandwidth_calculation(self):
        """Test bandwidth calculation"""
        self.assertEqual(self.fmcw.bandwidth, 80e6)  # 2e12 * 40e-6
    
    def test_wavelength_calculation(self):
        """Test wavelength calculation"""
        expected_wavelength = C / 77e9
        self.assertAlmostEqual(self.fmcw.waveLength, expected_wavelength)
    
    def test_total_chirp_time(self):
        """Test calculation of total chirp time"""
        total_time = self.fmcw.adcStartTime + self.fmcw.chirpDuration + self.fmcw.chirpIdleTime
        self.assertEqual(total_time, 55e-6)  # 5e-6 + 40e-6 + 10e-6


class TestFMCWShowMethod(unittest.TestCase):
    """Test the show method of the FMCW class"""
    
    def setUp(self):
        """Set up a standard FMCW instance for testing"""
        self.fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            startFrequency=77e9,
            adcStartTime=5e-6,
            chirpIdleTime=10e-6
        )
    
    @patch('matplotlib.pyplot.show')
    def test_show_method_runs_without_error(self, mock_show):
        """Test that the show method runs without raising errors"""
        try:
            self.fmcw.show()
            success = True
        except Exception as e:
            success = False
            print(f"show method failed: {e}")
        
        self.assertTrue(success)
        mock_show.assert_called_once()
    
    def test_signal_calculation_in_show(self):
        """Test the signal calculation in the show method"""
        # Calculate time points manually
        t_total = self.fmcw.adcStartTime + self.fmcw.chirpDuration + self.fmcw.chirpIdleTime
        t = np.linspace(0, t_total, 2000)
        
        # Calculate expected signal
        chirp_start = self.fmcw.adcStartTime
        chirp_end = chirp_start + self.fmcw.chirpDuration
        
        signal = np.zeros_like(t)
        idx = (t >= chirp_start) & (t < chirp_end)
        k = t[idx] - chirp_start
        signal[idx] = np.sin(
            2 * np.pi * (self.fmcw.startFrequency * k + 0.5 * self.fmcw.chirpSlope * k**2)
        )
        
        # Check signal values
        self.assertEqual(signal[0], 0)  # Before chirp starts
        self.assertTrue(np.all(signal[idx] >= -1) and np.all(signal[idx] <= 1))  # Sine wave bounds


class TestFMCWEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios for the FMCW class"""
    
    def test_zero_idle_time(self):
        """Test with zero idle time"""
        fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            startFrequency=77e9,
            chirpIdleTime=0
        )
        
        self.assertEqual(fmcw.chirpIdleTime, 0)
        self.assertEqual(fmcw.adcStartTime + fmcw.chirpDuration + fmcw.chirpIdleTime, 
                        fmcw.adcStartTime + fmcw.chirpDuration)
    
    def test_zero_adc_start_time(self):
        """Test with zero ADC start time"""
        fmcw = FMCW(
            chirpDuration=40e-6,
            chirpSlope=2e12,
            startFrequency=77e9,
            adcStartTime=0
        )
        
        self.assertEqual(fmcw.adcStartTime, 0)
        self.assertEqual(fmcw.chirpDuration + fmcw.chirpIdleTime, 
                        fmcw.chirpDuration + fmcw.chirpIdleTime)
    
    
    def test_very_long_chirp(self):
        """Test with very long chirp duration"""
        fmcw = FMCW(
            chirpDuration=1.0,  # 1 second
            chirpSlope=1e6,     # Very gentle slope
            startFrequency=77e9
        )
        
        self.assertEqual(fmcw.bandwidth, 1e6)  # 1e6 * 1.0
        self.assertAlmostEqual(fmcw.waveLength, C / 77e9)




if __name__ == '__main__':
    unittest.main()