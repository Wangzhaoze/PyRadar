#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pyradar.base import Transceivers  # Assuming your code is saved in transceiver.py

class TestTransceiversInitialization(unittest.TestCase):
    """Test initialization functionality of Transceivers class"""
    
    def setUp(self):
        """Set up test TX and RX antenna positions"""
        self.tx_positions = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
        self.rx_positions = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
        self.transceivers = Transceivers(TX=self.tx_positions, RX=self.rx_positions)
    
    def test_initialization_with_valid_data(self):
        """Test initialization with valid data"""
        self.assertIsNotNone(self.transceivers)
        self.assertIsInstance(self.transceivers.TX, np.ndarray)
        self.assertIsInstance(self.transceivers.RX, np.ndarray)
    
    def test_initialization_with_empty_data(self):
        """Test initialization with empty data"""
        with self.assertRaises(Exception):
            Transceivers(TX=[], RX=[])
    
    def test_initialization_with_none_data(self):
        """Test initialization with None data"""
        with self.assertRaises(Exception):
            Transceivers(TX=None, RX=None)
    
    def test_tx_rx_arrays_shape(self):
        """Test shape of TX and RX arrays"""
        self.assertEqual(self.transceivers.TX.shape, (3, 3))
        self.assertEqual(self.transceivers.RX.shape, (4, 3))
    
    def test_dTX_dRX_properties(self):
        """Test dTX and dRX properties"""
        self.assertIsInstance(self.transceivers.dTX, np.ndarray)
        self.assertIsInstance(self.transceivers.dRX, np.ndarray)
        self.assertEqual(self.transceivers.dTX.shape, self.transceivers.TX.shape)
        self.assertEqual(self.transceivers.dRX.shape, self.transceivers.RX.shape)


class TestTransceiversProperties(unittest.TestCase):
    """Test properties of Transceivers class"""
    
    def setUp(self):
        self.tx_positions = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
        self.rx_positions = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
        self.transceivers = Transceivers(TX=self.tx_positions, RX=self.rx_positions)
    
    def test_numTX_property(self):
        """Test numTX property"""
        self.assertEqual(self.transceivers.numTX, 3)
    
    def test_numRX_property(self):
        """Test numRX property"""
        self.assertEqual(self.transceivers.numRX, 4)
    
    def test_numVirtualAntennas_property(self):
        """Test numVirtualAntennas property"""
        self.assertEqual(self.transceivers.numVirtualAntennas, 12)  # 3 TX * 4 RX
    
    def test_antennaGain_property(self):
        """Test antennaGain property"""
        # According to your implementation, this property should return NotImplemented
        self.assertEqual(self.transceivers.antennaGain, NotImplemented)


class TestTransceiversVirtualAntennaArray(unittest.TestCase):
    """Test virtual antenna array functionality"""
    
    def setUp(self):
        self.tx_positions = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
        self.rx_positions = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
        self.transceivers = Transceivers(TX=self.tx_positions, RX=self.rx_positions)
    
    def test_virtualAntennaArray_shape(self):
        """Test shape of virtual antenna array"""
        virtual_array = self.transceivers.virtualAntennaArray
        self.assertIsInstance(virtual_array, np.ndarray)
        self.assertEqual(virtual_array.shape, (12, 3))  # 12 virtual antennas, each with 3 coordinates
    
    def test_virtualAntennaArray_values(self):
        """Test values of virtual antenna array"""
        virtual_array = self.transceivers.virtualAntennaArray
        
        # Check if all Z coordinates are 0
        self.assertTrue(np.all(virtual_array[:, 2] == 0))
        
        # Check range of X and Y coordinates
        self.assertTrue(np.all(virtual_array[:, 0] >= 0))
        self.assertTrue(np.all(virtual_array[:, 1] >= 0))
        
        # Check specific values (first virtual antenna)
        expected_first_va = np.array([0, 0, 0])  # TX0 + RX0
        np.testing.assert_array_almost_equal(virtual_array[0], expected_first_va)


class TestTransceiversMaskFunction(unittest.TestCase):
    """Test mask functionality"""
    
    def setUp(self):
        self.tx_positions = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
        self.rx_positions = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
        self.transceivers = Transceivers(TX=self.tx_positions, RX=self.rx_positions)
    
    def test_mask_with_valid_indices(self):
        """Test mask with valid indices"""
        maskTX = np.array([True, False, True])  # Keep TX0 and TX2
        maskRX = np.array([True, True, False, True])  # Keep RX0, RX1, and RX3
        
        self.transceivers.mask(maskTX, maskRX)
        
        # Check number of antennas after masking
        self.assertEqual(self.transceivers.numTX, 2)
        self.assertEqual(self.transceivers.numRX, 3)
        self.assertEqual(self.transceivers.numVirtualAntennas, 6)  # 2 TX * 3 RX
    
    def test_mask_with_all_true(self):
        """Test mask with all True (no filtering)"""
        maskTX = np.array([True, True, True])
        maskRX = np.array([True, True, True, True])
        
        original_numTX = self.transceivers.numTX
        original_numRX = self.transceivers.numRX
        
        self.transceivers.mask(maskTX, maskRX)
        
        # Should remain unchanged
        self.assertEqual(self.transceivers.numTX, original_numTX)
        self.assertEqual(self.transceivers.numRX, original_numRX)
    
    def test_mask_with_all_false(self):
        """Test mask with all False (should raise error)"""
        maskTX = np.array([False, False, False])
        maskRX = np.array([False, False, False, False])
        
        with self.assertRaises(Exception):
            self.transceivers.mask(maskTX, maskRX)
    
    def test_mask_with_invalid_shape(self):
        """Test mask with invalid array shapes"""
        maskTX = np.array([True, False])  # Wrong shape for TX
        maskRX = np.array([True, True, False, True])
        
        with self.assertRaises(Exception):
            self.transceivers.mask(maskTX, maskRX)


class TestTransceiversVisualization(unittest.TestCase):
    """Test visualization methods (basic functionality tests)"""
    
    def setUp(self):
        self.tx_positions = [[0, 6, 0], [1, 8, 0], [0, 10, 0]]
        self.rx_positions = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
        self.transceivers = Transceivers(TX=self.tx_positions, RX=self.rx_positions)
    
    def test_show_method_exists(self):
        """Test that show method exists and is callable"""
        self.assertTrue(hasattr(self.transceivers, 'show'))
        self.assertTrue(callable(self.transceivers.show))
    
    def test_show3D_method_exists(self):
        """Test that show3D method exists and is callable"""
        self.assertTrue(hasattr(self.transceivers, 'show3D'))
        self.assertTrue(callable(self.transceivers.show3D))
    
    def test_show_method_no_error(self):
        """Test that show method doesn't raise errors"""
        try:
            # We'll test that the method can be called without errors
            # but we won't actually display the plot during testing
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            self.transceivers.show()
            success = True
        except Exception as e:
            success = False
            print(f"show method failed: {e}")
        
        self.assertTrue(success)
    
    def test_show3D_method_no_error(self):
        """Test that show3D method doesn't raise errors"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            self.transceivers.show3D()
            success = True
        except Exception as e:
            success = False
            print(f"show3D method failed: {e}")
        
        self.assertTrue(success)


class TestTransceiversEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""
    
    def test_single_tx_multiple_rx(self):
        """Test with single TX and multiple RX"""
        tx = [[0, 0, 0]]
        rx = [[0, 0, 0], [0, 1, 0], [0, 2, 0]]
        
        transceivers = Transceivers(TX=tx, RX=rx)
        
        self.assertEqual(transceivers.numTX, 1)
        self.assertEqual(transceivers.numRX, 3)
        self.assertEqual(transceivers.numVirtualAntennas, 3)
    
    def test_multiple_tx_single_rx(self):
        """Test with multiple TX and single RX"""
        tx = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        rx = [[0, 0, 0]]
        
        transceivers = Transceivers(TX=tx, RX=rx)
        
        self.assertEqual(transceivers.numTX, 3)
        self.assertEqual(transceivers.numRX, 1)
        self.assertEqual(transceivers.numVirtualAntennas, 3)
    
    def test_2d_positions(self):
        """Test with 2D positions (missing Z coordinate)"""
        tx = [[0, 6], [1, 8], [0, 10]]
        rx = [[0, 0], [0, 1], [0, 2], [0, 3]]
        
        # This should work as numpy will handle the conversion
        transceivers = Transceivers(TX=tx, RX=rx)
        
        self.assertEqual(transceivers.numTX, 3)
        self.assertEqual(transceivers.numRX, 4)
        
        # Virtual array should have 3 columns (X, Y, Z=0)
        self.assertEqual(transceivers.virtualAntennaArray.shape[1], 3)


if __name__ == '__main__':
    unittest.main()