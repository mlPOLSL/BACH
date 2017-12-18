import unittest

import numpy as np
from DataPreprocessing.Grid.data_types import WindowSize, Stride

from DataPreprocessing.Grid.grid import divide_into_patches, sliding_window


class TestDividingIntoPatches(unittest.TestCase):
    def setUp(self):
        self.pixels = np.ones((70, 50))

    def test_that_all_patches_have_equal_shape(self):
        patches = divide_into_patches(self.pixels, 5, 5)
        for patch in patches:
            self.assertTrue(patch.shape == (14, 10))

    def test_that_all_correct_number_of_patches_is_obtained(self):
        patches = divide_into_patches(self.pixels, 14, 10)
        self.assertEqual(len(patches), 140)

    def test_that_all_channels_are_returned(self):
        pixels3D = np.ones((70, 50, 3))
        patches = divide_into_patches(pixels3D, 5, 5)
        for patch in patches:
            self.assertTrue(patch.shape == (14, 10, 3))


class TestSlidingWindow(unittest.TestCase):
    def setUp(self):
        self.pixels = np.ones((70, 50))

    def test_that_all_patches_have_equal_shape(self):
        patches = sliding_window(self.pixels, WindowSize(5, 5))
        for patch in patches:
            self.assertTrue(patch.shape == (5, 5))

    def test_that_correct_number_of_patches_is_returned(self):
        pixels = np.ones((100, 50))
        patches = sliding_window(pixels, WindowSize(5, 5), Stride(3, 3))
        self.assertEqual(len(patches), 512)

    def test_that_all_channels_are_returned(self):
        pixels3D = np.ones((70, 50, 3))
        patches = sliding_window(pixels3D, WindowSize(5, 5))
        for patch in patches:
            self.assertTrue(patch.shape == (5, 5, 3))
