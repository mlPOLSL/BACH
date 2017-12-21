import unittest
import numpy as np
from data_types import Patch, WindowSize, Stride, GreyscaleImage


class TestPatch(unittest.TestCase):
    def test_if_will_be_converted_to_numpy(self):
        patch = Patch([[1, 2, 3], [1, 2, 3]])
        self.assertIsInstance(patch, Patch)

    def test_if_cannot_be_empty(self):
        with self.assertRaises(ValueError):
            Patch([])


class TestWindowSize(unittest.TestCase):
    def test_that_x_has_to_be_positive(self):
        with self.assertRaises(ValueError):
            WindowSize(-1, 1)

    def test_that_y_has_to_be_positive(self):
        with self.assertRaises(ValueError):
            WindowSize(1, -1)

    def test_that_x_and_y_have_to_be_integers(self):
        with self.assertRaises(TypeError):
            WindowSize(1.2, 1.5)


class TestStride(unittest.TestCase):
    def test_that_x_has_to_be_positive(self):
        with self.assertRaises(ValueError):
            Stride(-1, 1)

    def test_that_y_has_to_be_positive(self):
        with self.assertRaises(ValueError):
            Stride(1, -1)

    def test_that_x_and_y_have_to_be_integers(self):
        with self.assertRaises(TypeError):
            Stride(1.2, 1.5)


class TestGreyscaleImage(unittest.TestCase):
    def test_if_will_be_converted_to_numpy(self):
        image = GreyscaleImage([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
        self.assertIsInstance(image, np.ndarray)

    def test_if_cannot_be_empty(self):
        with self.assertRaises(ValueError):
            GreyscaleImage([])

    def test_if_will_be_converted_to_grayscale_shape(self):
        image = GreyscaleImage([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
        self.assertTrue(image.shape == (3, 3))

    def test_if_will_be_converted_to_grayscale_values(self):
        image = GreyscaleImage([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
        self.assertTrue(all(0.0 <= x <= 1.0 for x in image.flatten()))
