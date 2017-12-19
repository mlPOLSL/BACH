import unittest
import numpy as np
from FeatureExtraction.Wavelets.wavelets_features import get_wavelet_features, get_features_for_detail_coefficients
from data_types import GreyscaleImage

class TestGettingWaveletsFeatures(unittest.TestCase):
    def setUp(self):
        pixels = np.arange(0, 300).reshape((10, 10, 3))
        self.image = GreyscaleImage(pixels)

    def test_that_wrong_mother_wavelet_raise_exception(self):
        self.assertRaises(ValueError, get_wavelet_features, self.image, "wrong_wavelet")


class TestGettingDetailCoefficionsFeatures(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(0, 100)
        self.image.reshape((20, 5))

    def test_that_returns_correct_max(self):
        max, avg, kurtosis, skewness = get_features_for_detail_coefficients(self.image)
        self.assertEqual(max, 99)

    def test_that_returns_correct_avg(self):
        max, avg, kurtosis, skewness = get_features_for_detail_coefficients(self.image)
        self.assertEqual(avg, 49.5)

    def test_that_returns_correct_kurtosis(self):
        max, avg, kurtosis, skewness = get_features_for_detail_coefficients(self.image)
        self.assertAlmostEqual(kurtosis, -1.2002400240024003, 9)

    def test_that_returns_correct_skewness(self):
        max, avg, kurtosis, skewness = get_features_for_detail_coefficients(self.image)
        self.assertEqual(skewness, 0)
