import unittest
from collections import OrderedDict
import numpy as np
from FeatureExtraction.Wavelets.wavelets_features import get_wavelet_features, \
    get_features_for_detail_coefficients
from data_types import GreyscaleImage


class TestGettingWaveletsFeatures(unittest.TestCase):
    def setUp(self):
        pixels = np.arange(0, 300).reshape((10, 10, 3))
        self.image = GreyscaleImage(pixels)

    def test_that_wrong_mother_wavelet_raise_exception(self):
        self.assertRaises(ValueError, get_wavelet_features, self.image,
                          "wrong_wavelet")

    def test_that_returns_ordered_dict(self):
        feature_dict = get_wavelet_features(self.image, "haar")
        self.assertIsInstance(feature_dict, OrderedDict)


class TestGettingDetailCoefficionsFeatures(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(0, 100)
        self.image.reshape((20, 5))

    def test_that_returns_correct_max(self):
        features = get_features_for_detail_coefficients(
            self.image, "db1", "cA")
        self.assertEqual(features["db1_cA_max"], 99)

    def test_that_returns_correct_avg(self):
        features = get_features_for_detail_coefficients(
            self.image, "db1", "cA")
        self.assertEqual(features["db1_cA_avg"], 49.5)

    def test_that_returns_correct_kurtosis(self):
        features = get_features_for_detail_coefficients(
            self.image, "db1", "cA")
        self.assertEqual(features["db1_cA_kurtosis"], -1.2002400240024003, 9)

    def test_that_returns_correct_skewness(self):
        features = get_features_for_detail_coefficients(
            self.image, "db1", "cA")
        self.assertEqual(features["db1_cA_skewness"], 0)

    def test_that_returns_correct_sum(self):
        features = get_features_for_detail_coefficients(
            self.image, "db1", "cA")
        self.assertEqual(features["db1_cA_sum"], 4950)

    def test_that_returns_ordered_dict(self):
        feature_dict = get_features_for_detail_coefficients(self.image, "db1",
                                                            "cA")
        self.assertIsInstance(feature_dict, OrderedDict)
