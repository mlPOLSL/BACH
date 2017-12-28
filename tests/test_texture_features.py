import unittest
import numpy as np
from data_types import GLCM
from FeatureExtraction.Texture.texture_features import calculate_autocorrelation, \
    calculate_cluster_prominence, calculate_cluster_shade, \
    calculate_cluster_tendency, calculate_difference_entropy, calculate_entropy, \
    calculate_IMC2, calculate_IDMN, calculate_IDN, calculate_max_proba, \
    calculate_sum_average, calculate_sum_entropy, calculate_sum_variance


class TestTextureFeatures(unittest.TestCase):
    def setUp(self):
        self.first_test_glcm = GLCM(np.array([[1, 2, 3],
                                              [4, 5, 6],
                                              [7, 8, 9]]).reshape((3, 3, 1, 1)))
        self.first_test_glcm.levels = 3
        self.second_test_glcm = GLCM(np.array([[0.5, 0.7],
                                               [0.2, 0.1]]).reshape(
            (2, 2, 1, 1)))
        self.second_test_glcm.levels = 2
        self.third_test_glcm = GLCM(np.array([[1, 2],
                                              [3, 4]]).reshape((2, 2, 1, 1)))
        self.third_test_glcm.levels = 2

    def test_autocorrelation(self):
        result = calculate_autocorrelation(self.first_test_glcm)
        self.assertEqual(228, result[0, 0])

    def test_cluster_prominence(self):
        result = calculate_cluster_prominence(self.first_test_glcm)
        self.assertEqual(119280, result[0, 0])

    def test_cluster_shade(self):
        result = calculate_cluster_shade(self.first_test_glcm)
        self.assertEqual(-15912, result[0, 0])

    def test_cluster_tendency(self):
        result = calculate_cluster_tendency(self.first_test_glcm)
        self.assertEqual(2172, result[0, 0])

    def test_difference_entropy(self):
        result = calculate_difference_entropy(self.first_test_glcm)
        self.assertAlmostEqual(178.26120178, result[0, 0])

    def test_entropy(self):
        result = calculate_entropy(self.second_test_glcm)
        self.assertAlmostEqual(1.6567796494470395, result[0, 0])

    def test_IMC2(self):
        result = calculate_IMC2(self.second_test_glcm)
        self.assertAlmostEqual(0.89356725105, result[0, 0])

    def test_IDMN(self):
        result = calculate_IDMN(self.second_test_glcm)
        self.assertAlmostEqual(1.32, result[0, 0])

    def test_IDN(self):
        result = calculate_IDN(self.second_test_glcm)
        self.assertAlmostEqual(1.2, result[0,0])

    def test_sum_entropy(self):
        result = calculate_sum_entropy(self.third_test_glcm)
        self.assertAlmostEqual(-19.609640474436, result[0,0])

    def test_sum_average(self):
        result = calculate_sum_average(self.third_test_glcm)
        self.assertEqual(33, result[0, 0])

    def test_sum_variance(self):
        result = calculate_sum_variance(self.third_test_glcm)
        self.assertAlmostEqual(5252.61626667, result[0,0])

    def test_max_proba(self):
        result = calculate_max_proba(self.first_test_glcm)
        self.assertEqual(9, result[0,0])
