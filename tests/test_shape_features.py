import unittest
import numpy as np
from FeatureExtraction.Shape.extract_shape_features import \
    extract_shape_features
from data_types import SegmentedImage


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.image = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [68, 54, 154], [0., 0., 0.]],
                               [[0., 0., 0.], [68, 54, 154], [68, 54, 154]]])
        self.blank_image = SegmentedImage(
            np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                      [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                      [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]))
        self.segmented_image = SegmentedImage(self.image)
        self.features = extract_shape_features(self.segmented_image, 1)

    def test_if_wrong_type_raises_exception(self):
        self.assertRaises(TypeError, extract_shape_features, self.image, 50)

    def test_if_blank_image_returns_zeros(self):
        features = extract_shape_features(self.blank_image)
        features_values = [features[key] for key in features]
        self.assertEqual(all([value == 0 for value in features_values]), True)

    def test_if_returns_correct_number_of_cells(self):
        self.assertEqual(self.features["number_of_cells"], 1)

    def test_if_returns_correct_fill_coefficient(self):
        self.assertAlmostEqual(self.features["fill_coefficient"], 0.33, 2)

    def test_if_returns_correct_min_area(self):
        self.assertEqual(self.features["min_area"], 3)

    def test_if_returns_correct_max_area(self):
        self.assertEqual(self.features["max_area"], 3)

    def test_if_returns_correct_mean_area(self):
        self.assertEqual(self.features["mean_area"], 3)

    def test_if_returns_correct_std_area(self):
        self.assertEqual(self.features["std_area"], 0)

    def test_if_returns_correct_min_perimeter(self):
        self.assertAlmostEqual(self.features["min_perimeter"], 3.41, 2)

    def test_if_returns_correct_max_perimeter(self):
        self.assertAlmostEqual(self.features["max_perimeter"], 3.41, 2)

    def test_if_returns_correct_mean_perimeter(self):
        self.assertAlmostEqual(self.features["mean_perimeter"], 3.41, 2)

    def test_if_returns_correct_std_perimeter(self):
        self.assertEqual(self.features["std_perimeter"], 0)

    def test_if_returns_correct_min_eccentricity(self):
        self.assertAlmostEqual(self.features["min_eccentricity"], 0.82, 2)

    def test_if_returns_correct_max_eccentricity(self):
        self.assertAlmostEqual(self.features["max_eccentricity"], 0.82, 2)

    def test_if_returns_correct_mean_eccentricity(self):
        self.assertAlmostEqual(self.features["mean_eccentricity"], 0.82, 2)

    def test_if_returns_correct_std_eccentricity(self):
        self.assertEqual(self.features["std_eccentricity"], 0)

    def test_if_returns_correct_min_solidity(self):
        self.assertAlmostEqual(self.features["min_solidity"], 0.82, 2)

    def test_if_returns_correct_max_solidity(self):
        self.assertAlmostEqual(self.features["max_solidity"], 0.82, 2)

    def test_if_returns_correct_mean_solidity(self):
        self.assertAlmostEqual(self.features["mean_solidity"], 0.82, 2)

    def test_if_returns_correct_std_solidity(self):
        self.assertEqual(self.features["std_solidity"], 0)
