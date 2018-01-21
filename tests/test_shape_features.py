import unittest
import numpy as np
from FeatureExtraction.Shape.extract_shape_features import \
    extract_shape_features
from data_types import SegmentedImage


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.image = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                               [[0, 0, 0], [68, 54, 154], [0, 0, 0]],
                               [[0, 0, 0], [68, 54, 154], [68, 54, 154]]])
        self.segmented_image = SegmentedImage(self.image)

    def test_if_wrong_type_raises_exception(self):
        self.assertRaises(TypeError, extract_shape_features, self.image, 50)
