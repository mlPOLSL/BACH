import unittest
import numpy as np
from Segmentation.kmeans_segmentation import segment_blue_nuclei
from data_types import SegmentedImage


class TestShapeFeatures(unittest.TestCase):
    def setUp(self):
        self.image = np.array(
            [[[193., 11., 209.], [193., 114., 209.], [255., 255., 255.]],
             [[255., 215., 255.], [68., 54., 154.], [255., 255., 255.]],
             [[255., 251., 255.], [158, 54., 154.], [68., 233., 124.]]])
        self.segmented = segment_blue_nuclei(self.image)

    def test_if_returns_correct_type(self):
        self.assertIsInstance(self.segmented, SegmentedImage)
