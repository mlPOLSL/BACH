import unittest
import numpy as np
from Segmentation.kmeans_segmentation import segment_blue_nuclei
from data_types import SegmentedImage


class TestKMeansSegmentation(unittest.TestCase):
    def setUp(self):
        self.image = np.array([[[150., 143., 193.], [129., 125., 188],
                                [102., 101., 176], [90., 88., 170]],
                               [[111., 109., 180], [89., 86., 171],
                                [76., 70., 165],
                                [69., 59., 157]],
                               [[79., 74., 165], [69., 62., 159],
                                [58., 51., 151],
                                [67., 55., 156]],
                               [[62., 56., 151], [58., 46., 145],
                                [61., 49., 148],
                                [72., 54., 153]]])
        self.desired_segmentation = SegmentedImage(
            [[[0., 0., 0], [0., 0., 0], [0., 0., 0], [0., 0., 0]],
             [[0., 0., 0], [0., 0., 0], [0., 0., 0], [69., 59., 157]],
             [[0., 0., 0], [0., 0., 0], [58., 51., 151], [67., 55., 156]],
             [[62., 56., 151], [58., 46., 145], [61., 49., 148],
              [72., 54., 153]]])
        self.segmented = segment_blue_nuclei(self.image)

    def test_if_returns_correct_type(self):
        self.assertIsInstance(self.segmented, SegmentedImage)

    def test_if_segments_correct_part(self):
        correct = np.array_equal(self.segmented, self.desired_segmentation)
        self.assertEqual(correct, True)
