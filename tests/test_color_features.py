import unittest
from data_types import HSVImage
from FeatureExtraction.Color.color_features import color_standard_deviation, \
    color_mean, extract_color_features


class TestColorFeatures(unittest.TestCase):
    def setUp(self):
        self.image = HSVImage([[[1, 2, 3], [4, 5, 6]]])

    def test_color_standard_deviation(self):
        self.assertEqual(color_standard_deviation(self.image), (0.0,
                                                                0.13186813186813184,
                                                                6.9849193112423921e-10))

    def test_color_mean(self):
        self.assertEqual(color_mean(self.image),
                         (0.58333333333333337,
                          0.43956043956043955,
                          2.3283064370807974e-09))

    def test_extract_color_features(self):
        self.assertEqual(extract_color_features(self.image),
                         {
                             "color_mean_h": 0.58333333333333337,
                             "color_mean_s": 0.43956043956043955,
                             "color_mean_v": 2.3283064370807974e-09,
                             "color_std_h": 0.0,
                             "color_std_s": 0.13186813186813184,
                             "color_std_v": 6.9849193112423921e-10
                         })
