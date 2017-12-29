"""
Module for extraction of color features using HSV (hue, saturation, value)
model of color representation.
]"""

from typing import Tuple
from collections import OrderedDict
from data_types import HSVImage

HSV_COMPONENTS_AMOUNT = 3
H_INDEX = 0
S_INDEX = 1
V_INDEX = 2
COLUMN_AXIS = 0

def color_standard_deviation(image: HSVImage) -> Tuple[float, float, float]:
    """
    Calculates standard deviations for each component of HSV in the picture.

    :param image: image matrix where each cell is a HSV pixel
    :return: Tuple with each component standard deviation
    """

    pixels_amount = image.shape[0] * image.shape[1]
    pixels_as_one_dimensional_array = image.reshape(pixels_amount,
                                                    HSV_COMPONENTS_AMOUNT)
    (h_mean, s_mean, v_mean) = pixels_as_one_dimensional_array.std(COLUMN_AXIS)
    return (h_mean, s_mean, v_mean)


def color_mean(image: HSVImage) -> Tuple[float, float, float]:
    """
    Calculates mean for each component of HSV in the picture.

    :param image: image matrix where each cell is a HSV pixel
    :return: Tuple with each component mean
    """

    pixels_amount = image.shape[0] * image.shape[1]
    pixels_as_one_dimensional_array = image.reshape(pixels_amount,
                                                    HSV_COMPONENTS_AMOUNT)
    (h_mean, s_mean, v_mean) = pixels_as_one_dimensional_array.mean(COLUMN_AXIS)
    return (h_mean, s_mean, v_mean)

def extract_color_features(image: HSVImage) -> OrderedDict:
    """
    Extracts two color features for each component of HSV namely mean and
     standard deviation giving in total six features.

    :param image: image represented as 2D array where each cell represents a HSV
    pixel.
    :return: Dictionary with two keys, mean and std.
    """
    (mean_h, mean_s, mean_v) = color_mean(image)
    (std_h, std_s, std_v) = color_standard_deviation(image)
    return OrderedDict([("color_mean_h", mean_h), ("color_mean_s", mean_s),
                        ("color_mean_v", mean_v), ("color_std_h", std_h),
                        ("color_std_s", std_s), ("color_std_v", std_v)])
