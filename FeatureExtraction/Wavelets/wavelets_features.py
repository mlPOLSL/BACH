from collections import OrderedDict
import numpy as np
from scipy import stats
import pywt
from data_types import GreyscaleImage


def get_wavelet_features(image: GreyscaleImage, mother_wavelet: str) -> dict:
    """
    Extracts wavelet features from an image in a greyscale. For each detail
    coefficients matrices (cH, cV, cD) it extracts:
    maximal value,
    average value,
    kurtosis,
    skewness

    :param image: Pixel values of the image from which you want to extract
        features
    :param mother_wavelet: Wavelet type to perform Discrete Wavelet Transform
    :return: Dictionary with features extracted from an image.
    """
    if not isinstance(image, GreyscaleImage):
        raise TypeError(
            "Image should be in greyscale, use GreyscaleImage type")
    coeffs = pywt.dwt2(image, mother_wavelet)
    cA, (cH, cV, cD) = coeffs
    features_dict = OrderedDict()
    for detail_coefficients, name in zip([cH, cV, cD], ["cH", "cV", "cD"]):
        detail_features = get_features_for_detail_coefficients(
            detail_coefficients, mother_wavelet, name)
        features_dict = {**features_dict, **detail_features}
    return features_dict


def get_features_for_detail_coefficients(
        detail_coefficients: np.ndarray, mother_wavelet: str,
        detail_name: str) -> OrderedDict:
    """
    Extracts wavelet coefficients from a single detail coefficients matrix.
    :param detail_coefficients: Single detail coefficients matrix taken from
        Discrete Wavelet Transform
    :param mother_wavelet: Name of the mother wavelet.
    :param detail_name: Name of the detail matrix.
    :return: Ordered dict of the features: max, avg, kurtosis, skewness, sum
    """
    key_prefix = mother_wavelet + "_" + detail_name + "_"
    detail_coefficients_flattened = detail_coefficients.flatten()
    detail_features = OrderedDict()
    detail_features[key_prefix + "max"] = detail_coefficients_flattened.max()
    detail_features[key_prefix + "avg"] = detail_coefficients_flattened.mean()
    detail_features[key_prefix + "kurtosis"] = stats.kurtosis(
        detail_coefficients_flattened)
    detail_features[key_prefix + "skewness"] = stats.skew(
        detail_coefficients_flattened)
    detail_features[key_prefix + "sum"] = np.sum(detail_coefficients_flattened)
    detail_features[key_prefix + "energy_distance"] = stats.entropy(
        detail_coefficients_flattened)

    return detail_features
