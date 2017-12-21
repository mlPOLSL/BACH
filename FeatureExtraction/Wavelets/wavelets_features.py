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
    features_dict = {}
    for detail_coefficients, name in zip([cH, cV, cD], ["cH", "cV", "cD"]):
        max_value, avg_value, kurtosis, skewness = get_features_for_detail_coefficients(
            detail_coefficients)
        features = {mother_wavelet + "_" + name + "_max": max_value,
                    mother_wavelet + "_" + name + "_avg": avg_value,
                    mother_wavelet + "_" + name + "_kurtosis": kurtosis,
                    mother_wavelet + "_" + name + "_skewness": skewness}
        features_dict = {**features_dict, **features}
    return features_dict


def get_features_for_detail_coefficients(
        detail_coefficients: np.ndarray) -> dict:
    """
    Extracts wavelet coefficients from a single detail coefficients matrix.
    :param detail_coefficients: Single detail coefficients matrix taken from
        Discrete Wavelet Transform
    :return: Values of the features: max, avg, kurtosis, skewness
    """
    detail_coefficients_flattened = detail_coefficients.flatten()
    max_value = detail_coefficients_flattened.max()
    avg_value = detail_coefficients_flattened.mean()
    kurtosis = stats.kurtosis(detail_coefficients_flattened)
    skewness = stats.skew(detail_coefficients_flattened)
    return max_value, avg_value, kurtosis, skewness
