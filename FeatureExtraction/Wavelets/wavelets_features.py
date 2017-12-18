import math
import numpy as np
from scipy import stats
from skimage import color, io
import pywt


def get_wavelet_features(image: np.ndarray, mother_wavelet: str) -> dict:
    if mother_wavelet not in ["haar", "db", "sym", "coif", "bior", "rbio", "dmey", "gaus", "mexh", "morl", "cgau",
                              "shan", "fbsp", "cmor"]:
        raise ValueError("Given mother wavelet \"{}\" is not supported".format(mother_wavelet))
    greyscale_image = color.rgb2grey(image)
    coeffs = pywt.dwt2(greyscale_image, mother_wavelet)
    cA, (cH, cV, cD) = coeffs
    features_dict = {}
    for detail_coefficients, name in zip([cH, cV, cD], ["cH", "cV", "cD"]):
        max, avg, kurtosis, skewness = get_features_for_detail_coefficients(detail_coefficients)
        features = dict(
            {name + "_max": max, name + "_avg": avg, name + "_kurtosis": kurtosis, name + "_skewness": skewness})
        features_dict = {**features_dict, **features}
    return features_dict


def get_features_for_detail_coefficients(detail_coefficients: np.ndarray) -> dict:
    detail_coefficients_flattened = detail_coefficients.flatten()
    max = detail_coefficients_flattened.max()
    avg = detail_coefficients_flattened.mean()
    kurtosis = stats.kurtosis(detail_coefficients_flattened)
    skewness = stats.skew(detail_coefficients_flattened)
    return max, avg, kurtosis, skewness


img = io.imread("/Users/apple/PycharmProjects/BACH_local/Dataset/breasthistology/Training_data/InSitu/norm/normt43.tif")
print(get_wavelet_features(img, "haar"))
