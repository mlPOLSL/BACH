"""
Module extracting texture features based on Gray-Level Co-Occurrence Matrix.
The result of each function is a matrix, where the shape is equal to number of
distances and angles used to obtain the GLCM matrix. If GLCM was formed using
two distances and three angles, the shape of returned matrix will be (2, 3).
Functions which are not using already implemented in skimage feature extractions,
were implemented using the following paper:
https://doi.org/10.1371/journal.pone.0102107.s001
"""

import numpy as np
from math import pow, log, sqrt, exp, degrees
from copy import copy
from skimage.feature import greycomatrix, greycoprops
from data_types import NumpyImageUINT8, GLCM
from typing import List, Dict


def construct_glcm(image: NumpyImageUINT8, distances: List[int],
                   angles: List[float]) -> GLCM:
    if not isinstance(image, NumpyImageUINT8):
        raise TypeError("image should be an instance of NumpyImageUINT8")
    glcm = greycomatrix(image, distances=distances, angles=angles, normed=True,
                        levels=256)
    return GLCM(glcm, distances, angles)


def calculate_contrast(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    return greycoprops(glcm, 'contrast')


def calculate_dissimilarity(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    return greycoprops(glcm, 'dissimilarity')


def calculate_homogeneity(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    return greycoprops(glcm, 'homogeneity')


def calculate_correlation(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    return greycoprops(glcm, 'correlation')


def calculate_energy(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    return greycoprops(glcm, 'energy')


def calculate_autocorrelation(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            x_indexes = np.linspace(1, x_shape, x_shape)
            y_indexes = np.linspace(1, y_shape, y_shape)
            xv, yv = np.meshgrid(x_indexes, y_indexes)
            result = np.sum(
                np.multiply(glcm[:, :, distance, angle], np.multiply(xv, yv)))
            results[distance, angle] = result
    return results


def calculate_cluster_prominence(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            for i in range(0, y_shape):
                row_mean = np.mean(glcm[i, :])
                for j in range(0, x_shape):
                    column_mean = np.mean(glcm[:, j])
                    result += pow(i + 1 + j + 1 - row_mean - column_mean, 4) * \
                              glcm[i, j, distance, angle]
            results[distance, angle] = copy(result)
    return results


def calculate_cluster_shade(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            for i in range(0, y_shape):
                row_mean = np.mean(glcm[i, :])
                for j in range(0, x_shape):
                    column_mean = np.mean(glcm[:, j])
                    result += pow(i + 1 + j + 1 - row_mean - column_mean, 3) * \
                              glcm[i, j, distance, angle]
            results[distance, angle] = copy(result)
    return results


def calculate_cluster_tendency(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            for i in range(0, y_shape):
                row_mean = np.mean(glcm[i, :, distance, angle])
                for j in range(0, x_shape):
                    column_mean = np.mean(glcm[:, j, distance, angle])
                    result += pow(i + 1 + j + 1 - row_mean - column_mean, 2) * \
                              glcm[i, j, distance, angle]
            results[distance, angle] = copy(result)
    return results


def calculate_difference_entropy(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    x_indexes = np.linspace(1, x_shape, x_shape)
    y_indexes = np.linspace(1, y_shape, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            single_glcm = glcm[:, :, distance, angle].reshape(
                (y_shape, x_shape))
            for level in range(0, glcm.levels):
                diagonals = np.sum(single_glcm[abs(rows - cols) == level])
                try:
                    result += diagonals * log(diagonals, 2)
                except ValueError:
                    pass
            results[distance, angle] = copy(result)
    return results


def change_undefinied_to_zeros(matrix: np.ndarray):
    if matrix.ndim != 2:
        raise ValueError("The matrix should be two dimensional")
    matrix[matrix < 0] = 0
    return matrix


def calculate_entropy(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            log2_matrix = np.log2(glcm[:, :, distance, angle])
            log2_matrix[np.isinf(log2_matrix)] = 0
            result = np.sum(
                np.multiply(glcm[:, :, distance, angle], log2_matrix))
            results[distance, angle] = -copy(result)
    return results


def calculate_HXY2(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            for i in range(0, x_shape):
                row_sum = np.sum(glcm[i, :, distance, angle])
                for j in range(0, y_shape):
                    column_sum = np.sum(glcm[:, j, distance, angle])
                    row_col_product = row_sum * column_sum
                    if row_col_product != 0:
                        result += row_col_product * np.log(row_col_product)
            results[distance, angle] = -copy(result)
    return results


def calculate_IMC2(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    entropy = calculate_entropy(glcm)
    hxy2 = calculate_HXY2(glcm)
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = sqrt(
                1 - exp(-2. * (entropy[distance, angle] - hxy2[distance, angle])))
            results[distance, angle] = copy(result)
    return results


def calculate_IDMN(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    x_indexes = np.linspace(0, x_shape - 1, x_shape)
    y_indexes = np.linspace(0, y_shape - 1, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            single_glcm = glcm[:, :, distance, angle].reshape(x_shape, y_shape)
            denominator = np.subtract(rows, cols)
            denominator = np.power(denominator, 2)
            denominator = np.divide(denominator, pow(glcm.levels, 2))
            denominator = np.add(denominator, 1)
            results[distance, angle] = np.sum(
                np.divide(single_glcm, denominator))
    return results


def calculate_IDN(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    x_indexes = np.linspace(0, x_shape - 1, x_shape)
    y_indexes = np.linspace(0, y_shape - 1, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            single_glcm = glcm[:, :, distance, angle].reshape(y_shape, x_shape)
            denominator = np.absolute(np.subtract(rows, cols))
            denominator = np.divide(denominator, glcm.levels)
            denominator = np.add(denominator, 1)
            results[distance, angle] = np.sum(
                np.divide(single_glcm, denominator))
    return results


def calculate_max_proba(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    for distance in range(0, distances):
        for angle in range(0, angles):
            results[distance, angle] = np.amax(glcm[:, :, distance, angle])
    return results


def calculate_sum_average(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    x_indexes = np.linspace(1, x_shape, x_shape)
    y_indexes = np.linspace(1, y_shape, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            single_glcm = glcm[:, :, distance, angle].reshape(
                (y_shape, x_shape))
            for i in range(2, 2 * glcm.levels + 1):
                result += i * np.sum(single_glcm[cols + rows == i])
            results[distance, angle] = copy(result)
    return results


def calculate_sum_entropy(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    x_indexes = np.linspace(1, x_shape, x_shape)
    y_indexes = np.linspace(1, y_shape, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            single_glcm = glcm[:, :, distance, angle].reshape(
                (y_shape, x_shape))
            for i in range(2, 2 * glcm.levels + 1):
                diagonals = np.sum(single_glcm[cols + rows == i])
                if diagonals != 0:
                    result += diagonals * np.log2(diagonals)
            results[distance, angle] = -copy(result)
    return results


def calculate_sum_variance(glcm: GLCM):
    if not isinstance(glcm, GLCM):
        raise TypeError("glcm should be an instance of GLCM class")
    y_shape, x_shape = glcm.shape[0:2]
    distances = glcm.shape[2]
    angles = glcm.shape[3]
    results = np.zeros((distances, angles))
    x_indexes = np.linspace(1, x_shape, x_shape)
    y_indexes = np.linspace(1, y_shape, y_shape)
    cols, rows = np.meshgrid(x_indexes, y_indexes)
    sum_entropy = calculate_sum_entropy(glcm)
    for distance in range(0, distances):
        for angle in range(0, angles):
            result = 0
            single_glcm = glcm[:, :, distance, angle].reshape(
                (y_shape, x_shape))
            for i in range(2, 2 * glcm.levels + 1):
                result += pow(i - sum_entropy[distance, angle], 2) * np.sum(
                    single_glcm[cols + rows == i])
            results[distance, angle] = result
    return results


def add_features_to_dict(dictionary: Dict[str, float], features: np.ndarray,
                         feature_name: str, distances: List[int],
                         angles: List[float]):
    distances_shape, angles_shape = features.shape
    for i in range(0, distances_shape):
        for j in range(0, angles_shape):
            name = feature_name + "_distance" + str(
                distances[i]) + "_angle" + str(int(degrees(angles[j])))
            dictionary[name] = copy(features[i, j])
    return dictionary


def extract_texture_features(image: NumpyImageUINT8, distances: List[int],
                             angles: List[float]):
    if not isinstance(image, NumpyImageUINT8):
        raise TypeError("image should be an instance of NumpyImageUINT8")
    glcm = construct_glcm(image, distances, angles)
    features = {}
    contrast = calculate_contrast(glcm)
    features = add_features_to_dict(features, contrast, 'contrast', distances,
                                    angles)
    dissimilarity = calculate_dissimilarity(glcm)
    features = add_features_to_dict(features, dissimilarity, 'dissimilarity',
                                    distances, angles)
    homogeneity = calculate_homogeneity(glcm)
    features = add_features_to_dict(features, homogeneity, 'homogeneity',
                                    distances, angles)
    correlation = calculate_correlation(glcm)
    features = add_features_to_dict(features, correlation, 'correlation',
                                    distances, angles)
    energy = calculate_energy(glcm)
    features = add_features_to_dict(features, energy, 'energy', distances,
                                    angles)
    autocorrelation = calculate_autocorrelation(glcm)
    features = add_features_to_dict(features, autocorrelation,
                                    'autocorrelation', distances, angles)
    cluster_prominence = calculate_cluster_prominence(glcm)
    features = add_features_to_dict(features, cluster_prominence,
                                    'cluster_prominence', distances, angles)
    cluster_tendency = calculate_cluster_tendency(glcm)
    features = add_features_to_dict(features, cluster_tendency,
                                    'cluster_tendency', distances, angles)
    cluster_shade = calculate_cluster_shade(glcm)
    features = add_features_to_dict(features, cluster_shade, 'cluster_shade',
                                    distances, angles)
    difference_entropy = calculate_difference_entropy(glcm)
    features = add_features_to_dict(features, difference_entropy,
                                    'difference_entropy', distances, angles)
    entropy = calculate_entropy(glcm)
    features = add_features_to_dict(features, entropy, 'entropy', distances,
                                    angles)
    imc2 = calculate_IMC2(glcm)
    features = add_features_to_dict(features, imc2, 'IMC2', distances, angles)
    idmn = calculate_IDMN(glcm)
    features = add_features_to_dict(features, idmn, 'IDMN', distances, angles)
    idn = calculate_IDN(glcm)
    features = add_features_to_dict(features, idn, 'IDN', distances, angles)
    max_proba = calculate_max_proba(glcm)
    features = add_features_to_dict(features, max_proba, 'max_proba', distances,
                                    angles)
    sum_average = calculate_sum_average(glcm)
    features = add_features_to_dict(features, sum_average, 'sum_average',
                                    distances, angles)
    sum_variance = calculate_sum_variance(glcm)
    features = add_features_to_dict(features, sum_variance, 'sum_variance',
                                    distances, angles)
    sum_entropy = calculate_sum_entropy(glcm)
    features = add_features_to_dict(features, sum_entropy, 'sum_entropy',
                                    distances, angles)
    return features
