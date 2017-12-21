"""
Normalization, code is mostly a copy of sample provided by the authors in the
article "Classification of breast cancer histology images using Convolutional
Neural Networks"
(http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177544).
Slight changes were made to improve the readability.
"""

from __future__ import division
import numpy as np


def remove_invalid_vectors(optical_density_matrix: np.ndarray,
                           beta: float) -> np.ndarray:
    """
    Returns new array of only optical densities that are

    :param optical_density_matrix: array of vectors of optical densities
    :param beta: threshold when considering valid
     values in a vector (value must be greater that beta)
    :return: array with valid vectors
    """

    valid_vectors = np.logical_not((optical_density_matrix < beta).any(axis=1))
    return optical_density_matrix[valid_vectors, :]


def normalize_image(image_matrix: np.ndarray,
                    alpha: float = 1.0,
                    beta: float = 0.15):
    """

    :param image_matrix: image matrix where each cell represents vector of
    RGB values, these values should be of type float
    :param alpha: upper threshold for correct optical density
    :param beta: lower threshold for correct optical density
    :return: normalized image
    """

    Io = 240
    HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    (h, w, c) = np.shape(image_matrix)
    image_matrix = np.reshape(image_matrix, (h * w, c), order='F')
    optical_density = - np.log((image_matrix + 1) / Io)
    valid_optical_density = remove_invalid_vectors(optical_density, beta)
    (W, V) = np.linalg.eig(np.cov(valid_optical_density, rowvar=False))

    Vec = - np.transpose(np.array([V[:, 1], V[:, 0]]))
    That = np.dot(valid_optical_density, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, int(alpha))
    maxPhi = np.percentile(phi, 100 - int(alpha))
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = np.transpose(HE)
    Y = np.transpose(np.reshape(optical_density, (h * w, c)))

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(- np.dot(HERef, C))
    Inorm = np.reshape(np.transpose(Inorm), (h, w, c), order='F')
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)
    return Inorm
