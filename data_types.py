import numpy as np
from typing import List
from skimage import img_as_ubyte, color

UINT8_UNIQUE_VALUES = 256


class Patch(np.ndarray):
    def __new__(cls, pixels: np.ndarray):
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
        if pixels.size == 0:
            raise ValueError("Patch cannot be empty")
        return np.array(pixels).view(Patch)


class WindowSize(object):
    def __init__(self, x: int, y: int):
        if not x > 0 or not y > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x, y))
        elif not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(type(x),
                                                                      type(y)))
        self.x = x
        self.y = y


class Stride(object):
    def __init__(self, x_stride: int, y_stride: int):
        if not x_stride > 0 or not y_stride > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x_stride,
                                                                  y_stride))
        elif not isinstance(x_stride, int) or not isinstance(y_stride, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(
                    type(x_stride),
                    type(y_stride)))
        self.x = x_stride
        self.y = y_stride


class NumpyImageUINT8(np.ndarray):
    def __new__(cls, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be provided as a numpy array")
        if image.ndim > 1:
            image = color.rgb2gray(image)
        if image.dtype != np.uint8:
            image = img_as_ubyte(image)
        return image.view(NumpyImageUINT8)


class GLCM(np.ndarray):
    """
    Class representing Grey-Level Co-occurrence Matrices. Third and fourth
    dimension represent matrices for provided distances and angles respectively.
    """
    def __new__(cls, matrix: np.ndarray, distances: List[int] = 0,
                angles: List[float] = 0):
        if matrix.ndim != 4:
            raise ValueError(
                "The GLCM should have 4 dimensions, not".format(matrix.ndim))
        cls.levels = UINT8_UNIQUE_VALUES
        cls.distances = distances
        cls.angles = angles
        return matrix.view(GLCM)

      
class GreyscaleImage(np.ndarray):
    def __new__(cls, pixels: np.ndarray):
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
        if pixels.size == 0:
            raise ValueError("Image cannot be empty")
        return color.rgb2grey(pixels).view(GreyscaleImage)

class HSVImage(np.ndarray):
    def __new__(cls, rgb_image: np.ndarray):
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)
        if rgb_image.size == 0:
            raise ValueError("Image cannot be empty")
        return color.rgb2hsv(rgb_image)
