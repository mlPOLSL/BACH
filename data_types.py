import numpy as np
from skimage import color


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


class GreyscaleImage(np.ndarray):
    def __new__(cls, pixels: np.ndarray):
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
        if pixels.size == 0:
            raise ValueError("Image cannot be empty")
        return color.rgb2grey(pixels).view(GreyscaleImage)
