from typing import List

import numpy as np

from DataPreprocessing.Grid.data_types import Patch, WindowSize, Stride


def divide_into_patches(image: np.ndarray, number_of_patches_in_x: int,
                        number_of_patches_in_y: int) -> List[Patch]:
    """
    Divides an image into equally shaped patches. The size of each patch is based
    on the number of patches in x and y that you provide. The patch has always
    the same number of channels as the original image.
    :param image: Pixel values of the image you want to divide
    :param number_of_patches_in_x: The number of patches you would like to obtain
    in single x dimension
    :param number_of_patches_in_y: The number of patches you would like to obtain
    in single y dimension
    :return: A list of patches, where the number of patches is obtained
    by multiplaying number of patches in x by number of patches in y
    """

    if not isinstance(number_of_patches_in_x, int) or not isinstance(
            number_of_patches_in_y, int):
        raise TypeError("number_of_patches_in_x and number_of_patches_in_y "
                        "should be integers")
    x_window_size, x_stride = divmod(image.shape[1], number_of_patches_in_x)
    y_window_size, y_stride = divmod(image.shape[0], number_of_patches_in_y)
    window_size = WindowSize(x_window_size + x_stride, y_window_size + y_stride)
    stride = Stride(x_window_size, y_window_size)
    patches = []
    for y_dim_patch_number in range(0, number_of_patches_in_y):
        for x_dim_patch_number in range(0, number_of_patches_in_x):
            left_border_x = int(0 + stride.x * x_dim_patch_number)
            right_border_x = int(window_size.x + stride.x * x_dim_patch_number)
            upper_border_y = int(0 + stride.y * y_dim_patch_number)
            lower_border_y = int(window_size.y + stride.y * y_dim_patch_number)
            patch = image[upper_border_y:lower_border_y,
                          left_border_x:right_border_x]
            patch = Patch(patch)
            patches.append(patch)
    return patches


def sliding_window(image: np.ndarray, window_size: WindowSize,
                   stride: Stride = 0) -> List[Patch]:
    """
    Apply sliding window to the image, which extract patches of the size provided
    as a window_size argument. If the stride argument is not calculated properly,
    there might be some loss of data. The patch has always the same number of
    channels as the original image.
    :param image: Pixels of the image you would like to apply sliding window to
    :param window_size: Size of the sliding window
    :param stride: The amount by which the window should be moved. If not provided,
    the stride will be equal to the window size so no there will be no overlap
    :return: A list of patches
    """
    if not isinstance(window_size, WindowSize):
        raise TypeError("window_size should be a WindowSize class instance")
    if not stride == 0 and not isinstance(stride, Stride):
        raise TypeError("stride should be a Stride class instance")
    if stride == 0:
        stride = Stride(window_size.x, window_size.y)

    number_of_patches_in_x = int(((image.shape[1] - window_size.x) / stride.x) + 1)
    number_of_patches_in_y = int(((image.shape[0] - window_size.y) / stride.y) + 1)
    patches = []
    for y_dim_patch_number in range(0, number_of_patches_in_y):
        for x_dim_patch_number in range(0, number_of_patches_in_x):
            left_border_x = int(0 + stride.x * x_dim_patch_number)
            right_border_x = int(window_size.x + stride.x * x_dim_patch_number)
            upper_border_y = int(0 + stride.y * y_dim_patch_number)
            lower_border_y = int(window_size.y + stride.y * y_dim_patch_number)
            patch = image[upper_border_y:lower_border_y,
                          left_border_x:right_border_x]
            patch = Patch(patch)
            patches.append(patch)
    return patches
