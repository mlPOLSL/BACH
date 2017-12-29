from typing import Tuple
from logging import warning
from skimage.feature import hog
from data_types import GreyscaleImage
from collections import OrderedDict


def extract_hog_features(greyscale_image: GreyscaleImage, orientations=9,
                         pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                         block_norm='L2'):
    """
    A function extracting HOG features from the greyscale image.
    Please be careful with the parameters for big images, as the number of
    extracted features might be high. You may also consider applying PCA.
    For all the parameter descriptions please refer to:
    http://scikit-image.org/docs/0.13.x/api/skimage.feature.html?highlight=hog#skimage.feature.hog
    """
    if not isinstance(greyscale_image, GreyscaleImage):
        raise TypeError("Image should be an instance of GreyscaleImage")
    img_shape_y, img_shape_x = greyscale_image.shape
    positions_in_x = float(img_shape_x) / float(pixels_per_cell[1])
    positions_in_y = float(img_shape_y) / float(pixels_per_cell[0])
    cells_in_block = cells_per_block[0] * cells_per_block[1]
    number_of_features = orientations * cells_in_block * positions_in_x * \
                         positions_in_y
    warning("The number of hog features will "
            "be around: {}".format(number_of_features))

    hog_features = hog(greyscale_image, orientations, pixels_per_cell,
                       cells_per_block, block_norm)
    features = OrderedDict()
    for index, feature in enumerate(hog_features):
        name = 'hog_' + str(index)
        features[name] = feature
    return features
