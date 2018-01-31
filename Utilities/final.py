
import numpy as np
from sklearn.externals import joblib
from collections import OrderedDict
from DataPreprocessing.Grid.grid import divide_into_patches
from FeatureExtraction.Color.color_features import extract_color_features
from FeatureExtraction.Texture.tamuras_features import extract_tamuras_features
from FeatureExtraction.Shape.extract_shape_features import \
    extract_shape_features
from FeatureExtraction.Wavelets.wavelets_features import get_wavelet_features
from FeatureExtraction.hog_features import extract_hog_features
from Segmentation.kmeans_segmentation import segment_blue_nuclei
from data_types import GreyscaleImage, SegmentedImage, HSVImage
from skimage import io, img_as_float
from copy import copy
import json
def classify_image(path: str):
    clf = []
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_random_forest_1x1.pkl'))
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_random_forest_2x2.pkl'))
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_random_forest_4x4.pkl'))

    weights = [0.4, 0.8, 1., 0.4]
    sums = [0., 0., 0., 0.]
    image = io.imread(path)
    print("segment")
    segmented = segment_blue_nuclei(image)
    image = img_as_float(image)
    grids = [divide_into_patches(image, x, x) for x in [1, 2, 4]]
    grids_segmented = [divide_into_patches(segmented, x, x) for x in
                       [1, 2, 4, 8]]

    hog_pixels_per_cell = [(300, 300), (200, 200), (100, 100)]
    hog_cells_per_block = [(2, 2), (2, 2), (2, 2)]
    print("classif")
    for index_grid, grid in enumerate(grids):
        for index_patch, patch in enumerate(grid):
            feature_dict = OrderedDict()
            greyscale = GreyscaleImage(copy(patch))
            feature_dict.update(get_wavelet_features(greyscale,
                                                     "db1"))
            feature_dict.update(extract_color_features(HSVImage(copy(patch))))
            segmented_patch = SegmentedImage(
                grids_segmented[index_grid][index_patch])
            feature_dict.update(extract_shape_features(segmented_patch))
            feature_dict.update(extract_tamuras_features(greyscale))
            hogs = extract_hog_features(greyscale, 9, hog_pixels_per_cell[
                                                         index_grid],
                                                     hog_cells_per_block[
                                                         index_grid])
            hog_len = len(hogs)
            print(hog_len)
            feature_dict.update(hogs)
            features = [feature_dict[key] for key in feature_dict]
            response = clf[index_grid].predict(
                np.array(features).reshape((1, -1)))
            print(response)
            sums[int(response)] += weights[int(response)]


# classify_image(
#     "C:\\Users\\user\PycharmProjects\BACH\\b001.tif")