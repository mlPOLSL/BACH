import os
import numpy as np
from copy import copy
from skimage import io, img_as_float
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

BENIGN_LABEL = 0
INSITU_LABEL = 1
INVASIVE_LABEL = 2
NORMAL_LABEL = 3

BENIGN_WEIGHT = 0.4
NORMAL_WEIGHT = 0.4
INSITU_WEIGHT = 0.8
INVASIVE_WEIGHT = 1.0

LABELS_IMPORTANCE = [INVASIVE_LABEL, INSITU_LABEL, BENIGN_LABEL, NORMAL_LABEL]
LABELS = ['Benign', 'InSitu', 'Invsive', 'Normal']
WEIGHTS = [BENIGN_WEIGHT, INSITU_WEIGHT, INVASIVE_WEIGHT, NORMAL_WEIGHT]
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = [(300, 300), (200, 200), (100, 100)]
HOG_CELLS_PER_BLOCK = [(2, 2), (2, 2), (2, 2)]


def classify_image(path: str):
    clf = []
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_repaired_random_forest1x1.pkl'))
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_repaired_random_forest2x2.pkl'))
    clf.append(joblib.load(
        'C:\\Users\\user\PycharmProjects\BACH'
        '\\not_normalized_repaired_random_forest4x4.pkl'))
    labels = ['Benign', 'InSitu', 'Invsive', 'Normal']
    sums = [0., 0., 0., 0.]
    image = io.imread(path)
    print("segment")
    segmented = segment_blue_nuclei(image)
    image = img_as_float(image)
    grids = [divide_into_patches(image, x, x) for x in [1, 2, 4]]
    grids_segmented = [divide_into_patches(segmented, x, x) for x in
                       [1, 2, 4, 8]]

    print("classif")
    for index_grid, grid in enumerate(grids):
        for index_patch, patch in enumerate(grid):
            feature_dict = OrderedDict()
            greyscale = GreyscaleImage(copy(patch))
            feature_dict.update(get_wavelet_features(greyscale, "db1"))
            feature_dict.update(extract_color_features(HSVImage(copy(patch))))
            segmented_patch = SegmentedImage(
                grids_segmented[index_grid][index_patch])
            feature_dict.update(extract_shape_features(segmented_patch))
            feature_dict.update(extract_tamuras_features(greyscale))
            feature_dict.update(extract_hog_features(greyscale, HOG_ORIENTATIONS,
                                                     HOG_PIXELS_PER_CELL[index_grid],
                                                     HOG_CELLS_PER_BLOCK[index_grid]))
            features = [feature_dict[key] for key in feature_dict]
            response = clf[index_grid].predict(
                np.array(features).reshape((1, -1)))
            sums[int(response)] += WEIGHTS[int(response)]
    max_values_indexes = [index for index, val in enumerate(sums) if
                          val == max(sums)]
    if len(max_values_indexes) > 1:         # case of a draw
        for label in LABELS_IMPORTANCE:
            if label in max_values_indexes:
                final_prediction = label
                break
    else:
        final_prediction = max_values_indexes[0]
    print("{},{}".format(os.path.basename(path),labels[final_prediction]))

classify_image(
    "C:\\Users\\user\Desktop\ICIAR2018_BACH_Challenge\Photos\Invasive\\iv001"
    ".tif")