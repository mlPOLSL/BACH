from typing import List, Tuple
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import OrderedDict
import random
from copy import copy
import os
import json
import numpy as np
from data_types import ImagePatchFeatures

BENIGN_LABEL = 0
INSITU_LABEL = 1
INVASIVE_LABEL = 2
NORMAL_LABEL = 3

LABELS_IMPORTANCE = [INVASIVE_LABEL, INSITU_LABEL, BENIGN_LABEL, NORMAL_LABEL]

def get_features_and_label_from_file(file_path: str) -> (int, List[float]):
    """
    Read features and label from .json file.

    :param file_path: path to .json file.
    :return: Tuple[int, List[float]]
    """

    with open(file_path) as json_file:
        image_dict = json.load(json_file)
        label = image_dict["label"]
        features = image_dict["features"]
        return label, [feature for feature in features.values()]


def get_paths_to_features_files(features_root_directory: str) -> List[str]:
    """
    Given a root path walks through it and all its subdirectories and saves
    paths to all files that end with .json extension.

    :param features_root_directory: path to root directory.
    :return: list of paths to json files containing label and features of image
    patch.
    """

    features_files_path = []
    for root, directories, files in os.walk(features_root_directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root,file)
                features_files_path.append(file_path)
    return features_files_path


def load_images_features(paths_to_feature_files: List[str]) -> List[ImagePatchFeatures]:
    """
    Load features of patches grouped into images.
    :param paths_to_feature_files:
    :return: A list of two dimensional matrices where features of patches
    are grouped into corresponding images
    """
    patch_labels = []
    image_labels = []
    features = []
    for path in paths_to_feature_files:
        image_id = os.path.basename(os.path.dirname(path))
        label, feature_list = get_features_and_label_from_file(path)
        feature_list = [feature_list[1][key] for key in feature_list[1]]
        patch_labels.append(label)
        if any(image_id == image_features.image_id for image_features in
               features):
            for index, image_features in enumerate(features):
                if image_features.image_id == image_id:
                    features[index] = ImagePatchFeatures(np.vstack(
                        (image_features, np.array(feature_list))), image_id,
                        label)
                    break
        else:
            features.append(ImagePatchFeatures(np.array(feature_list),
                                               image_id, label))
            image_labels.append(label)
    return features, patch_labels, image_labels


def get_features_only(images_features):
    features = []
    for image in images_features:
        for patch_features in image:
            features.append(patch_features)
    return features


def get_labels_only(images_features):
    labels = []
    for image in images_features:
        for features in image:
            labels.append(features.label)
    return labels


def predict(test_x, clf):
    features = get_features_only(test_x)
    # features = normalize(features, axis=0)
    predictions = clf.predict(features)
    return predictions


def score(test_x, predictions):
    predictions_index = 0
    no_of_predictions = len(test_x)
    correct_predictions = 0.0
    for image in test_x:
        normal = 0
        benign = 0
        invasive = 0
        insitu = 0
        for patch in image:
            if predictions[predictions_index] == BENIGN_LABEL:
                benign += 1
            elif predictions[predictions_index] == NORMAL_LABEL:
                normal += 1
            elif predictions[predictions_index] == INSITU_LABEL:
                insitu += 1
            else:
                invasive += 1
            predictions_index += 1
        labels = [benign, insitu, invasive, normal]
        max_values_indexes = [index for index, val in enumerate(labels) if
                              val == max(labels)]
        if len(max_values_indexes) > 1:
            for label in LABELS_IMPORTANCE:
                if label in max_values_indexes:
                    final_prediction = label
                    break
        else:
            final_prediction = max_values_indexes[0]
        if final_prediction == image.label:
            correct_predictions += 1
    return float(correct_predictions/no_of_predictions)



paths = get_paths_to_features_files(
    "C:\\Users\\user\Documents\ziemowit\\2x2_no_texture_no_norm")
features, patches_labels, image_labels = load_images_features(paths)
train_x, test_x, train_t, test_y = train_test_split(features, image_labels,
                                          test_size=0.2)
features = get_features_only(train_x)
# features = normalize(features, axis=0)
train_y = get_labels_only(train_x)
test_y = get_labels_only(test_x)
clf = RandomForestClassifier(n_estimators=200, )
# clf = MLPClassifier(hidden_layer_sizes=(130,20), verbose=True)
# clf = KNeighborsClassifier(n_neighbors=20)
# clf = SVC(C=0.01)
clf.fit(features, train_y)

predictions = predict(test_x, clf)
print(clf.__class__)
print(score(test_x, predictions))







