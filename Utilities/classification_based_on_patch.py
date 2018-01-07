from typing import List, Tuple
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import random
import os
import json
import numpy as np

BENIGN_LABEL = 0
INSITU_LABEL = 1
INVASIVE_LABEL = 2
NORMAL_LABEL = 3


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

def load_features(paths_to_feature_files: List[str]):
    """

    :param paths_to_feature_files:
    :return:
    """
    labels = []
    features = []
    for path in paths_to_feature_files:
        label, feature_list = get_features_and_label_from_file(path)
        labels.append(label)
        features.append(feature_list)
    return (labels, features)

def load_images_patches_features(paths_to_feature_files: List[str]) -> OrderedDict:
    """
    Reads all the patches and groups them in a dictionary where each key stands
    for an image and its value is a list of tuples where each tuple is one patch
    label and features.

    :param paths_to_feature_files: absolute paths to .jsons files.
    :return: Dictionary of images with all its patches labels and features.
    :rtype: OrderedDict[str, Tuple[int, List[float]]
    """
    images = OrderedDict()
    for path in paths_to_feature_files:
        image_name = os.path.basename(os.path.dirname(path))
        label, feature_list = get_features_and_label_from_file(path)
        if image_name not in images:
            images[image_name] = []
        images[image_name].append((label, feature_list))
    return images

def predict(test_images: OrderedDict, classifier) -> OrderedDict:
    """
    For a given dictionary of test images predicts the images labels using the
    most common predicted label for all its patches.

    :param test_images: OrderedDict where key is an image name, and value is a
    list of Tuple[int, List[Float]], int is a label and List[Float] stores
    features.

    :param classifier: sklearn compatible classifier
    :return: OrderedDict where key is a name of an image and its value is a
    predicted label for it.
    """
    image_predictions = OrderedDict()
    for image, patches in test_images.items():
        patch_predictions = []
        for patch in patches:
            test_y = patch[0]
            test_x = patch[1]
            patch_prediction = classifier.predict(np.array([test_x]))
            patch_predictions.append(patch_prediction)
        benigns = 0
        insitus = 0
        invasives = 0
        normals = 0
        for patch_prediction in patch_predictions:
            if patch_prediction[0] == BENIGN_LABEL:
                benigns += 1
            elif patch_prediction[0] == INSITU_LABEL:
                insitus += 1
            elif patch_prediction[0] == INVASIVE_LABEL:
                invasives += 1
            else:
                normals += 1
        labels = [benigns, insitus, invasives, normals]
        whole_image_prediction = labels.index(max(labels))
        image_predictions[image] = whole_image_prediction
    #print(image_predictions)
    return image_predictions

def score(test_images, classifier, whole_images_labels):
    image_predictions = predict(test_images, classifier)
    predictions_count = len(test_images)
    positive_predictions = 0
    for image_prediction_name, prediction in image_predictions.items():
        if prediction == whole_images_labels[image_prediction_name][0][0]:
            positive_predictions +=1
    return float(positive_predictions/predictions_count)

def get_training_and_test_sets(images):
    training_set = []
    test_set = []
    for index, image in enumerate(list(images.items())):
        if(index % 5 == 0):
            test_set.append(image)
        else:
            training_set.append(image)
    return(OrderedDict(training_set), OrderedDict(test_set))

def get_training_and_test_sets_randomly(images):
    list_of_items = list(images.items())
    random.shuffle(list_of_items)
    train_part = OrderedDict(list_of_items[int(len(images)/5):])
    test_part = OrderedDict(list_of_items[:int(len(images)/5)])
    return train_part, test_part


if __name__ == "__main__":

    paths = get_paths_to_features_files(
        "F:\ICIAR2018_BACH_Challenge\Photos/features")
    images = load_images_patches_features(paths)
    whole_images_labels = load_images_patches_features(
        get_paths_to_features_files("F:\ICIAR2018_BACH_Challenge\Photos/features/1x1"))
    # clf = RandomForestClassifier(n_estimators=100)
    clf = KNeighborsClassifier(n_neighbors=15)
    train_part, test_part = get_training_and_test_sets(images)
    train_y = []
    train_x = []
    for image, patches in train_part.items():
        for patch in patches:
            train_y.append(patch[0])
            train_x.append(patch[1])
    clf.fit(train_x, train_y)
    print(score(test_part, clf, whole_images_labels))

