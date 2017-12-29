import os
import numpy as np
from skimage import io
import pickle
from sklearn.svm import SVC
from DataPreprocessing.Grid import grid
from FeatureExtraction.Wavelets.wavelets_features import get_wavelet_features
from FeatureExtraction.Texture.texture_features import extract_texture_features
from data_types import GreyscaleImage, NumpyImageUINT8

training_data_path = "Dataset/Training_data"

list_of_files = os.listdir(training_data_path)
list_of_files.remove(".DS_Store")
labels = {key: value for value, key in enumerate(list_of_files)}
list_of_images = []
list_of_labels = []
for key in labels:
    list_of_files = os.listdir(training_data_path + "/" + key)
    list_of_files.remove(".DS_Store")
    for image_path in list_of_files:
        path = training_data_path + "/" + key + "/" + image_path
        image = io.imread(path)
        label = labels[key]
        list_of_images.append(image)
        list_of_labels.append(label)
print("Images loaded")
images_features = []
index = 0
for image in list_of_images:
    greyscale = GreyscaleImage(image)
    wavelet_fetures = get_wavelet_features(greyscale, "haar")
    npImUINT8 = NumpyImageUINT8(image)
    texture_features = extract_texture_features(npImUINT8, [1, 3],
                                                [0, np.pi / 2, np.pi,
                                                 3. * np.pi / 2.])
    features = {**wavelet_fetures, **texture_features}
    images_features.append(features)
    print(index)
    index += 1

print("Features extracted")
lists_of_features = [list(features.values()) for features in images_features]
clf = SVC()
clf.fit(lists_of_features, list_of_labels)
print("Clf fited with training data")
with open("first_clf", "wb") as file:
    pickle.dump(clf, file)
