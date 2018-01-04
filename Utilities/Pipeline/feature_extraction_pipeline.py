import os
from collections import OrderedDict
from skimage import io, img_as_uint
import numpy as np

from FeatureExtraction.Color.color_features import extract_color_features
from FeatureExtraction.Wavelets.wavelets_features import get_wavelet_features
from FeatureExtraction.Texture.texture_features import extract_texture_features
from FeatureExtraction.hog_features import extract_hog_features

from Utilities.Pipeline.pipeline import PipelineStrategy, PipelineDataPoint, \
    Pipeline
from Utilities.save_features import load_image_info, save_image_info
from data_types import GreyscaleImage, NumpyImageUINT8


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def recreate_hierarchy(dir_path, new_dir_path):
    list_of_subdirs = get_immediate_subdirectories(dir_path)
    if list_of_subdirs != []:
        for dir in list_of_subdirs:
            new_dir = new_dir_path + "/" + dir
            pathlib.Path(new_dir).mkdir(parents=True,
                                        exist_ok=True)


def create_list_of_paths_to_load(dataset_path):
    list_of_paths = []
    for path, subdirs, files in os.walk(dataset_path):
        for name in files:
            list_of_paths.append(os.path.join(path, name))
    return list_of_paths


def remove_redundant_files_from_list(list_of_paths):
    new_list_of_paths = []
    for path in list_of_paths:
        filename, file_extension = os.path.splitext(path)
        if file_extension == ".tif":
            new_list_of_paths.append(path)
    return new_list_of_paths


def create_list_of_paths_to_save(list_of_paths_to_load, dataset_path,
                                 normalized_directory_path):
    list_of_paths = []
    for path in list_of_paths_to_load:
        new_path = path.replace(dataset_path, normalized_directory_path)
        new_path = new_path.replace(".tif", ".json")
        list_of_paths.append(new_path)
    return list_of_paths


def feature_extraction_pipeline(image_file_path, features_file_path,
                                mother_wavelet, distances, angles, label,
                                orientations, pixels_per_cell,
                                cells_per_block):
    data_point10 = PipelineDataPoint(image_file_path, 10)
    data_point11 = PipelineDataPoint(features_file_path, 11)
    data_point12 = PipelineDataPoint(mother_wavelet, 12)
    data_point13 = PipelineDataPoint(distances, 13)
    data_point14 = PipelineDataPoint(angles, 14)
    data_point15 = PipelineDataPoint(label, 15)
    data_point16 = PipelineDataPoint(orientations, 16)
    data_point17 = PipelineDataPoint(pixels_per_cell, 17)
    data_point18 = PipelineDataPoint(cells_per_block, 18)

    def load_image_file(input):
        return io.imread(input[0])

    def load_features_file(input):
        try:
            info = load_image_info(input[0])
        except IOError:
            info = OrderedDict()
            info["label"] = input[1]
            info["features"] = OrderedDict()
        return info

    def save_features_file(input):
        save_image_info(input[1], input[2], input[0])
        return 0

    def wavelet_features_extraction(input):
        greyscale = GreyscaleImage(input[0])
        return get_wavelet_features(greyscale, input[1])

    def color_features_extraction(input):
        return extract_color_features(input[0])

    def texture_features_extraction(input):
        uint = NumpyImageUINT8(input[0])
        return extract_texture_features(uint, input[1], input[2])

    def hog_features_extraction(input):
        greyscale = GreyscaleImage(input[0])
        return extract_hog_features(greyscale, input[1], input[2])

    def combine_features(input):
        info = input[0]
        for feature_type in input[1:]:
            for feature in feature_type:
                info["features"][feature] = feature_type[feature]
        return info

    strategy1 = PipelineStrategy(load_image_file, [10], 1)
    strategy2 = PipelineStrategy(load_features_file, [11, 15], 2)
    strategy3 = PipelineStrategy(wavelet_features_extraction, [1, 12], 3)
    strategy4 = PipelineStrategy(color_features_extraction, [1], 4)
    strategy5 = PipelineStrategy(texture_features_extraction, [1, 13, 14], 5)
    strategy6 = PipelineStrategy(hog_features_extraction, [1, 16, 17, 18], 6)
    strategy7 = PipelineStrategy(combine_features, [2, 3, 4, 5, 6], 7)
    strategy8 = PipelineStrategy(save_features_file, [11, 15, 7], 8)

    pipeline = Pipeline()

    pipeline.add_data_point(data_point10)
    pipeline.add_data_point(data_point11)
    pipeline.add_data_point(data_point12)
    pipeline.add_data_point(data_point13)
    pipeline.add_data_point(data_point14)
    pipeline.add_data_point(data_point15)
    pipeline.add_data_point(data_point16)
    pipeline.add_data_point(data_point17)
    pipeline.add_data_point(data_point18)

    pipeline.add_strategy(strategy1)
    pipeline.add_strategy(strategy2)
    pipeline.add_strategy(strategy3)
    pipeline.add_strategy(strategy4)
    pipeline.add_strategy(strategy5)
    pipeline.add_strategy(strategy6)
    pipeline.add_strategy(strategy7)
    pipeline.add_strategy(strategy8)

    pipeline.run()


def extract_label(filename):
    for index, label in enumerate(["b", "is", "iv", "n"]):
        if label in filename:
            return index


def dataset_feature_extraction(path_to_dataset, feature_dir_path,
                               mother_wavelet, distances, angles,
                               orientations, pixels_per_cell,
                               cells_per_block):
    data_point_10 = PipelineDataPoint(path_to_dataset, 10)
    data_point_11 = PipelineDataPoint(feature_dir_path, 11)
    data_point_12 = PipelineDataPoint(mother_wavelet, 12)
    data_point_13 = PipelineDataPoint(distances, 13)
    data_point_14 = PipelineDataPoint(angles, 14)
    data_point_15 = PipelineDataPoint(orientations, 15)
    data_point_16 = PipelineDataPoint(pixels_per_cell, 16)
    data_point_17 = PipelineDataPoint(cells_per_block, 17)

    def get_list_of_paths_to_load(input):
        return create_list_of_paths_to_load(input[0])

    def remove_redundant_paths(input):
        return remove_redundant_files_from_list(input[0])

    def get_list_of_paths_to_save(input):
        return create_list_of_paths_to_save(input[0], input[1], input[2])

    def perform_extraction_on_files(input):
        for load_path, save_path in zip(input[0], input[1]):
            label = extract_label(os.path.basename(load_path))
            feature_extraction_pipeline(load_path, save_path, input[2],
                                        input[3], input[4], label, input[5],
                                        input[6], input[7])
            print(os.path.basename(load_path) + " extracted")
        return 0

    strategy1 = PipelineStrategy(get_list_of_paths_to_load, [10], 1)
    strategy2 = PipelineStrategy(remove_redundant_paths, [1], 2)
    strategy3 = PipelineStrategy(get_list_of_paths_to_save, [2, 10, 11], 3)
    strategy4 = PipelineStrategy(perform_extraction_on_files,
                                 [2, 3, 12, 13, 14, 15, 16, 17], 4)


    pipeline = Pipeline()

    pipeline.add_data_point(data_point_10)
    pipeline.add_data_point(data_point_11)
    pipeline.add_data_point(data_point_12)
    pipeline.add_data_point(data_point_13)
    pipeline.add_data_point(data_point_14)
    pipeline.add_data_point(data_point_15)
    pipeline.add_data_point(data_point_16)
    pipeline.add_data_point(data_point_17)

    pipeline.add_strategy(strategy1)
    pipeline.add_strategy(strategy2)
    pipeline.add_strategy(strategy3)
    pipeline.add_strategy(strategy4)

    pipeline.run()


if __name__ == "__main__":
    img_path = "/Users/apple/PycharmProjects/BACH/Dataset/iciar_test/Photos"
    features_path = "/Users/apple/PycharmProjects/BACH/Dataset/iciar_test/whole_images_features"
    mother_wavelet = "db1"
    distance = [1, 3]
    angle = [0, np.pi / 2, np.pi, 3. * np.pi / 2.]
    label = 0
    orientations = 9
    pixels_per_cell = (300, 300)
    cells_per_block = (4, 4)
    dataset_feature_extraction(img_path, features_path, mother_wavelet, distance,
                                angle, orientations, pixels_per_cell,
                                cells_per_block)
