import os
import pathlib
from skimage import io, img_as_float
from Utilities.Pipeline.pipeline import Pipeline, PipelineDataPoint, \
    PipelineStrategy
from normalization import normalize_image


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
    for path in list_of_paths:
        filename, file_extension = os.path.splitext(path)
        if file_extension != ".tif":
            list_of_paths.remove(path)
    return list_of_paths


def create_list_of_paths_to_save(list_of_paths_to_load, dataset_path,
                                 normalized_directory_path):
    list_of_paths = []
    for path in list_of_paths_to_load:
        new_path = path.replace(dataset_path, normalized_directory_path)
        list_of_paths.append(new_path)
    return list_of_paths


def single_file_normalization_pipeline(path_to_load, path_to_save):
    data_point_10 = PipelineDataPoint(path_to_load, 10)
    data_point_11 = PipelineDataPoint(path_to_save, 11)

    def load_image(input):
        return img_as_float(io.imread(input[0]))

    def normalize(input):
        return normalize_image(input[0])

    def save_image(input):
        io.imsave(input[1], input[0])
        return 0

    normalization_pipeline = Pipeline()

    normalization_pipeline.add_data_point(data_point_10)
    normalization_pipeline.add_data_point(data_point_11)

    strategy1 = PipelineStrategy(load_image, [10], 1)
    strategy2 = PipelineStrategy(normalize, [1], 2)
    strategy3 = PipelineStrategy(save_image, [2, 11], 3)

    normalization_pipeline.add_strategy(strategy1)
    normalization_pipeline.add_strategy(strategy2)
    normalization_pipeline.add_strategy(strategy3)
    normalization_pipeline.run()


def dataset_normalization_pipeline(path_to_dataset, normalized_dir_path):
    data_point_10 = PipelineDataPoint(path_to_dataset, 10)
    data_point_11 = PipelineDataPoint(normalized_dir_path, 11)

    def create_hierarchy(input):
        recreate_hierarchy(input[0], input[1])
        return 0

    def get_list_of_paths_to_load(input):
        return create_list_of_paths_to_load(input[0])

    def remove_redundant_paths(input):
        return remove_redundant_files_from_list(input[0])

    def get_list_of_paths_to_save(input):
        return create_list_of_paths_to_save(input[0], input[1], input[2])

    def perform_normalization_on_files(input):
        for load_path, save_path in zip(input[0], input[1]):
            single_file_normalization_pipeline(load_path, save_path)
            print(os.path.basename(load_path) + " normalized")
        return 0

    strategy1 = PipelineStrategy(create_hierarchy, [10, 11], 0)
    strategy2 = PipelineStrategy(get_list_of_paths_to_load, [10], 1)
    strategy3 = PipelineStrategy(remove_redundant_paths, [1], 2)
    strategy4 = PipelineStrategy(get_list_of_paths_to_save, [2, 10, 11], 3)
    strategy5 = PipelineStrategy(perform_normalization_on_files, [2, 3], 4)

    pipeline = Pipeline()

    pipeline.add_data_point(data_point_10)
    pipeline.add_data_point(data_point_11)

    pipeline.add_strategy(strategy1)
    pipeline.add_strategy(strategy2)
    pipeline.add_strategy(strategy3)
    pipeline.add_strategy(strategy4)
    pipeline.add_strategy(strategy5)

    pipeline.run()


path_to_dataset = "/Users/apple/PycharmProjects/BACH/Dataset/iciar_test/Photos"
norm_dir_path = "/Users/apple/PycharmProjects/BACH/Dataset/iciar_test/Normalized"

dataset_normalization_pipeline(path_to_dataset, norm_dir_path)
