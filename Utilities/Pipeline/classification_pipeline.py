from Utilities.save_features import load_image_info
from Utilities.Pipeline.pipeline import PipelineStrategy, Pipeline, \
    PipelineDataPoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os


def create_list_of_paths_to_load(dataset_path):
    list_of_paths = []
    for path, subdirs, files in os.walk(dataset_path):
        for name in files:
            list_of_paths.append(os.path.join(path, name))
    return list_of_paths


def clf_train_pipeline(clf, normalizer, path_to_dataset):
    data_point10 = PipelineDataPoint(path_to_dataset, 10)
    data_point11 = PipelineDataPoint(clf, 11)
    data_point12 = PipelineDataPoint(normalizer, 12)

    def create_list_of_files(input):
        return create_list_of_paths_to_load(input[0])

    def load_data(input):
        info_data_list = []
        for path in input[0]:
            if ".DS_Store" in path:
                continue
            info_data_list.append(load_image_info(path))
        return info_data_list

    def prepare_data(input):
        list_of_labels = []
        list_of_features = []
        for data in input[0]:
            list_of_labels.append(data["label"])
            features = [data["features"][key] for key in data["features"]]
            list_of_features.append(features)
        return (list_of_labels, list_of_features)

    def normalize_features(input):
        normalizer = input[1]
        labels = input[0][0]
        features = input[0][1]
        features = normalize(features, axis=0)
        return (labels, features)

    def divide_sets(input):
        labels = input[0][0]
        features = input[0][1]
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels,
                                                            test_size=0.1,
                                                            shuffle=1)
        return X_train, X_test, Y_train, Y_test

    def train(input):
        clf = input[1]
        features = input[0][0]
        labels = input[0][2]
        clf.fit(features, labels)
        return 0

    def test(input):
        clf = input[1]
        features = input[0][1]
        labels = input[0][3]
        return clf.score(features, labels)

    def combine_outputs(input):
        return input

    strategy1 = PipelineStrategy(create_list_of_files, [10], 1)
    strategy2 = PipelineStrategy(load_data, [1], 2)
    strategy3 = PipelineStrategy(prepare_data, [2], 3)
    strategy4 = PipelineStrategy(normalize_features, [3, 12], 4)
    strategy5 = PipelineStrategy(divide_sets, [4], 5)
    strategy6 = PipelineStrategy(train, [5, 11], 6)
    strategy7 = PipelineStrategy(test, [5, 11], 7)
    strategy8 = PipelineStrategy(combine_outputs, [7, 11, 12], 8)

    pipeline = Pipeline()

    pipeline.add_data_point(data_point10)
    pipeline.add_data_point(data_point11)
    pipeline.add_data_point(data_point12)

    pipeline.add_strategy(strategy1)
    pipeline.add_strategy(strategy2)
    pipeline.add_strategy(strategy3)
    pipeline.add_strategy(strategy4)
    pipeline.add_strategy(strategy5)
    pipeline.add_strategy(strategy6)
    pipeline.add_strategy(strategy7)
    pipeline.add_strategy(strategy8)

    return pipeline.run()

if __name__ == "__main__":
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import Normalizer

    clf = MLPClassifier(verbose=1,hidden_layer_sizes=(440))
    normalizer = Normalizer()
    path = "/Users/apple/Downloads/whole_images_features_no_texture_no_norm"

    output = clf_train_pipeline(clf, normalizer, path).data

    print(output[0])
