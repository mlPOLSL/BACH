import json
from collections import OrderedDict


def save_image_info(path: str, label: float, features: OrderedDict,
                    mode="w") -> None:
    """
    Takes information about an image and saves it to json file.
    :param path: Path to json file
    :param label: Label of an image
    :param features: Extracted features
    :param mode: Mode to open the file, by default set to "w" (write)
    :return: None
    """
    if not isinstance(features, OrderedDict):
        raise TypeError("Features should be an OrderedDict")
    info = OrderedDict()
    info["label"] = label
    info["features"] = features
    with open(path, mode) as info_file:
        json.dump(info, info_file)


def load_image_info(path: str, mode="r") -> OrderedDict:
    """
    Loads information about an image from a json file.
    :param path: Path to json file
    :param mode: Mode to open the file, by default set to "r" (read)
    :return: OrderedDict with information from json
    """
    with open(path, mode) as info_file:
        json_info = json.loads(info_file.read())
    info = OrderedDict(json_info)
    return info
