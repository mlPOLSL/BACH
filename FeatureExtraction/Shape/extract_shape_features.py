from collections import OrderedDict
import numpy as np
from skimage.measure import regionprops, label
from data_types import SegmentedImage, GreyscaleImage


def extract_shape_features(segmented_image: SegmentedImage,
                           minimal_cell_size: int = 50) -> OrderedDict:
    """
    Function for extracting shape features from a image with segmented blue nuclei.
    :param segmented_image: Image with segmented blue nuclei.
    :param minimal_cell_size: Minimal are of cell to be counted
    :return: Ordered dict with features:
    number_of_cells;
    total area of cells;
    coefficient of total are of cells and total area of an image;
    min, mean, max, std of area;
    min, mean, max, std of perimeter;
    min, mean, max, std of eccentricity;
    min, mean, max, std of solidity;
    """
    if not isinstance(segmented_image, SegmentedImage):
        raise TypeError(
            "Image should be segmentation, use SegmentedImage type")

    segmented_image = GreyscaleImage(segmented_image)
    mask = segmented_image[...] != 0
    masked = segmented_image.copy()
    masked[mask] = 255
    labels = label(masked)

    props = regionprops(labels)
    number_of_cells = count_cells(props, minimal_cell_size)

    total_area = calculate_total_area_of_cells(props, minimal_cell_size)

    fill_coefficient = total_area / (
        segmented_image.shape[0] * segmented_image.shape[1])
    min_area, mean_area, max_area, std_area = get_area_features(props,
                                                                minimal_cell_size)

    min_perimeter, mean_perimeter, max_perimeter, std_perimeter = get_perimeter_features(
        props, minimal_cell_size)

    min_eccentricity, mean_eccentricity, max_eccentricity, std_eccentricity = get_eccentricity_features(
        props, minimal_cell_size)

    min_solidity, mean_solidity, max_solidity, std_solidity = get_solidity_features(
        props, minimal_cell_size)

    features_dict = OrderedDict()
    features_dict["number_of_cells"] = number_of_cells
    features_dict["fill_coefficient"] = fill_coefficient
    features_dict["min_area"] = float(min_area)
    features_dict["mean_area"] = float(mean_area)
    features_dict["max_area"] = float(max_area)
    features_dict["std_area"] = std_area
    features_dict["min_perimeter"] = min_perimeter
    features_dict["mean_perimeter"] = mean_perimeter
    features_dict["max_perimeter"] = max_perimeter
    features_dict["std_perimeter"] = std_perimeter
    features_dict["min_eccentricity"] = min_eccentricity
    features_dict["mean_eccentricity"] = mean_eccentricity
    features_dict["max_eccentricity"] = max_eccentricity
    features_dict["std_eccentricity"] = std_eccentricity
    features_dict["min_solidity"] = min_solidity
    features_dict["mean_solidity"] = mean_solidity
    features_dict["max_solidity"] = max_solidity
    features_dict["std_solidity"] = std_solidity

    return features_dict


def count_cells(props, min_cell_area):
    """
    Function for counting number of cells (regions).
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Number of cells.
    """
    return len([region for region in props if region.area > min_cell_area])


def calculate_total_area_of_cells(props, min_cell_area):
    """
    Function for calculation of total area of cells (regions).
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Total area of cells.
    """
    try:
        return sum(
            [region.area for region in props if region.area > min_cell_area])
    except ValueError:
        return 0


def get_area_features(props, min_cell_area):
    """
    Function for calculation of area features of cells.
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Min, mean, max, std of areas of all cells
    """
    try:
        areas = [region.area for region in props if
                 region.area > min_cell_area]
        min_area = np.amin(areas)
        mean_area = np.mean(areas)
        max_area = np.amax(areas)
        std_area = np.std(areas)
        return min_area, mean_area, max_area, std_area
    except ValueError:
        return 0, 0, 0, 0


def get_perimeter_features(props, min_cell_area):
    """
    Function for calculation of perimeter features of cells.
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Min, mean, max, std of perimeters of all cells
    """
    try:
        perimeters = [region.perimeter for region in props if
                      region.area > min_cell_area]
        min_perimeter = np.amin(perimeters)
        mean_perimeter = np.mean(perimeters)
        max_perimeter = np.amax(perimeters)
        std_perimeter = np.std(perimeters)
        return min_perimeter, mean_perimeter, max_perimeter, std_perimeter
    except ValueError:
        return 0, 0, 0, 0


def get_eccentricity_features(props, min_cell_area):
    """
    Function for calculation of eccentricity features of cells.
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Min, mean, max, std of eccentricity of all cells
    """
    try:
        eccentrities = [region.eccentricity for region in props if
                        region.area > min_cell_area]
        min_eccentricity = np.amin(eccentrities)
        mean_eccentricity = np.mean(eccentrities)
        max_eccentricity = np.amax(eccentrities)
        std_eccentricity = np.std(eccentrities)
        return min_eccentricity, mean_eccentricity, max_eccentricity, std_eccentricity
    except ValueError:
        return 0, 0, 0, 0


def get_solidity_features(props, min_cell_area):
    """
    Function for calculation of solidity features of cells.
    :param props: Regions taken from regionprops.
    :param min_cell_area: Threshold for a size of a cell to eliminate
    oversegmented regions.
    :return: Min, mean, max, std of solidity of all cells
    """
    try:
        solidities = [region.eccentricity for region in props if
                      region.area > min_cell_area]
        min_solidity = np.amin(solidities)
        mean_solidity = np.mean(solidities)
        max_solidity = np.amax(solidities)
        std_solidity = np.std(solidities)
        return min_solidity, mean_solidity, max_solidity, std_solidity
    except ValueError:
        return 0, 0, 0, 0
