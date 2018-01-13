from skimage import color, filters
from sklearn.cluster import KMeans
import numpy as np
from data_types import GreyscaleImage, SegmentedImage


def segment_blue_nuclei(image: np.ndarray) -> SegmentedImage:
    """
    Kmeans segmentation of the nuclei from histology microscopy images.
    :param image: Image from which blue nuclei woll be segmented
    :return: Image with segmented clue nuclei
    """
    NUMBER_OF_CLUSTERS = 3
    lab = color.rgb2lab(image)
    AB = lab[:, :, 1:]

    kmeans = KMeans(NUMBER_OF_CLUSTERS).fit(
        np.reshape(AB, (AB.shape[0] * AB.shape[1], AB.shape[2])))

    labels = np.reshape(kmeans.labels_, (image.shape[0], image.shape[1], 1))
    clusters = []
    for x in range(0, NUMBER_OF_CLUSTERS):
        mask = labels[..., 0] == x
        cluster = image.copy()
        cluster[~mask] = 0
        clusters.append(cluster)

    clusters_centers_Y = [center[1] for center in kmeans.cluster_centers_]
    blue_cluster_index = clusters_centers_Y.index(
        np.min(clusters_centers_Y))

    cluster_greyscale = GreyscaleImage(clusters[blue_cluster_index])
    cluster_greyscale_removed_black = np.array(
        [x for x in cluster_greyscale.flatten() if x != 0])
    threshold = filters.threshold_otsu(cluster_greyscale_removed_black)

    mask = cluster_greyscale[...] > threshold
    segmented = clusters[blue_cluster_index].copy()
    segmented[mask] = 0

    return SegmentedImage(segmented)
