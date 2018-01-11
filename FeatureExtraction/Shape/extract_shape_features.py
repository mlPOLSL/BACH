from skimage import io, color, img_as_float, filters
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

path = "/Users/apple/PycharmProjects/BACH/Dataset/Test_data/3.tif"
img = img_as_float(io.imread(path))
lab = color.rgb2lab(img)
L = lab[:, :, 0]
AB = lab[:, :, 1:]

kmeans = KMeans(3)
print(AB.shape)
output = kmeans.fit(np.reshape(AB, (AB.shape[0] * AB.shape[1], AB.shape[2])))
labels = output.labels_
labels = np.reshape(labels, (1536, 2048, 1))
clusters = [np.zeros(img.shape) for x in range(0,3)]
for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        if labels[x, y] == 0:
            clusters[0][x, y] = img[x, y]
            clusters[1][x, y] = 0
            clusters[2][x, y] = 0
        if labels[x, y] == 1:
            clusters[0][x, y] = 0
            clusters[1][x, y] = img[x, y]
            clusters[2][x, y] = 0
        if labels[x, y] == 2:
            clusters[0][x, y] = 0
            clusters[1][x, y] = 0
            clusters[2][x, y] = img[x, y]

plt.imshow(clusters[0])
plt.show()
plt.imshow(clusters[1])
plt.show()
plt.imshow(clusters[2])
plt.show()

clusters_mean_centers = [np.mean(center) for center in output.cluster_centers_]
blue_cluster_index = clusters_mean_centers.index(
    np.min(clusters_mean_centers))

print(blue_cluster_index)
print(output.cluster_centers_)
print(clusters_mean_centers)
brightness_threshold = filters.threshold_otsu(L)
L = clusters[blue_cluster_index][:, :, 0]
L = L < brightness_threshold

segmented = np.zeros(img.shape)
for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        if L[x,y] == 0:
            segmented[x,y] = 0
        else:
            segmented[x,y] =  clusters[blue_cluster_index][x,y]

io.imsave("1.tif",img)
io.imsave("1segmented.tif",segmented)

plt.imshow(img)
plt.show()
plt.imshow(segmented)
plt.show()

