from skimage.feature import hog
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.color import rgb2gray

img = Image.open("C:\\Users\\user\PycharmProjects\BACH\\b001.tif")
arr = np.asarray(img)
img.close()
arr = rgb2gray(arr)
fd= hog(arr, orientations=1, pixels_per_cell=(100, 100), feature_vector=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(arr, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_adjustable('box-forced')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# ax1.set_adjustable('box-forced')
# plt.show()
