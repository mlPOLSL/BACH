import pywt
import numpy as np
from skimage import io, img_as_float, color
import matplotlib.pyplot as plt
import time
from scipy import misc

path = "/Users/apple/PycharmProjects/BACH_local/Dataset/breasthistology/Training_data/InSitu/norm/normt1.tif"
image = io.imread(path)
image = color.rgb2grey(image)
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs
# cA = np.array([cA[x, y,] for x in range(0, cA.shape[0]) for y in range(0, cA.shape[1])])
# cH = np.array([cH[x, y] for x in range(0, cH.shape[0]) for y in range(0, cH.shape[1])])
# cV = np.array([cV[x, y] for x in range(0, cV.shape[0]) for y in range(0, cV.shape[1])])
# cD = np.array([cD[x, y] for x in range(0, cD.shape[0]) for y in range(0, cD.shape[1])])
# fig = plt.figure()
# for i, a in enumerate([cA, cH, cV, cD]):
#     ax = fig.add_subplot(2, 2, i + 1)
#     ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
#
# plt.show()

print("Max cA: {}".format(cA.max()))
print("Mean cA: {}".format(cA.mean()))
print("Std cA: {}".format(cA.std()))
print("Min cA: {}".format(cA.min()))
print("\n")
print("Max cH: {}".format(cH.max()))
print("Mean cH: {}".format(cH.mean()))
print("Std cH: {}".format(cH.std()))
print("Min cH: {}".format(cH.min()))
print("\n")
print("Max cV: {}".format(cV.max()))
print("Mean cV: {}".format(cV.mean()))
print("Std cV: {}".format(cV.std()))
print("Min cV: {}".format(cV.min()))
print("\n")
print("Max cD: {}".format(cD.max()))
print("Mean cD: {}".format(cD.mean()))
print("Std cD: {}".format(cD.std()))
print("Min cD: {}".format(cD.min()))

# cA.reshape((1536, 1024, 2))
# plt.imshow(cA)
# plt.show()