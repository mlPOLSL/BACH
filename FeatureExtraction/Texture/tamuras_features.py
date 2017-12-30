"""
Module for extracting Tamura's features. This code is a small part of:
https://github.com/loli/medpy/tree/Release_0.3.0p3
Some slight changes were made to fit Python 3
All credits are listed down below

Copyright (C) 2013 Oskar Maier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

author Alexander Ruesch
version r0.1.1
since 2013-08-24
status Release
"""
import numpy as np
from scipy import stats
from scipy.ndimage.filters import uniform_filter
from data_types import GreyscaleImage
from collections import OrderedDict


def coarseness(image: GreyscaleImage, voxel_spacing=None, mask=slice(None)):
    r"""
    Takes a simple or multi-spectral image and returns the coarseness of the
    texture.

    Step1  At each pixel, compute six averages for the windows of size 2**k x
           2**k, k=0,1,...,5, around the pixel.
    Step2  At each pixel, compute absolute differences E between the pairs of
           non overlapping averages in every directions.
    step3  At each pixel, find the value of k that maximises the difference Ek
           in either direction and set the best size Sbest=2**k
    step4  Compute the coarseness feature Fcrs by averaging Sbest over the
           entire image.
    Parameters
    ----------
    image : array_like or list/tuple of array_like
        A single image or a list/tuple of images (for multi-spectral case).
    voxel_spacing : sequence of floats
        The side-length of each voxel.
    mask : array_like
        A binary mask for the image or a slice object

    Returns
    -------
    coarseness : float
        The size of coarseness of the given texture. It is basically the size of
        repeating elements in the image.

    See Also
    --------


    """
    if not isinstance(image, GreyscaleImage):
        raise TypeError("Image should be a GreyscaleImage instance")
    # Step1:  At each pixel (x,y), compute six averages for the windows
    # of size 2**k x 2**k, k=0,1,...,5, around the pixel.

    # set default mask or apply given mask
    if not type(mask) is slice:
        if not type(mask[0] is slice):
            mask = np.array(mask, copy=False, dtype=np.bool)
    image = image[mask]

    # set default voxel spacing if not suppliec
    if voxel_spacing is None:
        voxel_spacing = tuple([1.] * image.ndim)

    if len(voxel_spacing) != image.ndim:
        print("Voxel spacing and image dimensions do not fit.")
        return None
    # set padding for image border control
    pad_size = np.asarray(
        [(np.rint((2 ** 5.0) * voxel_spacing[jj]), 0) for jj in
         range(image.ndim)]).astype(np.int)
    a_pad = np.pad(image, pad_width=pad_size, mode='reflect')

    # Allocate memory
    E = np.empty((6, image.ndim) + image.shape)
    # prepare some slicer
    raw_slicer = [slice(None)] * image.ndim
    slicer_for_image_in_pad = [slice(pad_size[d][0], None)
                               for d in range(image.ndim)]

    for k in range(6):

        size_vs = tuple(np.rint((2 ** k) * voxel_spacing[jj]) for jj in
                        range(image.ndim))
        A = uniform_filter(a_pad, size=size_vs, mode='mirror')

        # Step2: At each pixel, compute absolute differences E(x,y) between
        # the pairs of non overlapping averages in the horizontal and vertical
        # directions.
        for d in range(image.ndim):
            borders = np.rint((2 ** k) * voxel_spacing[d])
            slicer_pad_k_d = slicer_for_image_in_pad[:]
            slicer_pad_k_d[d] = slice(
                (int(pad_size[d][0] - borders) if
                 borders < pad_size[d][0] else 0), None)
            a_k_d = A[slicer_pad_k_d]
            aslicer_l = raw_slicer[:]
            aslicer_l[d] = slice(0, -int(borders))
            AslicerR = raw_slicer[:]
            AslicerR[d] = slice(int(borders), None)
            E[k, d, ...] = np.abs(a_k_d[aslicer_l] - a_k_d[AslicerR])

    # step3: At each pixel, find the value of k that maximises the difference
    # Ek(x,y) in either direction and set the best size Sbest(x,y)=2**k
    k_max = E.max(1).argmax(0)
    dim = E.argmax(1)
    dim_vox_space = np.asarray(
        [voxel_spacing[dim[k_max.flat[i]].flat[i]] for i in
         range(k_max.size)]).reshape(k_max.shape)
    S = (2 ** k_max) * dim_vox_space
    # step4: Compute the coarseness feature Fcrs by averaging Sbest(x,y) over
    # the entire image.
    return S.mean()


def contrast(image: GreyscaleImage, mask=slice(None)):
    r"""
    Takes a simple or multi-spectral image and returns the contrast of the texture.

    f_con = standard_deviation(gray_value) / (kurtosis(gray_value)**0.25)

    Parameters
    ----------
    image : array_like or list/tuple of array_like
        A single image or a list/tuple of images (for multi-spectral case).
    mask : array_like
        A binary mask for the image or a slice object
    Returns
    -------
    contrast : float
        High differences in gray value distribution is represented in a high
        contrast value.

    See Also
    --------


    """
    if not isinstance(image, GreyscaleImage):
        raise TypeError("Image should be an instance of GreyscaleImage")
    # set default mask or apply given mask
    if not type(mask) is slice:
        if not type(mask[0] is slice):
            mask = np.array(mask, copy=False, dtype=np.bool)
    image = image[mask]
    standard_deviation = np.std(image)
    kurtosis = stats.kurtosis(image, axis=None, bias=True, fisher=False)
    n = 0.25  # The value n=0.25 is recommended as the best for discriminating
              # the textures.

    f_con = standard_deviation / (kurtosis ** n)
    return f_con


def extract_tamuras_features(image: GreyscaleImage) -> OrderedDict:
    if not isinstance(image, GreyscaleImage):
        raise TypeError("Image should be an instance of GreyscaleImage")
    coarseness_ = coarseness(image)
    contrast_ = contrast(image)
    features = OrderedDict()
    features['coarseness'] = coarseness_
    features['contrast'] = contrast_
    return features
