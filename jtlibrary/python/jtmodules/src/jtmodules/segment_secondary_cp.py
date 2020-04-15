# Copyright (C) 2016 University of Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from CellProfiler3, code at https://github.com/CellProfiler/
# CellProfiler/blob/master/cellprofiler/modules/identifysecondaryobjects.py
# which is governed by the BSD 3-Clause License

# The BSD 3-Clause License
#
# Copyright (C) 2003 - 2019 Broad Institute, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1.  Redistributions of source code must retain the above copyright notice,
#         this list of conditions and the following disclaimer.
#
#     2.  Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in the
#         documentation and/or other materials provided with the distribution.
#
#     3.  Neither the name of the Broad Institute, Inc. nor the names of its
#         contributors may be used to endorse or promote products derived from
#         this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED "AS IS."  BROAD MAKES NO EXPRESS OR IMPLIED
# REPRESENTATIONS OR WARRANTIES OF ANY KIND REGARDING THE SOFTWARE AND
# COPYRIGHT, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, CONFORMITY WITH ANY DOCUMENTATION,
# NON-INFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER OR NOT
# DISCOVERABLE. IN NO EVENT SHALL BROAD, THE COPYRIGHT HOLDERS, OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF, HAVE REASON TO KNOW, OR IN
# FACT SHALL KNOW OF THE POSSIBILITY OF SUCH DAMAGE.
#
# If, by operation of law or otherwise, any of the aforementioned warranty
# disclaimers are determined inapplicable, your sole remedy, regardless of the
# form of action, including, but not limited to, negligence and strict
# liability, shall be replacement of the software with an updated version if one
# exists.
#
# Development of CellProfiler has been funded in whole or in part with federal
# funds from the National Institutes of Health, the National Science Foundation,
# and the Human Frontier Science Program.

'''Jterator module for segmentation of secondary objects around existing
primary objects.
'''

import logging
import numpy as np
import mahotas as mh
import collections
import centrosome.cpmorphology
import centrosome.propagate
import scipy.ndimage
import skimage.morphology

logger = logging.getLogger(__name__)

VERSION = '0.0.1'
M_PROPAGATION = "propagation"
M_WATERSHED_G = "watershed_gradient"
M_WATERSHED_I = "watershed_image"
M_DISTANCE_N = "distance_n"
M_DISTANCE_B = "distance_b"

Output = collections.namedtuple('Output', ['secondary_label_image', 'figure'])


def filter_labels(primary_label_image, secondary_label_image):

    labels_in = np.unique(primary_label_image) - [0]
    labels_out = np.unique(secondary_label_image) - [0]

    extra_labels = set(labels_out) - set(labels_in)
    missing_labels = set(labels_in) - set(labels_out)

    if len(extra_labels) != 0:
        logger.warn(
            'Removing labels {} detected in secondary_label_image, which'
            'are not present in primary_label_image'.format(extra_labels))
        for label in extra_labels:
            secondary_label_image[secondary_label_image == label] = 0
    elif len(missing_labels) != 0:
        logger.warn(
            'Labels {} detected in primary_label_image, which'
            'are not present in secondary_label_image'.format(missing_labels))

    return secondary_label_image


def main(primary_label_image, intensity_image, method, threshold,
         regularization_factor=0.01, distance_to_dilate=3, fill_holes=True,
         plot=False):
    '''Detects secondary objects in an image by expanding the primary objects
    encoded in `primary_label_image`. The outlines of secondary objects are
    determined based on the watershed transform of `intensity_image` using the
    primary objects in `primary_label_image` as seeds.

    Parameters
    ----------
    primary_label_image: numpy.ndarray[numpy.int32]
        2D labeled array encoding primary objects, which serve as seeds for
        watershed transform
    intensity_image: numpy.ndarray[numpy.uint8 or numpy.uint16]
        2D grayscale array that serves as gradient for watershed transform;
        optimally this image is enhanced with a low-pass filter
    method: str
        one of ['propagation', 'watershed_gradient', 'watershed_image',
        'watershed_gradient', 'distance_n', 'distance_b'] specifying which
        segmentation method to use
    threshold: int
        maximum background value; pixels above `threshold` are considered
        foreground
    regularization_factor: float, optional
        used only for 'propagation' method. Larger values cause the distance
        between objects to be more important than the intensity image in
        determining cut lines. Smaller values cause the intensity image to
        be more important than the distance between objects.
    distance_to_dilate: int, optional
        used only for 'distance_n', 'distance_b' methods. The number of
        pixels by which the primary objects will be expanded.
    fill_holes: bool, optional
        whether holes should be filled in the secondary objects
    plot: bool, optional
        whether a plot should be generated

    Returns
    -------
    jtmodules.segment_secondary.Output

    Note
    ----

    Original text from CellProfiler module:

    There are several methods available to find the dividing lines between
    secondary objects that touch each other:
    -  propagation: This method will find dividing lines between
       clumped objects where the image stained for secondary objects shows a
       change in staining (i.e., either a dimmer or a brighter line).
       Smoother lines work better, but unlike the Watershed method, small
       gaps are tolerated. This method is considered an improvement on the
       traditional Watershed method. The dividing lines between objects
       are determined by a combination of the distance to the nearest
       primary object and intensity gradients. This algorithm uses local
       image similarity to guide the location of boundaries between cells.
       Boundaries are preferentially placed where the image's local
       appearance changes perpendicularly to the boundary (Jones et al,
       2005).
       The propagation algorithm is the default approach for secondary object
       creation. Each primary object is a "seed" for its corresponding
       secondary object, guided by the input
       image and limited to the foreground region as determined by the chosen
       thresholding method. lambda is a regularization parameter; see the help for
       the setting for more details. Propagation of secondary object labels is
       by the shortest path to an adjacent primary object from the starting
       ("seeding") primary object. The seed-to-pixel distances are calculated
       as the sum of absolute differences in a 3x3 (8-connected) image
       neighborhood, combined with lambda via sqrt(differences^2 +
       lambda^2).
    -  watershed_gradient: This method uses the watershed algorithm
       (Vincent and Soille, 1991) to assign pixels to the primary objects
       which act as seeds for the watershed. In this variant, the watershed
       algorithm operates on the Sobel transformed image which computes an
       intensity gradient. This method works best when the image intensity
       drops off or increases rapidly near the boundary between cells.
    -  watershed_image: This method is similar to the above, but it
       uses the inverted intensity of the image for the watershed. The areas
       of lowest intensity will be detected as the boundaries between cells.
       This method works best when there is a saddle of relatively low
       intensity at the cell-cell boundary.
    -  Distance: In this method, the edges of the primary objects are
       expanded a specified distance to create the secondary objects. For
       example, if nuclei are labeled but there is no stain to help locate
       cell edges, the nuclei can simply be expanded in order to estimate
       the cell's location. This is often called the "doughnut" or "annulus"
       or "ring" approach for identifying the cytoplasm. There are two
       methods that can be used:
       -  distance_n: In this method, the image of the secondary
          staining is not used at all; the expanded objects are the final
          secondary objects.
       -  distance_b: Thresholding of the secondary staining image
          is used to eliminate background regions from the secondary
          objects. This allows the extent of the secondary objects to be
          limited to a certain distance away from the edge of the primary
          objects without including regions of background.
    References
    ^^^^^^^^^^
    Jones TR, Carpenter AE, Golland P (2005) "Voronoi-Based Segmentation of
    Cells on Image Manifolds", ICCV Workshop on Computer Vision for
    Biomedical Image Applications, 535-543.
    Vincent L, Soille P (1991) "Watersheds in Digital Spaces: An Efficient
    Algorithm Based on Immersion Simulations", IEEE Transactions on Pattern
    Analysis and Machine Intelligence, Vol. 13, No. 6, 583-598
    '''

    if not np.any(primary_label_image == 0):
        secondary_label_image = primary_label_image
    else:
        n_objects = len(np.unique(primary_label_image[1:]))
        logger.info(
            'primary label image has %d objects',
            n_objects - 1
        )
        if n_objects > 1:

            labels_in = primary_label_image.copy()

            # convert intensity image to float64
            if intensity_image.dtype == np.uint8:
                img = np.float64(intensity_image) / 255
            elif intensity_image.dtype == np.uint16:
                img = np.float64(intensity_image) / 65535

            # threshold intensity image
            thresholded_image = intensity_image > threshold

            if method == M_DISTANCE_N:
                distances, (i, j) = scipy.ndimage.distance_transform_edt(
                    labels_in == 0,return_indices=True)
                labels_out = np.zeros(labels_in.shape, int)
                dilate_mask = distances <= distance_to_dilate
                labels_out[dilate_mask] = \
                    labels_in[i[dilate_mask], j[dilate_mask]]

            elif method == M_DISTANCE_B:
                labels_out, distances = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, 1.0)
                labels_out[distances > distance_to_dilate] = 0
                labels_out[labels_in > 0] = labels_in[labels_in > 0]

            elif method == M_PROPAGATION:
                labels_out, distance = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, regularization_factor)

            elif method == M_WATERSHED_G:
                # Apply the sobel filter to the image (both horizontal
                # and vertical). The filter measures gradient.
                sobel_image = np.abs(scipy.ndimage.sobel(img))

                # Combine the seeds and thresholded image to mask the watershed
                watershed_mask = np.logical_or(
                    thresholded_image, labels_in > 0)

                # Perform the first watershed
                labels_out = skimage.morphology.watershed(
                    connectivity=np.ones((3, 3), bool),
                    image=sobel_image,
                    markers=labels_in,
                    mask=watershed_mask
                )

            elif method == M_WATERSHED_I:
                # invert the image so that the maxima are filled first
                # and the cells compete over what's close to the threshold
                inverted_img = 1 - img

                # Combine the seeds and thresholded image to mask the watershed
                watershed_mask = np.logical_or(
                    thresholded_image, labels_in > 0)

                # Perform the watershed
                labels_out = skimage.morphology.watershed(
                    connectivity=np.ones((3, 3), bool),
                    image=inverted_img,
                    markers=labels_in,
                    mask=watershed_mask
                )

            if fill_holes:
                secondary_label_image = \
                    centrosome.cpmorphology.fill_labeled_holes(labels_out)
            else:
                secondary_label_image = labels_out

            secondary_label_image = filter_labels(
                primary_label_image, secondary_label_image).astype(np.int32)

        else:
            logger.info('skipping secondary segmentation')
            secondary_label_image = np.zeros(
                primary_label_image.shape, dtype=np.int32
            )

    # Tissuemaps label images must be separated by lines,
    # need to identify the boundary pixels
    borders_with_background = np.zeros(primary_label_image.shape, dtype=np.int32)

    for i in range(1,np.max(secondary_label_image)):
      borders_with_background = borders_with_background + mh.labeled.border(labeled=secondary_label_image,i=i,j=0)

    all_borders = mh.labeled.borders(secondary_label_image)
    touching_pixels = np.logical_xor(borders_with_background,all_borders)
    secondary_label_image[touching_pixels!=0] = 0

    n_objects = len(np.unique(secondary_label_image)[1:])
    logger.info('identified %d objects', n_objects)

    if plot:
        from jtlib import plotting
        colorscale = plotting.create_colorscale(
            'Spectral', n=n_objects, permute=True, add_background=True
        )
        outlines = mh.morph.dilate(
            mh.labeled.bwperim(secondary_label_image > 0))
        plots = [
            plotting.create_mask_image_plot(
                primary_label_image, 'ul', colorscale=colorscale
            ),
            plotting.create_mask_image_plot(
                secondary_label_image, 'ur', colorscale=colorscale
            ),
            plotting.create_intensity_overlay_image_plot(
                intensity_image, outlines, 'll'
            )
        ]
        figure = plotting.create_figure(plots, title='secondary objects')
    else:
        figure = str()

    return Output(secondary_label_image, figure)
