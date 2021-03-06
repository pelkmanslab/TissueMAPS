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
'''Jterator module for generating a channel by combining two grayscale images
into one, such that the resulting combined image is a weighed sum of the two
input images.'''
import numpy as np
import mahotas as mh
import logging
import collections

logger = logging.getLogger(__name__)

VERSION = '0.0.3'

Output = collections.namedtuple('Output', ['combined_image', 'figure'])


def main(image_1, image_2, weight_1, weight_2, plot=False):
    '''Combines `image_1` with `image_2`.

    Parameters
    ----------
    input_mask_1: numpy.ndarray[numpy.uint8 or numpy.uint16]
        2D unsigned integer array
    input_mask_2: numpy.ndarray[numpy.uint8 or numpy.uint16]
        2D unsigned integer array
    weight_1: float
        weight for `image_1`
    weight_2: float
        weight for `image_2`

    Returns
    -------
    jtmodules.combine_channels.Output

    Raises
    ------
    ValueError
        when `image_1` and `image_2` don't have the same dimensions
        and data type and if they don't have unsigned integer type
    '''
    logger.info('weight for first image: %d', weight_1)
    logger.info('weight for second image: %d', weight_2)

    if image_1.shape != image_2.shape:
        raise ValueError('The two images must have identical dimensions.')
    if image_1.dtype != image_2.dtype:
        raise ValueError('The two images must have identical data type.')

    if image_1.dtype == np.uint8:
        max_val = 2**8 - 1
    elif image_1.dtype == np.uint16:
        max_val = 2**16 - 1
    else:
        raise ValueError('The two images must have unsigned integer type.')

    logger.info('combine images using the provided weights')
    combined_image = image_1 * weight_1 + image_2 * weight_2
    # Set all values below 0 to 0
    combined_image[combined_image < 0] = 0
    combined_image[combined_image > max_val] = max_val
    logger.info('cast combined image back to correct data type')

    # Cast image to the same type as the input image
    combined_image = combined_image.astype(image_1.dtype)

    if plot:
        from jtlib import plotting
        plots = [
            plotting.create_intensity_image_plot(image_1, 'ul'),
            plotting.create_intensity_image_plot(image_2, 'ur'),
            plotting.create_intensity_image_plot(combined_image, 'll')
        ]
        figure = plotting.create_figure(plots, title='combined image')
    else:
        figure = str()

    return Output(combined_image, figure)
