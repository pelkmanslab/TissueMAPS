# Copyright 2017 Scott Berry, University of Zurich
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
'''Jterator module for extracting a volume image from a 3D stack'''
import collections
import logging
import numpy as np
import mahotas as mh
from jtlib.filter import log_2d, log_3d

logger = logging.getLogger(__name__)

VERSION = '0.5.0'

Output = collections.namedtuple(
    'Output', ['volume_image', 'smoothed_surface_image', 'bead_coordinate_image', 'slide_surface_image', 'figure']
)
Beads = collections.namedtuple('Beads', ['coordinates', 'coordinate_image'])


class InvalidSlideError(ValueError):
    '''Raised when the slide surface is computed from a biased
    set of points'''


def array_to_coordinate_list(array, offset=[0,0]):
    '''Convert a 2D array representation of points in 3D
    to a list of x,y,z coordinates'''
    nonzero = np.nonzero(array)
    coordinates = np.vstack([
        np.stack([nonzero[0] + offset[0], nonzero[1] + offset[1]]),
        array[nonzero]
    ]).T
    return list(map(tuple, coordinates))


def coordinate_list_to_array(coordinates, shape, dtype=np.uint16):
    '''Convert a list of x,y,z coordinates to a 2D array
    representation of points in 3D'''
    image = np.zeros(shape, dtype=dtype)
    for i in range(len(coordinates)):
        image[coordinates[i][0], coordinates[i][1]] = coordinates[i][2]
    return image.astype(dtype=dtype)


def subsample_coordinate_list(points, num):
    if len(points) > num:
        subpoints = np.array(points)[np.linspace(
            start=0, stop=len(points), endpoint=False,
            num=num, dtype=np.uint32)]
    else:
        subpoints = points
    return list(map(tuple, subpoints))


def plane(x, y, params):
    '''Compute z-coordinate of plane in 3D'''
    a, b, c = params
    z = (a * x) + (b * y) + c
    return z


def slide_surface_params(slide):
    '''Determine the parameters (a,b,c) for the equation of the slide
    surface in 3D (z = ax + bx + c), given a coordinate image of the
    slide'''

    # Sample evenly from image quadrants
    m, n = np.floor(np.array(slide.shape) / 2.0).astype(np.int)
    ul, ur = slide[:m,:n], slide[m:,:n]
    ll, lr = slide[:m,n:], slide[m:,n:]

    ul_coords = array_to_coordinate_list(ul, offset=[0,0])
    ur_coords = array_to_coordinate_list(ur, offset=[m,0])
    ll_coords = array_to_coordinate_list(ll, offset=[0,n])
    lr_coords = array_to_coordinate_list(lr, offset=[m,n])

    lim = 200
    ul_n = len(ul_coords)
    ur_n = len(ur_coords)
    ll_n = len(ll_coords)
    lr_n = len(lr_coords)

    logger.debug('number of beads on slide surface in each quadrant' +
                 ' is %d, %d, %d, %d', ul_n, ur_n, ll_n, lr_n)
    if ((ur_n < lim and ll_n < lim and lr_n < lim) or
        (ul_n < lim and ll_n < lim and lr_n < lim) or
        (ul_n < lim and ur_n < lim and lr_n < lim) or
        (ul_n < lim and ur_n < lim and ll_n < lim)):
        raise InvalidSlideError('slide surface determined primarily' +
                                ' from one quadrant.')

    if ul_n < lim or ur_n < lim or ll_n < lim or lr_n < lim:
        logger.warn('one or more quadrants has < %d' +
                    ' points on slide surface,' +
                    ' upper-left = %d,' +
                    ' upper-right = %d,' +
                    ' lower-left = %d,' +
                    ' lower-right = %d',
                    lim, ul_n, ur_n, ll_n, lr_n)

    coordinates = (subsample_coordinate_list(ul_coords, 500) +
                   subsample_coordinate_list(ur_coords, 500) +
                   subsample_coordinate_list(ll_coords, 500) +
                   subsample_coordinate_list(lr_coords, 500))

    surface = fit_plane(coordinates)
    return surface


def squared_error(params, points):
    '''Compute the sum of squared residuals'''
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def fit_plane(points):
    '''Fit a plane to the 3D beads surface'''
    import scipy.optimize
    import functools

    fun = functools.partial(squared_error, points=points)
    params0 = [0.0, 0.0, 0.0]
    return scipy.optimize.minimize(fun, params0)


def interpolate_surface(coords, output_shape, method='linear'):
    '''Given a set of coordinates (not necessarily on a grid), an
    interpolation is returned as a numpy array'''
    from scipy.interpolate import griddata

    coords = coords.astype(np.float32)

    xy = np.column_stack((coords[:, 0], coords[:, 1]))
    xv, yv = np.meshgrid(
        range(output_shape[0]),
        range(output_shape[1])
    )
    if method == 'nearest':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='nearest', rescale=False
        )
    elif method == 'cubic':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='cubic', rescale=False
        )
    elif method == 'linear':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='linear', rescale=False
        )

    return interpolate.T


def localise_bead_maxima_3D(image, labeled_beads, minimum_bead_intensity):
    '''Given a 3D image of beads, and a 3D segmentation image,
    bead maxima are returned as a numpy array'''

    logger.debug('removing beads with intensity < %d', minimum_bead_intensity)
    bead_coords = []
    bboxes = mh.labeled.bbox(labeled_beads)
    for bead in range(np.max(labeled_beads)):
        x_min, x_max, y_min, y_max, z_min, z_max = bboxes[bead]
        label_image = labeled_beads[x_min:x_max, y_min:y_max, z_min:z_max]
        bounded = image[x_min:x_max, y_min:y_max, z_min:z_max]
        bounded = np.copy(bounded)
        bounded[label_image != bead] = 0
        local_coords = np.unravel_index(
            bounded.argmax(),
            bounded.shape
        )
        bbox_min = (x_min, y_min, z_min)
        centre = tuple(a + b for a, b in zip(bbox_min, local_coords))

        if (image[centre[0], centre[1], centre[2]] > minimum_bead_intensity):
            bead_coords.append(centre)

    logger.debug('convert %d bead vertices to image', len(bead_coords))
    coord_image = coordinate_list_to_array(bead_coords, image[:,:,0].shape)

    return Beads(bead_coords, coord_image)


def filter_vertices_per_cell_alpha_shape(coord_image_abs, mask, alpha, z_step, pixel_size):
    '''Vertices are filtered based on whether their coordinates form a
    likely part of a cell. This step is for rejection of spatial outliers'''
    import alpha_shape
    import random

    logger.debug('filtering vertices with alpha = %d,' +
                 ' z_step = %0.4f, pixel_size = %0.4f',
                 alpha, z_step, pixel_size)
    n_cells = np.max(mask)
    bboxes = mh.labeled.bbox(mask)
    conversion_factor = 2.0 * z_step / pixel_size

    def z_steps_to_abs(z):
        return float(conversion_factor * z)

    def abs_to_z_steps(z):
        return int(round(z / conversion_factor))

    filtered_coords_global = []
    if alpha > 0:
        for cell in range(1, n_cells + 1):
            x_min, x_max, y_min, y_max = bboxes[cell]
            cell_isolated = np.copy(coord_image_abs[x_min:x_max, y_min:y_max])
            label_image_isolated = np.copy(mask[x_min:x_max, y_min:y_max])

            cell_isolated[label_image_isolated != cell] = 0
            label_image_isolated[label_image_isolated != cell] = 0
            border_isolated = mh.labeled.bwperim(label_image_isolated, n=4)

            cell_isolated_coords = array_to_coordinate_list(
                cell_isolated
            )
            border_isolated_coords = array_to_coordinate_list(
                border_isolated.astype(np.float32)
            )

            # set border coordinates to zero-height.
            border_isolated_coords_zeroed = [(x, y, 0.0) for (x,y,z) in border_isolated_coords]

            # get coordinates from cell surface
            all_coords_local = cell_isolated_coords + border_isolated_coords_zeroed

            # convert to absolute positions for z-coordinate
            all_coords_local = [(x, y, z_steps_to_abs(z)) for (x, y, z) in all_coords_local]

            # filter vertices based on alpha_shape
            filtered_coords = alpha_shape.filter_vertices(
                all_coords_local, alpha
            )

            # convert back from absolute positions to integer number of z-steps
            filtered_coords = [(x, y, abs_to_z_steps(z)) for (x, y, z) in filtered_coords]

            # transform to global coords and add border coordinates
            n_border_coordinates = 100
            try:
                s = random.sample(set(border_isolated_coords_zeroed), n_border_coordinates)
            except ValueError:
                s = set(border_isolated_coords_zeroed)

            logger.debug(
                '%d bead vertices inside cell, adding %d border coordinates',
                len(filtered_coords),
                len(s),
            )

            filtered_coords_global += [(t[0] + x_min, t[1] + y_min, t[2]) for t in set(filtered_coords).union(s)]

    else:
        filtered_coords_global = array_to_coordinate_list(coord_image_abs)

    return filtered_coords_global


def smooth_surface(coordinate_image, mask,
                   output_shape, z_step,
                   pixel_size, smooth=0.5):
    '''For surface area calculations, it is necessary to smooth the
    volume image, this is done on a per-cell basis to reduce memory
    requirements. A list of filtered coordinates is used to fit a
    multiquadric surface'''
    from scipy.interpolate import Rbf

    logger.debug('generating smoothed image with smooth = %0.4f,' +
                 ' z_step = %0.4f, pixel_size = %0.4f',
                 smooth, z_step, pixel_size)
    n_cells = np.max(mask)
    bboxes = mh.labeled.bbox(mask)
    conversion_factor = z_step / pixel_size
    smoothed_image_float = np.zeros(shape=output_shape, dtype=np.float32)

    def z_steps_to_abs(z):
        return float(conversion_factor * z)

    def abs_to_z_steps(z):
        return int(round(z / conversion_factor))

    for cell in range(1, n_cells + 1):
        x_min, x_max, y_min, y_max = bboxes[cell]
        cell_isolated = np.copy(coordinate_image[x_min:x_max, y_min:y_max])
        label_image_isolated = np.copy(mask[x_min:x_max, y_min:y_max])

        cell_isolated[label_image_isolated != cell] = 0
        label_image_isolated[label_image_isolated != cell] = 0
        border_isolated = mh.labeled.bwperim(label_image_isolated, n=4)

        cell_isolated_coords = array_to_coordinate_list(
            cell_isolated
        )
        border_isolated_coords = array_to_coordinate_list(
            border_isolated.astype(np.float32)
        )

        # set border coordinates to zero-height.
        border_isolated_coords_zeroed = [(x, y, 0.0) for (x,y,z) in border_isolated_coords]

        # get coordinates from cell surface
        all_coords_local = cell_isolated_coords + border_isolated_coords_zeroed

        # convert to absolute positions for z-coordinate
        all_coords_local = [(x, y, z_steps_to_abs(z)) for (x, y, z) in all_coords_local]

        # generate smoothed surface
        XI, YI = np.meshgrid(
            np.arange(start=0, stop=cell_isolated.shape[1], step=1),
            np.arange(start=0, stop=cell_isolated.shape[0], step=1)
        )

        XYZ = np.array(all_coords_local)
        rbf = Rbf(XYZ[:,1], XYZ[:,0], XYZ[:,2],
                  function='multiquadric', smooth=smooth)
        ZI = rbf(XI,YI)

        # copy cell to the smoothed image
        np.copyto(
            dst=smoothed_image_float[x_min:x_max, y_min:y_max],
            src=ZI, where=ZI > 0
        )

        # multiply 'height' by 100 to avoid rounding errors
    return np.round(smoothed_image_float * 100 / conversion_factor).astype(np.uint16)


def main(image, mask, threshold=25,
         mean_size=6, min_size=10,
         filter_type='log_2d',
         minimum_bead_intensity=150,
         z_step=0.333, pixel_size=0.1625,
         alpha=0, smooth=0,
         plot=False):
    '''Converts an image stack with labelled cell surface to a cell
    `volume` image

    Parameters
    ----------
    image: numpy.ndarray[Union[numpy.uint8, numpy.uint16]]
        grayscale image in which beads should be detected (3D)
    mask: numpy.ndarray[Union[numpy.int32, numpy.bool]]
        binary or labeled image of cell segmentation (2D)
    threshold: int, optional
        intensity of bead in filtered image (default: ``25``)
    mean_size: int, optional
        mean size of bead (default: ``6``)
    min_size: int, optional
        minimal number of connected voxels per bead (default: ``10``)
    filter_type: str, optional
        filter used to emphasise the beads in 3D
        (options: ``log_2d`` (default) or ``log_3d``)
    minimum_bead_intensity: int, optional
        minimum intensity in the original image of an identified bead
        centre. Use to filter low intensity beads.
    z_step: float, optional
        distance between consecutive z-planes (um) (default: ``0.333``)
    pixel_size: float, optional
        size of pixel (um) (default: ``0.1625``)
    alpha: float, optional
        value of parameter for 3D alpha shape calculation
        (default: ``0``, no vertex filtering performed)
    smooth: float, optional
        value of parameter for smoothing for smoothed_surface_image
        (default: ``0``, no smoothed surface image generated)
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    jtmodules.generate_volume_image.Output
    '''

    # Check that there are cells identified in image
    if (np.max(mask) > 0):
        volume_image_calculated = True

        n_slices = image.shape[-1]
        logger.debug('input image has z-dimension %d', n_slices)

        # Remove high intensity pixels
        detect_image = image.copy()
        p = np.percentile(detect_image, 99.9)
        detect_image[detect_image > p] = p

        # Perform LoG filtering in 3D to emphasise beads
        if filter_type == 'log_2d':
            logger.info('using stacked 2D LoG filter to detect beads')
            f = -1 * log_2d(size=mean_size, sigma=float(mean_size - 1) / 3)
            filt = np.stack([f for _ in range(mean_size)], axis=2)

        elif filter_type == 'log_3d':
            logger.info('using 3D LoG filter to detect beads')
            filt = -1 * log_3d(mean_size, (float(mean_size - 1) / 3,
                                           float(mean_size - 1) / 3,
                                           4 * float(mean_size - 1) / 3))
        else:
            logger.info('using unfiltered image to detect beads')

        if filter_type == 'log_2d' or filter_type == 'log_3d':
            logger.debug('convolve image with filter kernel')
            detect_image = mh.convolve(detect_image.astype(float), filt)
            detect_image[detect_image < 0] = 0

        logger.debug('threshold beads')
        labeled_beads, n_labels = mh.label(detect_image > threshold)
        logger.debug('detected %d beads', n_labels)

        logger.debug('remove small beads')
        sizes = mh.labeled.labeled_size(labeled_beads)
        too_small = np.where(sizes < min_size)
        labeled_beads = mh.labeled.remove_regions(labeled_beads, too_small)
        mh.labeled.relabel(labeled_beads, inplace=True)
        logger.info(
            '%d beads remain after removing beads with < %d voxels',
            np.max(labeled_beads),
            min_size
        )

        logger.debug('localise beads in 3D')
        localised_beads = localise_bead_maxima_3D(
            image, labeled_beads, minimum_bead_intensity
        )

        logger.debug('mask beads inside cells')
        # expand mask to ensure slide-beads are well away from cells
        slide = np.copy(localised_beads.coordinate_image)
        expand_mask = mh.dilate(
            A=mask > 0,
            Bc=np.ones([25,25], bool)
        )
        slide[expand_mask] = 0

        logger.debug('determine coordinates of slide surface')
        try:
            bottom_surface = slide_surface_params(slide)
        except InvalidSlideError:
            logger.error('slide surface calculation is invalid' +
                         ' returning empty volume image')
            volume_image = np.zeros(shape=image[:,:,0].shape,
                                    dtype=image.dtype)
            figure = str()
            return Output(volume_image, volume_image, volume_image, volume_image, figure)

        logger.debug('subtract slide surface to get absolute bead coordinates')
        bead_coords_abs = []
        for i in range(len(localised_beads.coordinates)):
            bead_height = (
                localised_beads.coordinates[i][2] -
                plane(localised_beads.coordinates[i][0],
                      localised_beads.coordinates[i][1],
                      bottom_surface.x)
            )
            if bead_height > 0:
                bead_coords_abs.append(
                    (localised_beads.coordinates[i][0],
                     localised_beads.coordinates[i][1],
                     bead_height)
                )

        logger.debug('convert absolute bead coordinates to image')
        coord_image_abs = coordinate_list_to_array(
            bead_coords_abs, shape=image[:,:,0].shape, dtype=np.float32
        )

        filtered_coords_global = filter_vertices_per_cell_alpha_shape(
            coord_image_abs=coord_image_abs,
            mask=mask,
            alpha=alpha,
            z_step=z_step,
            pixel_size=pixel_size
        )

        logger.info('linear interpolation of cell surface')
        volume_image = interpolate_surface(
            coords=np.asarray(filtered_coords_global, dtype=np.uint16),
            output_shape=image[:,:,0].shape,
            method='linear'
        )

        volume_image = (100.0 * volume_image).astype(image.dtype)

        logger.debug('set regions outside mask to zero')
        volume_image[mask == 0] = 0

        if smooth > 0:
            logger.info('smoothing cell surface with parameter %0.4f', smooth)
            filtered_coordinate_image = coordinate_list_to_array(
                coordinates=np.array(filtered_coords_global, dtype=np.uint16),
                shape=image[:,:,0].shape)
            smoothed_surface_image = smooth_surface(
                coordinate_image=filtered_coordinate_image,
                mask=mask,
                output_shape=np.shape(image[:,:,0]),
                z_step=z_step,
                pixel_size=pixel_size,
                smooth=smooth
            )
        else:
            logger.info('omitting cell surface smoothing')
            smoothed_surface_image = np.zeros(
                shape=image[:,:,0].shape, dtype=image.dtype
            )

        logger.debug('convert bottom surface plane to image')
        dt = np.dtype(float)
        bottom_surface_image = np.zeros(slide.shape, dtype=dt)
        for ix in range(slide.shape[0]):
            for iy in range(slide.shape[1]):
                bottom_surface_image[ix, iy] = plane(
                    ix, iy, bottom_surface.x)

    else:
        logger.warn(
            'no objects in input mask, skipping cell volume calculation.'
        )
        volume_image_calculated = False
        volume_image = np.zeros(shape=image[:,:,0].shape, dtype=image.dtype)
        smoothed_surface_image = volume_image
        bottom_surface_image = volume_image
        localised_beads = Beads([],volume_image)

    if (plot and volume_image_calculated):
        logger.info('create plot')
        from jtlib import plotting
        plots = [
            plotting.create_intensity_image_plot(
                np.max(image, axis=-1), 'ul', clip=True
            ),
            plotting.create_float_image_plot(
                bottom_surface_image, 'll', clip=True
            ),
            plotting.create_intensity_image_plot(
                volume_image, 'ur', clip=True
            ),
            plotting.create_intensity_image_plot(
                smoothed_surface_image, 'lr', clip=True
            )

        ]
        figure = plotting.create_figure(
            plots, title='Convert stack to volume image'
        )
    else:
        figure = str()

    if (volume_image_calculated):
        bottom_surface_image[bottom_surface_image < 0] = 0
        bottom_surface_image = np.floor(bottom_surface_image).astype(np.uint16)

    return Output(volume_image,
                  smoothed_surface_image,
                  localised_beads.coordinate_image.astype(np.uint16),
                  bottom_surface_image,
                  figure)
