#! /usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import h5py
import glob
import re


'''
Script for merging measurement data generated by "jterator"
for use with "tissueMAPS".

Some things are currently hard-coded and should become more flexible.
A lot of logic (e.g. config file) could be adopted from "illuminati".
'''

# TODO: yaml config file
header_mapper = {
    'Dapi': 'DAPI',
    'Red': 'EEA1',
    'Celltrace': 'SE'
}

cycle_mapper = {
    '_01': 'cycle1',
    '_02': 'cycle11',
    '_03': 'cycle12'
}


def get_pos_from_filename(site, image_folder,
                          site_pattern='*_s%.4d_*',
                          pos_regexp='_r(\d+)_c(\d+)_',
                          one_based=True):
    filename = glob.glob(os.path.join(image_folder, site_pattern % site))
    filename = filename[0]

    m = re.search(pos_regexp, filename)
    if not m:
            raise Exception('Can\'t create SiteImage object '
                            'from filename ' + filename)
    else:
        row, col = map(int, m.groups())
        if one_based:
            row -= 1
            col -= 1
        return (row, col)


def build_ids(original_object_ids, row_id, col_id):
    # TODO: generate final, global cell IDs
    ids = [
            '%d-%d-%d' % (row_id, col_id, local_id)
            for local_id in original_object_ids
    ]
    return ids


def merge_data(project_dir, output_dir, rename):

    project_name = os.path.basename(project_dir)
    # output_filename = os.path.join(output_dir, '%s.features' % project_name)
    output_filename = os.path.join(output_dir, 'features.h5')

    cycle_dirs = glob.glob(os.path.join(project_dir, '%s*' % project_name))

    # Loop over cycles

    first_cycle = True
    for cycle in cycle_dirs:

        print('. Processing cycle # %d' %
              int(re.search('%s.?(\d+)$' % project_name, cycle).group(1)))

        data_dir = os.path.join(cycle, 'data')
        data_files = glob.glob(os.path.join(data_dir, '*.data'))

        f = h5py.File(data_files[0], 'r')
        groups = f.keys()
        f.close()
        groups.remove('OriginalObjectIds')
        features = {k: [] for k in groups}

        print '.. Reading data from files'

        ids = []
        for filename in data_files:

            f = h5py.File(filename, 'r')
            if not f.keys():
                print 'Warning: file "%s" is emtpy' % filename
                continue

            # Convert site specific ids to global ids for tissueMAPS
            # Get positional information from filename
            site = int(re.search(r'(\d{5})\.data', filename).group(1))
            image_folder = os.path.join(data_dir, '../TIFF')
            (row, col) = get_pos_from_filename(site, image_folder)
            # Build identifier string
            nitems = len(f.values()[0])
            ids += build_ids(f['OriginalObjectIds'][:nitems], row, col)

            # Read measurement data
            for g in groups:
                if g in f.keys():
                    features[g].append(np.matrix(f[g][()]).T)  # use matrix!!!
                else:
                    print('Warning: group "%s" does not exist in file "%s' %
                          (g, filename))
                    features[g].append(None)
            f.close()

        # Check #1: do all sites (jobs) have the same number of features?
        check1 = map(len, features.values())

        if len(np.unique(check1)) == 1:
            print '🍺  Check 1 passed'
        else:
            raise Exception('Sites have a different number of features.')

        # Combine features per site into a vector
        dataset = np.array([np.vstack(feat) for feat in features.values()])

        # Check #2: do all features have the same number of measurements?
        check2 = [feat.shape[0] for feat in dataset]
        # ignore empty sites
        filter(lambda i: i[1] != check2[0], enumerate(check2))

        if len(np.unique(check2)) == 1:
            print '🍺  Check 2 passed'
        else:
            raise Exception('Features have a different number of measurements.')

        # Combine features into one nxp numpy array,
        # where n is the number of single-cell measurements
        # and p is the number of features
        dataset = np.hstack(dataset)
        header = groups
        dataset_name = os.path.basename(cycle)

        # Rename features for better interpretation
        if rename:
            print '.. Renaming features'
            for i, feat in enumerate(header):
                for j, substring in enumerate(header_mapper.keys()):
                    r = re.compile(substring)
                    match = re.search(r, feat)
                    if match:
                        header[i] = re.sub(header_mapper.keys()[j],
                                           header_mapper.values()[j],
                                           header[i])

            print '.. Renaming cycles'
            for j, substring in enumerate(cycle_mapper.keys()):
                r = re.compile(substring)
                match = re.search(r, dataset_name)
                if match:
                    dataset_name = cycle_mapper.values()[j]

        header = np.array(map(np.string_, groups))  # safest for hdf5

        print '.. Writing merged data into HDF5 file'

        if first_cycle:
            f = h5py.File(output_filename, 'w')  # truncate file if exists
            # Ids are the same for each cycle, so we store it in the root group
            f.create_dataset('/ids', data=np.array(ids))
            f.close()

        # Write merged data into a new hdf5 file
        f = h5py.File(output_filename, 'a')
        location = '/%s' % dataset_name
        # Write the dataset
        f.create_dataset(location, data=dataset)
        # Add the 'header' metadata as an attribute
        f[location].attrs.__setitem__('header', header)

        f.close()
        first_cycle = False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge data generated by Jterator for use in tissueMAPS.')

    parser.add_argument('project_dir', nargs='*',
                        help='absolute path to project directory')

    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='directory where the HDF5 file should be saved')

    parser.add_argument('-r', '--rename', dest='rename',
                        default=False, action='store_true',
                        help='rename features according to \'header_mapper\'')

    args = parser.parse_args()

    project_dir = args.project_dir[0]
    output_dir = args.output_dir

    if not project_dir:
        raise Exception('Project directory "%s" does not exist.' % project_dir)

    if not output_dir:
        raise Exception('Output directory "%s" does not exist.' % output_dir)

    merge_data(project_dir, output_dir, args.rename)
