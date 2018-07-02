#
# Author: David Riser
# Date:   July 2, 2018
# File:   bsa.py
# Description: Using methods from utils.py calculate
# the beam spin asymmetry for some cuts.
#

import json
import numpy as np
import pandas as pd
import utils
import time

def load_dataset(config):

    # Load data and drop nan values.
    data = pd.read_csv(config['file_path'],
                       compression=config['file_compression'],
                       nrows=config['sample_size'])
    data.dropna(how='any', inplace=True)

    # These axes will be kept, everything else will
    # be dropped.
    IMPORTANT_AXES = ['alpha', 'dist_cc', 'dist_cc_theta',
                      'dist_dcr1', 'dist_dcr3', 'dist_ecsf',
                      'dist_ec_edep', 'dist_ecu', 'dist_ecv',
                      'dist_ecw', 'dist_vz', 'helicity',
                      'missing_mass', 'p_mes', 'phi_h',
                      'pt', 'q2', 'x', 'z', 'dvz']

    # Perform the axis dropping.
    for col in data.columns:
        if col not in IMPORTANT_AXES:
            data.drop(col, axis=1, inplace=True)

    # Reduce memory usage and return loaded data for
    # analysis.
    data, _ = utils.reduce_mem_usage(data)
    return data

def setup_binning(config, data):
    ''' Return dictionary of bins chosen by quantile method. '''

    bins = {}
    for axis in config['axes']:
        if axis is not 'z':
            # Setup the non-z axes to have a limited z range
            # defined in the configuration dictionary.
            bins[axis] = utils.bin_by_quantile(
                data.query('z > %f and z < %f' % (config['z_range'][0], config['z_range'][1])),
                axis=axis,
                n_bins=config['n_bins']
            )
        else:
            bins[axis] = utils.bin_by_quantile(data,
                                               axis=axis,
                                               n_bins=config['n_bins']
                                               )
    return bins


def load_variations(path):
    ''' Read variations json file that defines
    the different systematics that will be checked.
    '''

    # Load the dictionary from the .json file.
    with open(path, 'r') as inputfile:
        data = json.load(inputfile)

    # Replace all strings on the second level
    # with integers.
    data_type_corrected = {}
    for par_name, levels in data.iteritems():
        data_type_corrected[par_name] = {}

        for level in levels:
            data_type_corrected[par_name][int(level)] = list([float(data[par_name][level][0]),
                                                              float(data[par_name][level][1])])

    return data_type_corrected

def process():

    start_time = time.time()

    config = {}
    config['axes'] = ['x', 'z', 'pt', 'q2']
    config['z_range'] = [0.25, 0.75]
    config['n_bins'] = 10
    config['file_path'] = '/Users/davidriser/Data/inclusive/inclusive_kaon_small.csv'
    config['sample_size'] = None
    config['file_compression'] = 'bz2'
    config['variation_file'] = '/Users/davidriser/repos/python-analysis/kaon-bsa/variations.json'

    # Load entire dataset, this
    # should only be done once
    # because it's 1.5 GB at load time.
    data = load_dataset(config)

    # Applying nominal cuts to get the subset
    # of events that I consider good when
    # using the "best" cut values.
    nominal_filter = utils.build_filter(data)
    nominal_data   = utils.build_dataframe(data, nominal_filter)

    # Use quantile binning to get integrated bins
    # for the axes listed in the configuration.
    bins = setup_binning(config, nominal_data)

    # Calculate the results for the nominal subset of data.
    results = {}
    results['nominal'] = utils.get_results(nominal_data, bins, config)
    del nominal_data

    # Define variations to consider.  These
    # are the systematics that are applied.
    variations = load_variations(config['variation_file'])
    for par in variations.keys():
        results[par] = {}

        for index in variations[par].keys():

            var_time = time.time()
            print('Doing  %.3f < %s < %.3f' % (variations[par][index][0], par,
                                               variations[par][index][1]))

            # get these cut values
            temp_dict = {}
            temp_dict[par] = variations[par][index]

            # get data
            temp_filter = utils.build_filter(data, temp_dict)
            temp_data = utils.build_dataframe(data, temp_filter)
            results[par][index] = utils.get_results(temp_data, bins, config)
            del temp_data

            end_var_time = var_time - time.time()
            print('Elapsed time %.3f' % end_var_time)

    # Using all variations, systematic
    # uncertainties are added to the dataframe.
    assign_systematics(results)

    exe_time = time.time() - start_time
    print('Finished execution in %.3f seconds.' % exe_time)

def assign_systematics(results):
    for conf in results.keys():
        if conf is not 'nominal':
            for val in results[conf].keys():
                # Is this safe?  They could be in different orders.  It has been checked visually.
                results[conf][val]['shift'] = results['nominal']['value'] - results[conf][val]['value']

    shift_df, var_to_col = utils.get_linearized_error(results)
    var_to_col['sys_0'] = 'beam_pol'
    results['nominal'] = pd.merge(left=results['nominal'],
                                  right=shift_df,
                                  on=['axis', 'global_index'])

# This is quite clearly the main function.
if __name__ == '__main__':
    process()