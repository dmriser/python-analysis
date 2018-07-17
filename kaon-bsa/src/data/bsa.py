#!/usr/bin/env python 
#
# Author: David Riser
# Date:   July 2, 2018
# File:   bsa.py
# Description: Using methods from utils.py calculate
# the beam spin asymmetry for some cuts.
#

import argparse
import json
import logging
import numpy as np
import pandas as pd
import pickle 
import utils
import time

def setup_binning(config, data):
    ''' Return dictionary of bins chosen by quantile method. '''

    log = logging.getLogger(__name__)

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

    # Retrieve logger
    log = logging.getLogger(__name__)

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

    log.debug('Loaded variations from file: %s', data_type_corrected)

    return data_type_corrected

def assign_systematics(results):
    ''' Start by calculating the shift between each variation and the
    nominal result.  Then use the linearization method to calculate the
    size of the assigned systematic uncertainty for each source.
    '''
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

    return var_to_col 

def process(config_file):

    # Setup logging.
    log = logging.getLogger(__name__)

    start_time = time.time()

    # Load config from file.
    config = utils.load_config(config_file)

    # Load entire dataset, this
    # should only be done once
    # because it's 1.5 GB at load time.
    data = utils.load_dataset(config)

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
            log.info('Doing  %.3f < %s < %.3f' % (variations[par][index][0], par,
                                                          variations[par][index][1]))

            # get these cut values
            temp_dict = {}
            temp_dict[par] = variations[par][index]

            # get data
            temp_filter = utils.build_filter(data, temp_dict)
            temp_data = utils.build_dataframe(data, temp_filter)
            results[par][index] = utils.get_results(temp_data, bins, config)
            del temp_data

            end_var_time = time.time() - var_time
            log.info('Elapsed time %.3f' % end_var_time)

    # Using all variations, systematic
    # uncertainties are added to the dataframe.
    systematic_sources = assign_systematics(results)
    with open(config['systematics_file'], 'w') as outputfile:
        pickle.dump(systematic_sources, outputfile)

    # Write results to file. 
    results['nominal'].to_csv(config['output_filename'], index=False)

    exe_time = time.time() - start_time
    log.info('Finished execution in %.3f seconds.' % exe_time)

# This is quite clearly the main function.
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    args = ap.parse_args()

    process(args.config)
