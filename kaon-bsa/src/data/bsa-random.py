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
import time
import bsautils as utils
import os

def setup_binning(config, data):
    ''' Return dictionary of bins chosen by quantile method. '''

    log = logging.getLogger(__name__)

    bins = {}
    for axis in config['axes']:
        if axis != 'z':
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

    for key, value in bins.items():
        log.debug('Setup binning %s: %s', key, value)

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
    for par_name, levels in data.items():
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
    log = logging.getLogger(__name__)
    
    dont_write = ['sector{}'.format(s) for s in range(1,7)]
    dont_write.append('nominal')
    for conf in results.keys():
        if conf not in dont_write:
            for val in results[conf].keys():
                log.debug('Assigning systematics for config {} with value {}'.format(conf, val))
                log.debug('Type of object {}'.format(type(results[conf][val]['value'])))

                # Is this safe?  They could be in different orders.  It has been checked visually.
                results[conf][val]['shift'] = results['nominal']['value'] - results[conf][val]['value']

    shift_df, var_to_col = utils.get_linearized_error(results)
    var_to_col['sys_0'] = 'beam_pol'
    results['nominal'] = pd.merge(left=results['nominal'],
                                  right=shift_df,
                                  on=['axis', 'global_index'])

    return var_to_col 

def process(config_file, samples):

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

    varfile = os.path.dirname(__file__) + '/../../variations.json'
    variations = load_variations(varfile)

    # Use quantile binning to get integrated bins
    # for the axes listed in the configuration.
    bins = setup_binning(config, nominal_data)

    # Calculate the results for the nominal subset of data.
    results = {}
    results['nominal'] = utils.get_results(nominal_data, bins, config)

    # Calculate the results for each sector.
    for sector in range(1,7):

        sector_data = data[data['sector'] == sector]

        for imc in range(samples):
            
            var_time = time.time()
            log.info('Doing sector {}'.format(sector))
            random_filter = utils.get_random_config(sector_data, variations)
            random_data = utils.build_dataframe(sector_data, random_filter)
            sect_result = utils.get_results(random_data, bins, config)
            elapsed_time = time.time() - var_time
            log.info('Elapsed time %.3f' % elapsed_time)
            output_filename = str(config['database_path'] + 'phi/random/sector_' + str(sector) + '_{}.csv'.format(imc))
            sect_result.to_csv(output_filename, index=False)


    exe_time = time.time() - start_time
    log.info('Finished execution in %.3f seconds.' % exe_time)

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    ap.add_argument('-s', '--samples', type=int, default=100, help='Number of random parameter sets')
    args = ap.parse_args()

    process(args.config, args.samples)
