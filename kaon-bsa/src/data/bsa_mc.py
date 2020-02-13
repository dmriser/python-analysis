#!/usr/bin/env python 
#
# Author: David Riser
# Date:   September 20, 2019
# File:   bsa_mc.py
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

def process(config_file):

    # Setup logging.
    log = logging.getLogger(__name__)

    start_time = time.time()

    # Load config from file.
    config = utils.load_config(config_file)
 
    nominal_conf = {}
    #nominal_conf['alpha'] = [0.55, 1.0]
    #nominal_conf['missing_mass'] = [0.0, 5.0]
    nominal_conf['p_mes'] = [0.35, 1.8]
    
    # Load entire dataset, this
    # should only be done once
    # because it's 1.5 GB at load time.
    data = utils.load_dataset(config)
    #data = data.dropna(how='any')
    print(data.info())
    
    # Applying nominal cuts to get the subset
    # of events that I consider good when
    # using the "best" cut values.
    #nominal_filter = utils.build_filter(data, nominal_conf)
    #nominal_data   = utils.build_dataframe(data, nominal_filter)

    # Use quantile binning to get integrated bins
    # for the axes listed in the configuration.
    #bins = setup_binning(config, nominal_data)
    bins = setup_binning(config, data)
    #kin_limits = find_kinematic_limits_in_bins(data, bins)
    #kin_limits.to_csv('kinematic_limits_mc.csv', index = False)
    
    # Calculate the results for the nominal subset of data.
    results = utils.get_results(data, bins, config)
    results.to_csv(config['output_filename'], index=False)
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    args = ap.parse_args()

    process(args.config)
