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
import tqdm

def asymmetry_model(x, z, pt):
    return 0.72 * x * (1 - x) * z * pt

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

def find_kinematic_limits_in_bins(data, bins):
    """ For each kinematic bin, there are a range
    of values for the other three variables.  """
    axes = bins.keys()

    for axis in axes:
        if axis not in data.columns:
            raise Exception("The column {} was in bins, but not in data.".format(axis))

    data_dict = {}
    data_dict['axis'] = []
    data_dict['axis_bin'] = []
    #data_dict['x_min'] = []
    #data_dict['x_max'] = []
    #data_dict['x_avg'] = []
    #data_dict['q2_min'] = []
    #data_dict['q2_max'] = []
    #data_dict['q2_avg'] = []
    #data_dict['z_min'] = []
    #data_dict['z_max'] = []
    #data_dict['z_avg'] = []
    #data_dict['pt_min'] = []
    #data_dict['pt_max'] = []
    #data_dict['pt_avg'] = []
    data_dict['gen'] = []
    data_dict['volume'] = []
    
    for axis in axes:
        for i in range(len(bins[axis]) - 1):
            print("Working on {} between [{},{}]".format(axis, bins[axis][i], bins[axis][i+1]))

            indices = np.logical_and(data[axis] > bins[axis][i], data[axis] < bins[axis][i+1])
            sample = data[indices]
            npoints = len(sample)
            #if npoints > 10000:
            #    npoints = 10000
                
            total = 0.
            for isamp in tqdm.tqdm(range(npoints)):
                total += asymmetry_model(
                    x = sample['g_x'].values[isamp],
                    z = sample['g_z'].values[isamp],
                    pt = sample['g_pt'].values[isamp]
                )
                #total += sample['g_asym'].values[isamp]
                
            data_dict['gen'].append(total / float(npoints))
            #data_dict['gen'].append(total)
            data_dict['axis'].append(axis)
            data_dict['axis_bin'].append(i)

            volume = 1.
            for second_axis in axes:
                amin, amax = min(sample[second_axis]), max(sample[second_axis])
                volume *= (amax - amin)

            data_dict['volume'].append(volume)
            
    return pd.DataFrame(data_dict)
                
def process(config_file):

    # Setup logging.
    log = logging.getLogger(__name__)

    start_time = time.time()

    # Load config from file.
    config = utils.load_config(config_file)
 
    nominal_conf = {}
    nominal_conf['alpha'] = [0.55, 1.0]
    nominal_conf['missing_mass'] = [0.0, 5.0]
    nominal_conf['p_mes'] = [0.35, 2.0]
    
    # Load entire dataset, this
    # should only be done once
    # because it's 1.5 GB at load time.
    data = utils.load_dataset(config)
    bins = setup_binning(config, data)
    kin_limits = find_kinematic_limits_in_bins(data, bins)
    kin_limits.to_csv('kinematic_limits_mc.csv', index = False)
    
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    args = ap.parse_args()

    process(args.config)
