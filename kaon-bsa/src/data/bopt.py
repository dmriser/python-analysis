#!/usr/bin/env python OA
#
# Author: David Riser
# Date:   July 2, 2018
# File:   bopt.py
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

from bayes_opt import BayesianOptimization
from functools import partial

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
    
    # Randomize the sectors to test
    # if we can at least get the same
    # answer.
    utils.randomize_sector(data)
    
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

    exe_time = time.time() - start_time
    log.info('Finished execution in %.3f seconds.' % exe_time)

def weighted_chi2(x, xerr):
    weights = 1.0 / xerr**2
    xbar = np.sum(x * weights) / np.sum(weights)
    chi2 = np.sum(weights * (x - xbar)**2)
    return chi2

def process_par_set(data, config,
                    alpha_min=0.55, alpha_max=1.0,
                    dist_cc_min=-1.0, dist_cc_max=1.0,
                    dist_cc_theta_min=-1.0, dist_cc_theta_max=1.0,
                    dist_dcr1_min=-1.0, dist_dcr1_max=1.0,
                    dist_dcr3_min=-1.0, dist_dcr3_max=1.0,
                    dist_ecsf_min=-1.0, dist_ecsf_max=1.0,
                    dist_ecu_min=-1.0, dist_ecu_max=1.0,
                    dist_ecv_min=-1.0, dist_ecv_max=1.0,
                    dist_ecw_min=-1.0, dist_ecw_max=1.0,
                    dist_ec_edep_min=-1.0, dist_ec_edep_max=1.0,
                    dist_vz_min=-1.0, dist_vz_max=1.0,
                    missing_mass_min=0.0, missing_mass_max=5.0,
                    p_mes_min=0.35, p_mes_max=2.0):
    """ Process one dataset. """

    conf = {}
    conf['alpha'] = [alpha_min, alpha_max]
    conf['dist_cc'] = [dist_cc_min, dist_cc_max]
    conf['dist_cc_theta'] = [dist_cc_theta_min, dist_cc_theta_max]
    conf['dist_dcr1'] = [dist_dcr1_min, dist_dcr1_max]
    conf['dist_dcr3'] = [dist_dcr3_min, dist_dcr3_max]
    conf['dist_ecsf'] = [dist_ecsf_min, dist_ecsf_max]
    conf['dist_ecu'] = [dist_ecu_min, dist_ecu_max]
    conf['dist_ecv'] = [dist_ecv_min, dist_ecv_max]
    conf['dist_ecw'] = [dist_ecw_min, dist_ecw_max]
    conf['dist_ec_edep'] = [dist_ec_edep_min, dist_ec_edep_max]
    conf['dist_vz'] = [dist_vz_min, dist_vz_max]
    conf['missing_mass'] = [missing_mass_min, missing_mass_max]
    conf['p_mes'] = [p_mes_min, p_mes_max]

    
    data_filter = utils.build_filter(data,conf)
    df = utils.build_dataframe(data, data_filter)

    # how many points?!
    npoints = len(config['axes']) * config['n_bins'] * 12
    sector_values = np.zeros(shape=(npoints,6))
    sector_errors = np.zeros(shape=(npoints,6))
    for i in range(1,7):
        sector_data = df[df['sector'] == i]
        sector_result = utils.get_results(sector_data, bins, config)
        sector_values[:,i-1] = sector_result['value'].values
        sector_errors[:,i-1] = sector_result['stat'].values

    var = np.var(sector_values, axis=1)
    metric = np.exp(-0.5 * np.sum(var))
    return metric

    #xw = np.sum(np.multiply(sector_values, 1/sector_errors**2))
    #wsum = np.sum(1/sector_errors**2,axis=1)
    #mu = xw/wsum
    #res = (np.subtract(sector_values,mu))**2 / sector_errors**2
    #likelihood = np.exp(-0.5 * np.sum(res))
    #return likelihood
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    ap.add_argument('--init_points', type=int, default=5)
    ap.add_argument('--n_iter', type=int, default=10)
    args = ap.parse_args()

    parameter_bounds = {
        'alpha_min':(0.5, 0.6),
        'dist_cc_min':(-1.1, -1.0),
        'dist_cc_max':(1.0, 1.1),
        'dist_cc_theta_min':(-1.1,-0.9),
        'dist_cc_theta_max':(0.9,1.1),
        'dist_dcr1_min':(-1.1,-0.9),
        'dist_dcr1_max':(0.9,1.1),
        'dist_dcr3_min':(-1.1,-0.9),
        'dist_dcr3_max':(0.9,1.1),
        'dist_ecsf_min':(-1.1,-0.9),
        'dist_ecsf_max':(0.9,1.1),
        'dist_ecu_min':(-1.1,-0.9),
        'dist_ecu_max':(0.9,1.1),
        'dist_ecv_min':(-1.1,-0.9),
        'dist_ecv_max':(0.9,1.1),
        'dist_ecw_min':(-1.1,-0.9),
        'dist_ecw_max':(0.9,1.1),
        'dist_ec_edep_min':(-1.1,-0.9),
        'dist_ec_edep_max':(0.9,1.1),
        'dist_vz_min':(-1.1,-0.9),
        'dist_vz_max':(0.9,1.1),
        'missing_mass_min':(0.0,1.75),
        'p_mes_min':(0.3,0.4),
        'p_mes_max':(1.6,1.8)        
    }

    # Load the configuration file and entire
    # dataset (once).
    config = utils.load_config(args.config)
    data = utils.load_dataset(config)
    
    # Nominal data to get binning 
    nominal_filter = utils.build_filter(data)
    nominal_data   = utils.build_dataframe(data, nominal_filter)
    bins = setup_binning(config, nominal_data)

    objective_fn = partial(process_par_set, data=data, config=config)
    
    opt = BayesianOptimization(
        f = objective_fn,
        pbounds = parameter_bounds,
        random_state = 1
    )

    opt.maximize(init_points=args.init_points, n_iter=args.n_iter)

    print(opt.max)

    output_file = open('best_params.pkl', 'wb')
    pickle.dump(opt.max, output_file)
    output_file.close()
