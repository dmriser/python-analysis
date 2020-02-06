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

def build_bayesian_optimized_config(
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
    conf['missing_mass'] = [0.0, 5.0]
    conf['p_mes'] = [p_mes_min, p_mes_max]
    return conf 
    
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
    utils.randomize_sector(data)
    
    # Applying nominal cuts to get the subset
    # of events that I consider good when
    # using the "best" cut values.
    if args.bayes_opt_pars is not None:
        log.info("Using Bayesian Optimized parameters for nominal.")
        with open(args.bayes_opt_pars, 'rb') as f:
            bayes_pars = pickle.load(f)

        params = {str(k):float(v) for k,v in bayes_pars['params'].items()}
        bayes_conf = build_bayesian_optimized_config(**params)
        nominal_filter = utils.build_filter(data,bayes_conf)

    else:
        nominal_filter = utils.build_filter(data)

    nominal_data   = utils.build_dataframe(data, nominal_filter)

    # Use quantile binning to get integrated bins
    # for the axes listed in the configuration.
    bins = setup_binning(config, nominal_data)

    # Calculate the results for the nominal subset of data.
    results = {}
    results['nominal'] = utils.get_results(nominal_data, bins, config)

    # Calculate the results for each sector.
    for sector in range(1,7):
        var_time = time.time()
        log.info('Doing sector {}'.format(sector))

        sector_data = nominal_data[nominal_data['sector'] == sector]
        sect_result = utils.get_results(sector_data, bins, config)

        elapsed_time = time.time() - var_time
        log.info('Elapsed time %.3f' % elapsed_time)

        output_filename = str(config['database_path'] + 'phi/sector_' + str(sector) + '.csv')
        sect_result.to_csv(output_filename, index=False)

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
    with open(config['systematics_file'], 'wb') as outputfile:
            pickle.dump(systematic_sources, outputfile)
    #pickle.dump(systematic_sources, config['systematics_file'])
    
    # Write results to file. 
    results['nominal'].to_csv(config['output_filename'], index=False)
    
    # Write other results too. 
    dont_write = ['sector'.format(s) for s in range(1,7)]
    dont_write.append('nominal')
    for key in results.keys():
        if key not in dont_write:
            for conf in results[key]:
                output_filename = str(config['database_path'] + 'phi/variation_' + key + '_' +str(conf) + '.csv')
                results[key][conf].to_csv(output_filename, index=False)

    exe_time = time.time() - start_time
    log.info('Finished execution in %.3f seconds.' % exe_time)

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser() 
    ap.add_argument('-c', '--config', required=True, help='Configuration file in JSON format.')
    ap.add_argument('--bayes_opt_pars', default=None)
    args = ap.parse_args()

    process(args.config)
