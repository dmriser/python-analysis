#!/usr/bin/env python 
#
# File: 1.0-fitter.py 
# Author: David Riser 
# Date: July 20, 2018
#
# This fitter is for use on 
# an entire file produced by
# src/data/bsa.py.  Many files
# are produced, one for each 
# variation.  This code will 
# accept one of those, and 
# output the name of your 
# choice. 

import numpy as np 
import pandas as pd 
import time 

# This library: https://github.com/dmriser/phifitter
from phifitter import (loss, fitter, physics_model)

def process_file(input_file, output_file):
    print('Processing %s to %s.' % (input_file, output_file))

    data = pd.read_csv(input_file)
    print('Data loaded with shape:', data.shape)
    print('Data columns:', data.columns)

    # Setup fitter. 
    bounds = [[-1, 1], [-1, 1], [-1, 1]]
    #    rep_fitter = fitter.ReplicaFitter(
    #        model=physics_model.BeamSpinAsymmetryModel(),
    #        loss_function=loss.chi2, 
    #        bounds=bounds,
    #        n_replicas=20,
    #        n_cores=4
    #        )
    model = physics_model.BeamSpinAsymmetryModel()
    rep_fitter = fitter.SingleFitter(
        model=model, 
        loss_function=loss.chi2,
        bounds=bounds
        )

    # Setup output data object. 
    output_data = {}
    output_data['axis'] = []
    output_data['axis_bin'] = []
    output_data['axis_min'] = []
    output_data['axis_max'] = []
    output_data['axis_value'] = []
    output_data['loss'] = []
    output_data['quality'] = []
    output_data['par_0'] = []
    output_data['par_1'] = []
    output_data['par_2'] = []
    output_data['err_0'] = []
    output_data['err_1'] = []
    output_data['err_2'] = []

    axes = np.unique(data['axis'])
    for axis in axes:
        print('Now processing axis:', axis)
        axis_data = data.loc[data['axis'] == axis]

        axis_bins = np.unique(axis_data['axis_bin'])
        n_bins = len(axis_bins)
        print('Found %d bins.' % n_bins)
        
        for b in axis_bins:
            start_time = time.time()
            print('Processing bin %d.' % b)
            bin_data = axis_data.loc[axis_data['axis_bin'] == b]
            err = bin_data.stat
            res = rep_fitter.fit(bin_data.phi, bin_data.value, err) 
            elapsed_time = time.time() - start_time 
            print('Finished in %.3f seconds' % elapsed_time)
            print('Fitter returned:', res)

            output_data['axis'].append(axis)
            output_data['axis_bin'].append(b)
            output_data['axis_min'].append(bin_data['axis_min'].values[0])
            output_data['axis_max'].append(bin_data['axis_max'].values[0])
            output_data['axis_value'].append(bin_data['axis_min'].values[0] + 0.5*(bin_data['axis_max'].values[0] - bin_data['axis_min'].values[0]))
            output_data['loss'].append(res['loss'])
            output_data['quality'].append(res['quality'])
            output_data['par_0'].append(res['fit_parameters'][0])
            output_data['par_1'].append(res['fit_parameters'][1])
            output_data['par_2'].append(res['fit_parameters'][2])
            output_data['err_0'].append(res['fit_errors'][0])
            output_data['err_1'].append(res['fit_errors'][1])
            output_data['err_2'].append(res['fit_errors'][2])

    # Write to storage. 
    out = pd.DataFrame(output_data)
    out.to_csv(output_file, index=False)

if __name__ == '__main__':
    process_file('../../results/phi/sys.csv', '../../database/fit/test-july20.csv')

