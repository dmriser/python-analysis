#!/usr/bin/env python 

import argparse
import numpy as np 
import pandas as pd
import tqdm 
import vegas 

TO_RADIANS = np.pi/180.0
TO_DEGREES = 1/TO_RADIANS

def physics_model(phi, a):
    return a[0]*np.sin(phi*TO_RADIANS)/(1+a[1]*np.cos(phi*TO_RADIANS)+a[2]*np.cos(2*phi*TO_RADIANS))

def likelihood(data, theory, error):
    return np.exp(-0.5*np.sum(((data-theory)/error)**2))

def integrand(phi, data, error, model, a):
    theory = model(phi, a)
    f = likelihood(data, theory, error)
    return [f, f*a[0], f*a[1], f*a[2], 
            f*a[0]**2, f*a[1]**2, f*a[2]**2]

def perform_vegas(integrand, bounds, phi, data, error, model, n_iter, n_eval):
    vegas_integrator = vegas.Integrator(bounds)
    
    # burning some 
    vegas_integrator(lambda p: integrand(phi, data, error, model, p), 
                    nitn=4, 
                    neval=1000)
    
    result = vegas_integrator(lambda p: integrand(phi, data, error, model, p), 
                    nitn=n_iter, 
                    neval=n_eval)
    
    results = {}
    results['z'] = result[0].mean
    results['Q'] = result.Q
    results['exp_par1'] = result[1].mean/results['z']
    results['exp_par2'] = result[2].mean/results['z']
    results['exp_par3'] = result[3].mean/results['z']
    results['var_par1'] = result[4].mean/results['z']-results['exp_par1']**2
    results['var_par2'] = result[5].mean/results['z']-results['exp_par2']**2
    results['var_par3'] = result[6].mean/results['z']-results['exp_par3']**2
    return results


def fit(input_file, output_file, n_samples):

    # integration box for parameter s
    bounds = [[-1,1],[-1,1],[-1,1]]

    # load dataset 
    dataset = pd.read_csv(input_file)

    # setup container for output 
    output_data = {}
    output_data['axis'] = [] 
    output_data['axis_bin'] = [] 
    output_data['axis_min'] = [] 
    output_data['axis_max'] = [] 
    output_data['par_0'] = [] 
    output_data['par_1'] = []
    output_data['par_2'] = []
    output_data['err_0'] = [] 
    output_data['err_1'] = []
    output_data['err_2'] = []

    axes = np.unique(dataset.axis)
    for axis in axes:
        print('fitting %s' % axis)
        axis_data = dataset.query('axis == "%s"' % axis)
        
        axis_bins = np.unique(axis_data.axis_bin)
        for axis_bin in tqdm.tqdm(axis_bins):
            data = axis_data.query('axis_bin == %d' % axis_bin)

            # perform vegas integration 
            result = perform_vegas(integrand, bounds, data.phi, data.value, 
                                   data.stat, physics_model, 12, n_samples
                                   )

            output_data['axis'].append(axis)
            output_data['axis_bin'].append(axis_bin)
            output_data['axis_min'].append(data.axis_min.values[0])
            output_data['axis_max'].append(data.axis_max.values[0])
            output_data['par_0'].append(result['exp_par1'])
            output_data['par_1'].append(result['exp_par2'])
            output_data['par_2'].append(result['exp_par3'])
            output_data['err_0'].append(np.sqrt(result['var_par1']))
            output_data['err_1'].append(np.sqrt(result['var_par2']))
            output_data['err_2'].append(np.sqrt(result['var_par3']))

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # setup defaults 
    parser.add_argument('-i', '--input_file',  required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-n', '--n_samples',   required=False, default=4000, type=int)
    args = parser.parse_args()

    fit(args.input_file, args.output_file, args.n_samples)
