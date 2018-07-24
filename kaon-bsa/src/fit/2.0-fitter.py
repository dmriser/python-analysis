#!/usr/bin/env python 

import argparse
import numpy as np 
import os 
import pandas as pd
import time

from contextlib import contextmanager 
from multiprocessing import Process, Queue

# single fit minimizer 
from scipy.optimize import minimize 

TO_RADIANS = np.pi/180.0
TO_DEGREES = 1/TO_RADIANS

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def create_replica(y, y_err):
    y_rep = [np.random.normal(yp,np.fabs(yp_err)) for yp,yp_err in zip(y,y_err)]
    return np.array(y_rep)

def bootstrap_worker(q, loss_function, physics_model, bounds,
                     phi, data, error, n_replicas=20):

    np.random.seed(os.getpid())

    results = []
    for irep in range(n_replicas):
        rep = create_replica(data, error) 
        pars,errs = perform_single(loss_function, physics_model, bounds, phi, rep, error)
        results.append(pars)

    q.put(results)

def bootstrap(loss_function, physics_model, bounds,
              phi, data, error, n_replicas=20):

    results = []
    for irep in range(n_replicas):
        rep = create_replica(data, error) 
        pars,errs = perform_single(loss_function, physics_model, bounds, phi, rep, error)
        results.append(pars)

    return results

def perform_bootstrap(loss_function, physics_model, bounds,
                      phi, data, error, n_replicas=20, n_cores=4):

    if n_cores is 1:
        results = bootstrap(loss_function, physics_model, bounds, 
                            phi, data, error, n_replicas)
        results = np.array(results, dtype=np.float32)

    elif n_cores > 1:
        q = Queue() 
        workers = []

        # spawn processes 
        reps_per_core = int(n_replicas/n_cores)
        for job in range(n_cores):
            workers.append(Process(target=bootstrap_worker, args=(q, loss_function, physics_model, bounds, phi, data, error, reps_per_core)))
            workers[job].start()
        
        # get results     
        result_pool = []
        for worker in workers:
            rv = q.get() 
            result_pool.append(rv)
            
        # end 
        for worker in workers:
            worker.join() 

        # aggregate results so they look normal 
        results = [item for sublist in result_pool for item in sublist]
        results = np.array(results, dtype=np.float32)

    else:
        print('What are you trying to do asking for %d cores?' % n_cores)

    pars = []
    errs = []
    for ipar in range(3):
        pars.append(np.average(results[:,ipar]))
        errs.append(np.std(results[:,ipar]))    

    return pars, errs 

def physics_model(phi, a):
    return a[0]*np.sin(phi*TO_RADIANS)/(1+a[1]*np.cos(phi*TO_RADIANS)+a[2]*np.cos(2*phi*TO_RADIANS))

def loss_function(data, theory, error):
    return np.sum(((data-theory)/error)**2)

def perform_single(loss_function, model, bounds, 
                   phi, data, error):

    func = lambda p: loss_function(data, model(phi, p), error)
    
    bad_fit = True

    while bad_fit:
        result = minimize(func, x0=np.random.uniform(-1,1,3), bounds=bounds)
        identity = np.identity(3)
        err = np.sqrt(np.array(np.matrix(result.hess_inv * identity).diagonal()))
        bad_fit = not result.success

    return result.x, err[0]    


def fit(input_file, output_file, bounds, n_reps, n_proc):

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
        print('Fitting %s' % axis)
        axis_data = dataset.query('axis == "%s"' % axis)
        
        axis_bins = np.unique(axis_data.axis_bin)
        for axis_bin in axis_bins:
            data = axis_data.query('axis_bin == %d' % axis_bin)

            # perform single fitting 
            with timer('Performing single fit'):
                pars, errs = perform_bootstrap(loss_function, physics_model, bounds, 
                                               data.phi, data.value, data.stat, n_reps, n_proc)

            output_data['axis'].append(axis)
            output_data['axis_bin'].append(axis_bin)
            output_data['axis_min'].append(data.axis_min.values[0])
            output_data['axis_max'].append(data.axis_max.values[0])
            output_data['par_0'].append(pars[0])
            output_data['par_1'].append(pars[1])
            output_data['par_2'].append(pars[2])
            output_data['err_0'].append(errs[0])
            output_data['err_1'].append(errs[1])
            output_data['err_2'].append(errs[2])

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # setup defaults 
    parser.add_argument('-i', '--input_file',  required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-b', '--bounded', default='no')
    parser.add_argument('-n', '--n_replicas', default=100, type=int)
    parser.add_argument('-m', '--n_proc', default=4, type=int)
    args = parser.parse_args()


    # parameter limits for parameters
    if args.bounded == 'no':
        bounds = [[-1,1],[-1,1],[-1,1]]
    elif args.bounded == 'tight':
        bounds = [[-1,1],[-1,1],[-1,1]]
    elif args.bounded == 'single':
        bounds = [[-1,1],[-1e-9,1e-9],[-1e-9,1e-9]]
    else:
        bounds = [[-1,1],[-1,1],[-1,1]]

    fit(args.input_file, args.output_file, bounds, args.n_replicas, args.n_proc)
