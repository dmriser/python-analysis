#!/usr/bin/env python 

import argparse 
import logging 
import numpy as np 
import pandas as pd 

from scipy.stats import chi2

def chi2_test_with_zero(x, err):
    '''
    The null hypothesis is that the asymmetry
    is zero for all kinematics.
    '''
    res = np.sum( (x/err)**2 )
    ndf = len(x)
    p = chi2.cdf(res,ndf)
    return 1-p

def main(input_filename):

    # Setup logging.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        )
    log = logging.getLogger(__file__)    
    log.info('Generating stats from %s.' % input_filename)
    
    # Load and report on data. 
    data = pd.read_csv(input_filename)
    n_points, n_cols = data.shape    
    log.info('Loaded data with shape (%d,%d)' % (n_points, n_cols))

    # Define observables for checking. 
    observables = ['err_0','err_1','err_2',
                   'par_0','par_1','par_2',
                   'sys_total_0','sys_total_1',
                   'sys_total_2']

    # Find all axes and process each one individually. 
    axes = data.axis.unique() 
    for axis in axes:
        log.debug('Found axis %s' % axis)

        # Subset of data on this axis.
        d = data[data.axis == axis]
        assert(len(d) > 0)
        
        for obs in observables:
            avg = np.average(np.abs(d[obs].values))
            std_dev = np.std(d[obs].values)
            log.info('Axis %s, Obs %s, Avg = %.3e, Std. = %.3e' % (axis, obs, avg, std_dev))

        
        # Now for each observable itself 
        for i in range(3):
            val = np.abs(d['par_%d' % i])
            err = np.sqrt(d['err_%d' % i]**2 + d['sys_total_%d' % i]**2)
            ratio = np.average(err/val)
            log.info('Axis %s, Coef = %d, dA/A = %.3e' % (axis, i, ratio))

            # Now check the compatability with zero 
            p = chi2_test_with_zero(val, err)
            log.info('Axis %s, Coef = %d, p = %f' % (axis, i, p))

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser() 

    ap.add_argument(
        '-i', 
        '--input_file',
        required=True,
        type=str
        )

    args = ap.parse_args()

    main(args.input_file) 
