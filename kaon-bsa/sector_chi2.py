""" 

Load with/without systematics and calculate
the weighted chi2.  Compare before/after 
systematics.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def weighted_chi2(x, xerr):
    weights = 1.0 / xerr**2
    xbar = np.sum(x * weights) / np.sum(weights)
    chi2 = np.sum(weights * (x - xbar)**2)
    return chi2

def load_data():
    data = []
    data_sys = []
    for i in range(1,7):
        data.append(pd.read_csv('database/phi/bootstrap_est_sector_{}.csv'.format(i)))
        data[i-1] = data[i-1][data[i-1]['axis'] != "missing_mass"]
        data_sys.append(pd.read_csv('database/phi/random/systematics_{}.csv'.format(i)))

    return data, data_sys

if __name__ == "__main__":

    data, data_sys = load_data()

    npoints = 240

    chi2 = np.empty(npoints)
    chi2_sys = np.empty(npoints)
    chi2_sys_no_shift = np.empty(npoints)
    for i in range(npoints):
        
        x = np.array([data[s].iloc[i]['value'] for s in range(6)])

        # Quadrature Sum of Sys0 and Stat.
        xerr = np.array([data[s].iloc[i]['stat'] for s in range(6)])
        xerr = xerr**2
        for s in range(6):
            xerr[s] += data[s].iloc[i]['sys_0']**2
        xerr = np.sqrt(xerr)

        chi2[i] = weighted_chi2(x, xerr)


        xerr = xerr**2
        for s in range(6):
            xerr[s] += data_sys[s].iloc[i]['mc_std']**2
        xerr = np.sqrt(xerr)

        chi2_sys_no_shift[i] = weighted_chi2(x, xerr)
        
        x = np.array([data_sys[s].iloc[i]['mc_mean'] for s in range(6)])
        chi2_sys[i] = weighted_chi2(x, xerr)


    #print(chi2 / 6.0)
    #print(chi2_sys / 6.0)
    #print(chi2_sys_no_shift / 6.0)

    goodpts = np.where(chi2 / 6. < 1.001)[0]
    goodpts_sys = np.where(chi2_sys / 6. < 1.001)[0]
    goodpts_sys_no_shift = np.where(chi2_sys_no_shift / 6. < 1.001)[0]

    print(f'{len(goodpts)} good points without sys')
    print(f'{len(goodpts_sys)} good points with sys')
    print(f'{len(goodpts_sys_no_shift)} good points with sys (no shift)')

    
    # Start Plotting
    example_axis = "x"
    example_axis_bin = 1

    idx = np.logical_and(data[0]['axis'] == example_axis,
                         data[0]['axis_bin'] == example_axis_bin)
    example_data = data[0][idx]
    example_data_sys = data_sys[0][idx]

    plt.errorbar(example_data['phi'], example_data['value'], example_data['stat'],
                 marker='o', linestyle='', label='Stat')
    plt.errorbar(example_data_sys['phi'], example_data_sys['mc_mean'],
                 np.sqrt(example_data['stat']**2 + example_data_sys['mc_std']**2),
                 marker='o', linestyle='', label='Sys + Stat')
    plt.legend(frameon=False)
    plt.savefig('compare_methods_sector_x_1.pdf', bbox_inches='tight')
    plt.close()
    
