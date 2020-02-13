#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def asymmetry_model(x, z, pt):
    return 0.72 * x * (1 - x) * z * pt

def integral(f,a,b):
    return f(b) - f(a)

def integ_x(x):
    return (x**2 / 2.) - (x**3 / 3.)

def integ_q2(x):
    return x

def integ_z(x):
    return (x**2 / 2.)

def integ_pt(x):
    return (x**2 / 2.)

def asymmetry_model_x(x, nz, npt, nq2):
    return 0.72 * x * (1 - x) * nz * npt

def asymmetry_model_z(z, nx, npt, nq2):
    return 0.72 * z * nx * npt

def asymmetry_model_pt(pt, nx, nz, nq2):
    return 0.72 * pt * nx * nz

def asymmetry_model_q2(nx, nz, npt):
    return 0.72 * nx * nz * npt


if __name__ == '__main__':

    input_filename = 'database/fit/monte_carlo.csv'
    kinematic_limits_filename = 'kinematic_limits_mc.csv'
    kin = pd.read_csv(kinematic_limits_filename)
    kin['nx'] = np.zeros(len(kin))
    kin['nq2'] = np.zeros(len(kin))
    kin['npt'] = np.zeros(len(kin))
    kin['nz'] = np.zeros(len(kin))
    kin['true_value'] = np.zeros(len(kin))

    for i in range(len(kin)):
        kin['nx'].values[i] = integral(integ_x, kin['x_min'].values[i], kin['x_max'].values[i])
        kin['nz'].values[i] = integral(integ_z, kin['z_min'].values[i], kin['z_max'].values[i])
        kin['npt'].values[i] = integral(integ_pt, kin['pt_min'].values[i], kin['pt_max'].values[i])
        kin['nq2'].values[i] = integral(integ_q2, kin['q2_min'].values[i], kin['q2_max'].values[i])

        if kin['axis'].values[i] == 'x':
            xp = np.mean([kin['x_min'].values[i], kin['x_max'].values[i]])
            kin['true_value'].values[i] = asymmetry_model_x(xp, nz=kin['nz'].values[i], npt=kin['npt'].values[i],
                                                            nq2=kin['nq2'].values[i])
        elif kin['axis'].values[i] == 'z':
            xp = np.mean([kin['z_min'].values[i], kin['z_max'].values[i]])
            kin['true_value'].values[i] = asymmetry_model_z(xp, nx=kin['nx'].values[i], npt=kin['npt'].values[i],
                                                            nq2=kin['nq2'].values[i])
        elif kin['axis'].values[i] == 'pt':
            xp = np.mean([kin['pt_min'].values[i], kin['pt_max'].values[i]])
            kin['true_value'].values[i] = asymmetry_model_pt(xp, nz=kin['nz'].values[i], nx=kin['nx'].values[i],
                                                             nq2=kin['nq2'].values[i])
        elif kin['axis'].values[i] == 'q2':
            xp = np.mean([kin['q2_min'].values[i], kin['q2_max'].values[i]])
            kin['true_value'].values[i] = asymmetry_model_q2(npt=kin['npt'].values[i], nz=kin['nz'].values[i], nx=kin['nx'].values[i])
        

    print(kin)
    # Load the answers.
    data = pd.read_csv(input_filename)

    # Add some important imfornation 
    data['axis_value'] = data['axis_min'] + 0.5 * (data['axis_max'] - data['axis_min'])

    # Normalization factors (rough)
    xdata = data[data['axis'] == "x"]
    zdata = data[data['axis'] == "z"]
    ptdata = data[data['axis'] == "pt"]
    q2data = data[data['axis'] == "q2"]

    xpred = kin[kin['axis'] == "x"]
    q2pred = kin[kin['axis'] == "q2"]
    zpred = kin[kin['axis'] == "z"]
    ptpred = kin[kin['axis'] == "pt"]
    
    plt.subplot(2,2,1)
    plt.errorbar(xdata['axis_value'], xdata['par_0'], xdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(xdata['axis_value'], xpred['true_value'])
    plt.grid(alpha=0.2)
    plt.title('x')

    plt.subplot(2,2,2)
    plt.errorbar(zdata['axis_value'], zdata['par_0'], zdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(zdata['axis_value'], zpred['true_value'])
    plt.grid(alpha=0.2)
    plt.title('z')
    
    plt.subplot(2,2,3)
    plt.errorbar(ptdata['axis_value'], ptdata['par_0'], ptdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(ptdata['axis_value'],ptpred['true_value'])
    plt.grid(alpha=0.2)
    plt.title('pt')
    
    plt.subplot(2,2,4)
    plt.errorbar(q2data['axis_value'], q2data['par_0'], q2data['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(q2data['axis_value'], q2pred['true_value'])
    plt.grid(alpha=0.2)
    plt.title('q2')

    plt.tight_layout()
    plt.savefig('image/mc_compare_to_gen.pdf')
