#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def asymmetry_model(x, z, pt):
    return 0.72 * x * (1 - x) * z * pt

def asymmetry_model_x(x):
    return 0.72 * x * (1 - x) * 0.5 * 0.5

def asymmetry_model_z(z):
    return 0.72 * z * 0.5 / 6.0

def asymmetry_model_pt(pt):
    return 0.72 * pt * 0.5 / 6.0

if __name__ == '__main__':

    input_filename = 'database/fit/monte_carlo.csv'

    # Load the answers.
    data = pd.read_csv(input_filename)

    # Add some important imfornation 
    data['axis_value'] = data['axis_min'] + 0.5 * (data['axis_max'] - data['axis_min']) 

    # Normalization factors (rough)
    nx = 1.0 / 6.0
    nz = 0.5
    npt = 0.5

    xdata = data[data['axis'] == "x"]
    zdata = data[data['axis'] == "z"]
    ptdata = data[data['axis'] == "pt"]
    q2data = data[data['axis'] == "q2"]

    x = np.linspace(0.1, 0.5, 100)
    x_model = asymmetry_model_x(x)

    z = np.linspace(0.1, 0.9, 100)
    z_model = asymmetry_model_z(z)

    pt = np.linspace(0.05, 1.2, 100)
    pt_model = asymmetry_model_pt(pt)

    plt.subplot(2,2,1)
    plt.errorbar(xdata['axis_value'], xdata['par_0'], xdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(x,x_model)
    plt.grid(alpha=0.2)
    plt.title('x')
    
    plt.subplot(2,2,2)
    plt.errorbar(zdata['axis_value'], zdata['par_0'], zdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(z,z_model)
    plt.grid(alpha=0.2)
    plt.title('z')
    
    plt.subplot(2,2,3)
    plt.errorbar(ptdata['axis_value'], ptdata['par_0'], ptdata['err_0'],
                 linestyle='', marker='o', color='k')
    plt.plot(pt,pt_model)
    plt.grid(alpha=0.2)
    plt.title('pt')
    
    plt.subplot(2,2,4)
    plt.errorbar(q2data['axis_value'], q2data['par_0'], q2data['err_0'],
                 linestyle='', marker='o', color='k')
    plt.grid(alpha=0.2)
    plt.title('q2')

    plt.tight_layout()
    #    plt.show()
    plt.savefig('image/mc_compare_to_gen.pdf')
