#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

def chi2(data1, err1, data2, err2):
    return np.mean((data1-data2)**2/(err1**2+err2**2))

if __name__ == '__main__':

    data = {}
    data[1] = pd.read_csv('database/phi/sector_1.csv').query('axis == "x"')
    data[2] = pd.read_csv('database/phi/sector_2.csv').query('axis == "x"')
    data[3] = pd.read_csv('database/phi/sector_3.csv').query('axis == "x"')
    data[4] = pd.read_csv('database/phi/sector_4.csv').query('axis == "x"')
    data[5] = pd.read_csv('database/phi/sector_5.csv').query('axis == "x"')
    data[6] = pd.read_csv('database/phi/sector_6.csv').query('axis == "x"')


    pad_idx = 1
    for i in range(1,7):
        for j in range(1,7):
            #plt.subplot(6,6,pad_idx)
            #plt.scatter(data[i]['value'], data[j]['value'], edgecolor='k', alpha=0.7)
            #pad_idx += 1

            #metric = chi2(data[i]['value'], data[i]['stat'],
            #              data[j]['value'], data[j]['stat'])
            #print('Sectors ({}/{}) w/ chi2 = {}'.format(i,j,metric))

            if i != j:
                residual = data[i]['value'] - data[j]['value']

                plt.figure(figsize=(8,6))
                plt.subplot(2,1,1)
                plt.errorbar(data[i]['global_index'], data[i]['value'], data[i]['stat'], linestyle='',
                            marker='o', color='orange', alpha=0.8)
                plt.errorbar(data[j]['global_index'], data[j]['value'], data[j]['stat'], linestyle='',
                            marker='o', color='purple', alpha=0.8)
                plt.grid(alpha=0.1)

                plt.subplot(2,1,2)
                plt.hist(residual, bins=np.linspace(-0.2, 0.2, 20), edgecolor='k', color='orange', alpha=0.6)
                
                plt.savefig('image/sectors/compare_{}_{}.pdf'.format(i,j))
                plt.close()
                
        pad_idx += 1
            
