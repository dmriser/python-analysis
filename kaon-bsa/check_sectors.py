#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

def chi2(data1, err1, data2, err2):
    return np.mean((data1-data2)**2/(err1**2+err2**2))

if __name__ == '__main__':

    for axis in ['x', 'q2', 'z', 'pt']:
        data = {}
        data[1] = pd.read_csv('database/phi/sector_1.csv').query('axis == "{}"'.format(axis))
        data[2] = pd.read_csv('database/phi/sector_2.csv').query('axis == "{}"'.format(axis))
        data[3] = pd.read_csv('database/phi/sector_3.csv').query('axis == "{}"'.format(axis))
        data[4] = pd.read_csv('database/phi/sector_4.csv').query('axis == "{}"'.format(axis))
        data[5] = pd.read_csv('database/phi/sector_5.csv').query('axis == "{}"'.format(axis))
        data[6] = pd.read_csv('database/phi/sector_6.csv').query('axis == "{}"'.format(axis))


        pad_idx = 1
        for i in range(1,7):
            for j in range(1,7): 
                metric = chi2(data[i]['value'], data[i]['stat'],
                              data[j]['value'], data[j]['stat'])
                print('Sectors ({}/{}) w/ chi2 = {}'.format(i,j,metric))

                if i != j:
                    xbins = data[i]['axis_bin'].unique()
                
                    plt.figure(figsize=(16,6))

                    for ii, xbin in enumerate(xbins):
                        top = data[i].query('axis_bin == {}'.format(xbin))
                        bot = data[j].query('axis_bin == {}'.format(xbin))
                        residual = top['value'] - bot['value']

                        plt.subplot(2, len(xbins), ii+1)
                        plt.errorbar(top['phi'], top['value'], top['stat'], linestyle='',
                                     marker='o', color='red', alpha=0.8)
                        plt.errorbar(bot['phi'], bot['value'], bot['stat'], linestyle='',
                                     marker='o', color='blue', alpha=0.8)
                        plt.axhline(0, linestyle='--', linewidth=1, color='k')
                        plt.grid(alpha=0.1)
                        plt.title('Sectors ({},{}), Bin {}, Axis {}'.format(
                            i, j, xbin, axis
                        ))
                    
                        plt.subplot(2, len(xbins), ii + 1 + len(xbins))
                        plt.hist(residual, bins=np.linspace(-0.2, 0.2, 15), edgecolor='k', color='red', alpha=0.6)
                    
                    plt.tight_layout()
                    plt.savefig('image/sectors/compare_{}_{}_{}.pdf'.format(axis,i,j))
                    plt.close()
        
