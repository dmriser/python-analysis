#!/usr/bin/env python 

import numpy as np 
import ROOT 

def generate_root_histogram(title='histo',n_fills=1000):
    h = ROOT.TH1F(title, title, 100, -3, 3)
    h.FillRandom("gaus", n_fills)
    return h

def convert_histo_to_numpy(histo=None):
    print(type(histo))

    vals = np.zeros(shape=(histo.GetNbinsX(),))
    bins = np.zeros(shape=(histo.GetNbinsX(),))
    errs = np.zeros(shape=(histo.GetNbinsX(),))

    for i in range(histo.GetNbinsX()):
        bins[i] = histo.GetBinCenter(i+1)
        vals[i] = histo.GetBinContent(i+1)
        errs[i] = histo.GetBinError(i+1)

    return bins, vals, errs 

if __name__ == '__main__':
    hist = generate_root_histogram() 
    print('Generated histogram has {} entries'.format(hist.GetEntries()))

    bins, vals, errs = convert_histo_to_numpy(histo=hist)
    print(vals, errs)

    import matplotlib.pyplot as plt 
    plt.rc('font', family='serif')
    plt.rc('font', size=16)
    plt.errorbar(bins, vals, errs, linestyle='', color='black', marker='o', alpha=0.8)
    plt.grid(alpha=0.2)
    plt.show()
