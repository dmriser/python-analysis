#
# File: run-fitter.py 
# Author: David Riser 
# Date: July 24, 2018 
#
# This is a simple file used to feed
# all of the input files one-by-one 
# into the single file fitting code
# 2.0-fitter.py in this folder. 
# 

import glob 
import os 

def get_files(dir):
    files = glob.glob(dir+'*.csv')
    return files 


def process():

    input_dir  = '../../database/phi/'
    output_dir = '../../database/fit/'
    nproc      = 8
    nreps      = 256

    for f in get_files(input_dir):
        name = f.split('/')[-1]
        print('Writing %s to %s' % (f, output_dir+name))

        system_command = 'python3 fitter.py -i=%s -o=%s -m=%d -n=%d' % (f, output_dir+name, nproc, nreps)
        os.system(system_command)

if __name__ == '__main__':
    process() 
