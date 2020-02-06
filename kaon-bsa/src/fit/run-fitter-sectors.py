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
        if 'bootstrap_est' in f:

            name = f.split('/')[-1]
            print('Writing %s to %s' % (f, output_dir+name))

            system_command = 'python3 fitter.py -i=%s -o=%s -m=%d -n=%d' % (f, output_dir+name, nproc, nreps)
            os.system(system_command)

if __name__ == '__main__':
    process() 
