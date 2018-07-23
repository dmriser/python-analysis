#!/bin/tcsh 

# setup python 
source jlab-root6.env 

echo "Finished sourcing environment.  Now starting code."

# do the job 
./2.0-fitter.py -i=input.csv -o=out.csv -n=200 --n_proc=1

echo "Code finished, folder contents:"
ls 
