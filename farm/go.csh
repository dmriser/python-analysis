#!/bin/tcsh 

# setup python 
source conda.env 

# do the job 
./replica-fitter.py -i=input.csv -o=out.csv -b="no" -n=200 --n_proc=1
