#!/bin/tcsh 

# setup python 
source conda.env 

# do the job 
./vegas-fitter.py -i=input.csv -o=out.csv -n=8000
