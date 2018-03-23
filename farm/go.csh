#!/bin/tcsh 

# setup python 
source conda.env 

# do the job 
./fit-playground.py -i=input.csv -o=out.csv -n=4000
