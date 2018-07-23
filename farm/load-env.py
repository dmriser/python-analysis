#!/usr/bin/env python 

import json 
import os

with open('snapshot.json', 'r') as infile:
    json_dict = json.load(infile)

    # This might be a security risk, I am 
    # very sure where my json file comes from 
    # and what the content is.
    for key, value in json_dict.iteritems():
        os.environ[key] = value 
    
