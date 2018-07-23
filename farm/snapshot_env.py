#!/usr/bin/env python 

import os 
import json 

# Save environmental variables into 
# json file. 


env_dict = dict(os.environ)

with open('snapshot.json', 'w') as outfile: 
    json.dump(env_dict, outfile, 
              sort_keys=True, 
              indent=4, 
              separators=(',', ': '))
    

