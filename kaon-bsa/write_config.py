import json 

# This configuration should be save into
# some external file and loaded.
config = {}
config['axes'] = ['x', 'z', 'pt', 'q2']
config['z_range'] = [0.25, 0.75]
config['n_bins'] = 10
config['file_path'] = '/home/dmriser/data/inclusive/inclusive_kaon_small.csv'
config['sample_size'] = 100000
config['file_compression'] = 'bz2'
config['variation_file'] = '../../variations.json'

with open('config.json', 'w') as output_file:

    # Write for inspection on the output stream.
    print(json.dumps(config, sort_keys=True, indent=4, 
                     separators=(',', ': ')))

    # Write to file. 
    output_file.write(json.dumps(config, sort_keys=True, indent=4, 
                     separators=(',', ': ')))

