#!/usr/bin/env python3
#####################################################################
# Reads the pickle file and produces a tsv file indexed by UUID
#
# Inputs: 
#   pickle_file: Pickle file with CSM data
#   outfile: File to write to
#
# Output is UUID \t title \t (list of products IDs)
# where products IDs are TYPE:VALUE
#####################################################################

import pickle
import sys

# This imports tqdm. If you dont have tqdm, it defaults to the identity function.
try:
    from tqdm import *
except ImportError:
    tqdm = lambda x: x

if len(sys.argv) < 2:
    print("Usage: ./extract_eidr.py pickle_file outfile")
    sys.exit(1)

data = pickle.load(open(sys.argv[1], 'rb'))
print("Loaded: {}".format(len(data)))

with open(sys.argv[2], 'w') as fout:
    for m in tqdm(data):
        uuid = m['uuid']
        title = m['title']
        prod = m['product']

        if 'references' in prod:
            fout.write('{}\t{}\t'.format(uuid, title))
            for z in prod['references']:
                fout.write("{}:{}\t".format(z['type'], z['value'])) #Write tab-separated TYPE:VALUE to file
            fout.write("\n")

print("Done")

