##################################################################
# Looks up the eidr in the website, and returns imdb id if found
#
# Input:
#   eidr_files: file produced by extract_eidr.py
#   out_file: File to write results to
#
# Output:
#   Tab-separated file indexed by UUID
##################################################################
import re
import sys
import requests

# This imports tqdm. 
# If you dont have tqdm, it defaults to the identity function.
try:
    from tqdm import *
except ImportError:
    tqdm = lambda x: x

BASE_URL = "https://ui.eidr.org/view/content?id={}"
FIELD = """<AlternateID xsi:type="IMDB">(.*)?</AlternateID>"""

####################################
# Read output to skip those files
####################################
skip = []
with open(sys.argv[2]) as inpt:
    for line in inpt:
        uuid, _ = line.strip().split("\t")
        skip.append(uuid)

# Loads EIDRs into memory
print("Loading EIDRs...")

eidrs = {}
with open(sys.argv[1]) as inpt:
    for line in inpt:
        params = line.strip().split("\t")
        uuid = params[0]
        
        if uuid in skip:
            continue

        for z in params[2:]:
            try:
                x, y = z.split(":")
                if x == "eidr":
                    eidrs[uuid] = y
            except:
                pass

print("Done loading EIDRs...")

# Gets the website and searches for the imdbid
with open(sys.argv[2], 'w') as outpt:
    for uuid, eidr in tqdm(eidrs.items()):
        url = BASE_URL.format(eidr)
        webpage = requests.get(url)
        if webpage.status_code == 200:
            imdbid = re.findall(FIELD, webpage.text)
            if len(imdbid) > 0:
                outpt.write("{}\t{}\n".format(uuid, imdbid[0]))

