#!/usr/bin/env python3
#Author: Brandon M Booth

import os
import sys
import pdb
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

def GenMovieViolenceCSV(input_csm_pkl_path, output_csm_path):
   with open(input_csm_pkl_path, 'rb') as f:
      csm_list = pickle.load(f)
      csm_data_dict = {'uuid':[], 'id':[], 'title':[], 'age':[], 'length_min':[], 'violence_rating':[], 'violence_text':[]}
      for csm_entry in csm_list:
         if 'contentGrid' in csm_entry and csm_entry['contentGrid']:
            csm_data_dict['uuid'].append(csm_entry['uuid'])
            csm_data_dict['id'].append(csm_entry['id'])
            csm_data_dict['title'].append(csm_entry['title'])
            csm_data_dict['age'].append(csm_entry['ageRating'])
            if 'product' in csm_entry.keys() and 'length' in csm_entry['product'].keys():
               units = csm_entry['product']['length']['units']
               value = csm_entry['product']['length']['value']
               if units == 'minutes':
                  csm_data_dict['length_min'].append(value)
               else:
                  print("Unknown units in the CSM file: %s. Fix me!"%(units))
                  pdb.set_trace()
            else:
               csm_data_dict['length_mins'].append(np.nan)
            if 'violence' in csm_entry['contentGrid'].keys():
               csm_data_dict['violence_rating'].append(csm_entry['contentGrid']['violence']['rating'])
               violence_text = csm_entry['contentGrid']['violence']['text']
               if not violence_text.startswith('"'):
                  violence_text = '"%s"'%(violence_text)
               csm_data_dict['violence_text'].append(violence_text)
            else:
               csm_data_dict['violence_rating'].append(np.nan)
               csm_data_dict['violence_text'].append('N/A')

      df = pd.DataFrame(data=csm_data_dict)
      df.to_csv(output_csm_path, index=False, header=True)

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--csm_pkl', required=True, help='Path to the pickle file containing CSM ratings')
   parser.add_argument('--output_csv', required=True, help='Output CSV file path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   GenMovieViolenceCSV(args.csm_pkl, args.output_csv)
