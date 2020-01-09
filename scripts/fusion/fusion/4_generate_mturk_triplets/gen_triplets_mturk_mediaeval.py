#!/usr/bin/env python

import os
import sys
import csv
import pdb
import math
import glob
import random
import argparse
import numpy as np
import pandas as pd

max_hits_per_batch = 5000
triplet_keep_scalar = 1
keep_all_triplets = False

def GenerateTripletsMechanicalTurkMediaeval(dimension, const_intervals_path, base_url, output_folder):
   header = ['dim_label','dim_definition','ref_audio_url','a_audio_url','b_audio_url','ref_start','ref_end','a_start','a_end','b_start','b_end']
   if dimension == 'arousal':
      definition = 'Arousal is a measure of the level of calmness or excitement of a stimulus'
   elif dimension == 'valence':
      definition = 'Valence is a measure of the level of attractiveness (liking) or averseness (disliking) of a stimulus.'
   else:
      print('Unknown mediaeval dimension %s.  Must be either "arousal" or "valence".'%(dimension))
      return

   comparison_items = []
   const_intervals_files = glob.glob(os.path.join(const_intervals_path, '*.csv'))
   for const_interval_file in const_intervals_files:
      const_intervals_df = pd.read_csv(const_interval_file)
      for row_idx in range(const_intervals_df.shape[0]):
         subject_id = os.path.basename(const_interval_file).split('_')[0]
         item_url = os.path.join(base_url, subject_id+'.mp3')
         start_time = const_intervals_df.iloc[row_idx,0]
         end_time = const_intervals_df.iloc[row_idx,1]
         comparison_items.append((item_url, start_time, end_time))

   n = len(comparison_items)
   if keep_all_triplets:
      print("Generating all possible triplets")
      num_triplets = n*(n-1)*(n-2)/2
      mturk_batch_df = pd.DataFrame(data=np.empty((num_triplets, len(header))), columns=header, index=range(num_triplets))
      row_idx = 0
      for i in range(n):
         for j in range(n):
            if i == j:
               continue
            for k in range(j+1, n):
               if i == k:
                  continue
               ref = comparison_items[i]
               a = comparison_items[j]
               b = comparison_items[k]
               # Note: must match the item order in the header
               mturk_batch_df.iloc[row_idx,:] = [dimension, definition, ref[0], a[0], b[0], ref[1], ref[2], a[1], a[2], b[1], b[2]]
               row_idx += 1
   else:
      print("Generating random subset of all possible triplets")
      num_keep_triplets = int(math.ceil(n*math.log(n)*triplet_keep_scalar))
      mturk_batch_df = pd.DataFrame(data=np.empty((num_keep_triplets, len(header))), columns=header, index=range(num_keep_triplets))
      random.seed() # Random seed
      for triplet_idx in range(num_keep_triplets):
         is_valid = False
         while not is_valid:
            i = random.randint(0,n-1)
            j = random.randint(0,n-1)
            try:
               k = random.randint(j+1,n-1)
            except ValueError:
               continue
            if i != j and i != k:
               is_valid = True
         if triplet_idx%100 == 0:
            print('Generated %d out of %d total random triplets'%(triplet_idx, num_keep_triplets))

         ref = comparison_items[i]
         a = comparison_items[j]
         b = comparison_items[k]
         # Note: must match the item order in the header
         mturk_batch_df.iloc[triplet_idx,:] = [dimension, definition, ref[0], a[0], b[0], ref[1], ref[2], a[1], a[2], b[1], b[2]]

   # Shuffle the rows
   mturk_batch_df = mturk_batch_df.sample(frac=1)

   # Output mturk batches
   if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
   num_batches = int(math.ceil(float(mturk_batch_df.shape[0])/max_hits_per_batch))
   for i in range(num_batches):
      output_file_name = 'mediaeval_mturk_batch_%d.csv'%(i)
      start_row = i*max_hits_per_batch
      end_row = min((i+1)*max_hits_per_batch, mturk_batch_df.shape[0])
      mturk_batch_df.iloc[start_row:end_row+1,:].to_csv(os.path.join(output_folder, output_file_name), header=True, index=False)
   return

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--dimension', dest='dimension', required=True, help='Mediaeval annotation dimension (e.g. either "arousal" or "valence")')
   parser.add_argument('--input_path', dest='input_path', required=True, help='Path to folder containing the extracted constant intervals files')
   parser.add_argument('--source_base_url', dest='base_url', required=True, help='URL path to directory containing the source stimuli (e.g. audio or video)')
   parser.add_argument('--output_folder', dest='output_folder', required=True, help='Output folder path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)

   GenerateTripletsMechanicalTurkMediaeval(args.dimension, args.input_path, args.base_url, args.output_folder)
