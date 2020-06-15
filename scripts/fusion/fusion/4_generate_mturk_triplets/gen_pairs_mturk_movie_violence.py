#!/usr/bin/env python

import os
import sys
import csv
import pdb
import math
import glob
import json
import random
import argparse
import numpy as np
import pandas as pd

max_hits_per_batch = 5000
pairs_keep_scalar = 2
keep_all_pairs = True

def ClipMovieFile(movie_file_path, clip_time_df, output_folder):
   output_movie_paths = []
   for i in range(clip_time_df.shape[0]):
      start_time = str(clip_time_df.iloc[i,0])
      end_time = str(clip_time_df.iloc[i,1])
      duration = str(clip_time_df.iloc[i,1]-clip_time_df.iloc[i,0])
      clip_suffix = '_s'+start_time+'e'+end_time
      output_file_path = os.path.join(output_folder, os.path.basename(movie_file_path).split('.')[0]+clip_suffix+'.mp4')
      #os.system('ffmpeg -y -i %s -ss %s -t %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(movie_file_path, start_time, duration, output_file_path))
      output_movie_paths.append(output_file_path)

   return output_movie_paths

def GeneratePairsMechanicalTurkMovieViolence(config_json_path):
   header = ['A', 'B', 'a_start', 'a_end', 'b_start', 'b_end']

   with open(config_json_path) as config_json_fp:
      config = json.load(config_json_fp)

   if not os.path.isdir(config['output_path']):
      os.makedirs(config['output_path'])

   comparison_items = []
   for clip_data in config['clip_data']:
      const_intervals_df = pd.read_csv(clip_data['const_intervals_path'], header=None)
      fused_seg_df = pd.read_csv(clip_data['fused_seg_seq_path'])

      # Clip the mp4 file into segments corresponding the the constant intervals
      clip_time_df = pd.DataFrame(data=const_intervals_df.values, columns=['start_time_sec', 'end_time_sec']).astype(float)
      for row_idx in range(clip_time_df.shape[0]):
         clip_time_df.iloc[row_idx,0] = fused_seg_df.iloc[const_intervals_df.iloc[row_idx,0],0]
         clip_time_df.iloc[row_idx,1] = fused_seg_df.iloc[const_intervals_df.iloc[row_idx,1],0]

      for row_idx in range(clip_time_df.shape[0]):
         duration = clip_time_df.iloc[row_idx,1] - clip_time_df.iloc[row_idx,0]
         # Include transitions between constant intervals in the shortest adjacent constant interval window
         if row_idx > 0:
            prev_duration = clip_time_df.iloc[row_idx-1,1] - clip_time_df.iloc[row_idx-1,0]
            if prev_duration > duration:
               clip_time_df.iloc[row_idx,0] = fused_seg_df.iloc[const_intervals_df.iloc[row_idx-1,1],0]
         if row_idx < clip_time_df.shape[0]-1:
            next_duration = clip_time_df.iloc[row_idx+1,1] - clip_time_df.iloc[row_idx+1,0]
            if next_duration > duration:
               clip_time_df.iloc[row_idx,1] = fused_seg_df.iloc[const_intervals_df.iloc[row_idx+1,0],0]
   
      clip_time_df -= config['lag_shift_seconds']
      clip_time_df.iloc[-1,1] += config['lag_shift_seconds']
      clip_time_df = clip_time_df.clip(lower=0.0)
      clipped_movie_paths = ClipMovieFile(clip_data['source_mp4_path'], clip_time_df, config['output_path'])

      for row_idx in range(const_intervals_df.shape[0]):
         item_url = os.path.join(config['base_url'], os.path.basename(clipped_movie_paths[row_idx]))
         start_time = const_intervals_df.iloc[row_idx,0]
         end_time = const_intervals_df.iloc[row_idx,1]
         comparison_items.append((item_url, start_time, end_time))

   n = len(comparison_items)
   if keep_all_pairs:
      print("Generating all possible pairs")
      num_pairs = int(n*(n-1)/2)
      mturk_batch_df = pd.DataFrame(data=np.empty((num_pairs, len(header))), columns=header, index=range(num_pairs))
      row_idx = 0
      for i in range(n):
         for j in range(i+1,n):
            if i == j:
               continue
            a = comparison_items[i]
            b = comparison_items[j]
            # Note: must match the item order in the head
            mturk_batch_df.iloc[row_idx,:] = [a[0], b[0], a[1], a[2], b[1], b[2]]
            row_idx += 1
   else:
      print("Generating random subset of all possible pairs")
      num_keep_pairs = int(math.ceil(math.log(n)*pairs_keep_scalar))
      mturk_batch_df = pd.DataFrame(data=np.empty((num_keep_pairs, len(header))), columns=header, index=range(num_keep_pairs))
      random.seed() # Random seed
      for pair_idx in range(num_keep_pairs):
         is_valid = False
         while not is_valid:
            i = random.randint(0,n-1)
            try:
               j = random.randint(i+1,n-1)
            except ValueError:
               continue
            is_valid = True
         if pair_idx%100 == 0:
            print('Generated %d out of %d total random pair'%(pair_idx, num_keep_pairs))

         a = comparison_items[i]
         b = comparison_items[j]
         # Note: must match the item order in the header
         mturk_batch_df.iloc[pair_idx,:] = [a[0], b[0], a[1], a[2], b[1], b[2]]

   # Shuffle the rows
   mturk_batch_df = mturk_batch_df.sample(frac=1)

   # Output mturk batches
   num_batches = int(math.ceil(float(mturk_batch_df.shape[0])/max_hits_per_batch))
   for i in range(num_batches):
      output_file_name = 'movie_violence_mturk_batch_%d.csv'%(i)
      start_row = i*max_hits_per_batch
      end_row = min((i+1)*max_hits_per_batch, mturk_batch_df.shape[0])
      mturk_batch_df.iloc[start_row:end_row+1,:].to_csv(os.path.join(config['output_path'], output_file_name), header=True, index=False)
   return

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--config_json', required=True, help='Path to the json config file')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)

   GeneratePairsMechanicalTurkMovieViolence(args.config_json)
