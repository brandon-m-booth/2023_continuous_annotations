#!/usr/bin/env python

import os
import sys
import pdb
import numpy as np
import pandas as pd
import argparse

def GetIntervalIndexFromTimeRange(intervals_times, start_time, end_time):
   eps = 0.0000001
   for idx in range(intervals_times.shape[0]):
      if np.sum(np.abs(intervals_times[idx] - [start_time, end_time])) < eps:
         return idx

   print('ERROR: Could not find an interval with the specified time range. FIX ME!')
   pdb.set_trace()
   return np.nan

def GeneratePairsMatrixFromMTurk(mturk_batch_path, output_path):
   mturk_df = pd.read_csv(mturk_batch_path)

   # For tracking each new tuple: (clip, start, end)
   unique_clips = []
   pairs_mat = np.zeros((mturk_df.shape[0],2)).astype(int)

   # Create dict for quick lookup for const intervals paths
   for i in range(mturk_df.shape[0]):
      a_clip_name = os.path.basename(mturk_df.loc[i,'Input.A'])
      b_clip_name = os.path.basename(mturk_df.loc[i,'Input.B'])
      a_start = mturk_df.loc[i,'Input.a_start']
      b_start = mturk_df.loc[i,'Input.b_start']
      a_end = mturk_df.loc[i,'Input.a_end']
      b_end = mturk_df.loc[i,'Input.b_end']

      try:
         a_idx = unique_clips.index((a_clip_name, a_start, a_end))
      except ValueError:
         unique_clips.append((a_clip_name, a_start, a_end))
         a_idx = len(unique_clips)-1
      try:
         b_idx = unique_clips.index((b_clip_name, b_start, b_end))
      except ValueError:
         unique_clips.append((b_clip_name, b_start, b_end))
         b_idx = len(unique_clips)-1

      # Generate triplet (ref,a,b) if ||ref-a|| < ||ref-b||
      if mturk_df.loc[i,'Answer.choice'] == 'optionA':
         pairs_mat[i] = [a_idx, b_idx]
      elif mturk_df.loc[i,'Answer.choice'] == 'optionB':
         pairs_mat[i] = [b_idx, a_idx]
      else:
         print('Unknown choice option in MTurk results file. FIX ME!')
         pdb.set_trace()

   batch_name = os.path.basename(mturk_batch_path).split('.')[0]
   pairs_df = pd.DataFrame(data=pairs_mat)
   pairs_df.to_csv(os.path.join(output_path, batch_name+'_pairs.csv'), header=False, index=False)

   pairs_code_df = pd.DataFrame(data=np.array(unique_clips), columns=['clip_name', 'start', 'end'])
   pairs_code_df.to_csv(os.path.join(output_path, batch_name+'_pairs_code.csv'), header=True, index=False)
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--mturk_batch_path', required=True, help='Path to the mturk batch results file')
   parser.add_argument('--output_path', required=True, help='')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)

   GeneratePairsMatrixFromMTurk(args.mturk_batch_path, args.output_path)
