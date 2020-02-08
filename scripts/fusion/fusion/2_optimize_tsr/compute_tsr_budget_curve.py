#!/usr/bin/env python3
# Author: Brandon M Booth

import os
import re
import sys
import pdb
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_extract_const_intervals')))
from extract_const_intervals import ExtractConstIntervalsSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'util')))
from util import GetTSRSumSquareError

show_final_plots = True

def ComputeTSRBudgetCurve(signal_path, tsr_path, output_file_path, subj_id_signal_regex='(\d+)_average\.csv', subj_id_tsr_regex='(\d+)_tsr_T.*\.csv', seg_tsr_regex='.*_tsr_T(\d+)\.csv'):
   if not os.path.exists(os.path.dirname(output_file_path)):
      os.makedirs(os.path.dirname(output_file_path))

   signal_files = glob.glob(os.path.join(signal_path, '*.csv'))
   if len(signal_files) == 0:
      print("No signal files found for path: %s.  Please check the path and try again"%(signal_path))
      return

   tsr_files = glob.glob(os.path.join(tsr_path, '*.csv'))
   if len(tsr_files) == 0:
      print("No TSR files found for path: %s.  Please check the path and try again"%(tsr_path))
      return

   subj_signal_files = {}
   # Store the signal files by subject id
   for signal_file in signal_files:
      subject_id = re.search(subj_id_signal_regex, signal_file).group(1)
      if subject_id not in subj_signal_files.keys():
         subj_signal_files[subject_id] = signal_file
      else:
         print("Error! Multiple signal files detected for subject %s"%(subject_id))
         pdb.set_trace()

   # Group the TSR files per subject
   subj_tsr_files = {}
   for tsr_file in tsr_files:
      subject_id = re.search(subj_id_tsr_regex, tsr_file).group(1)
      if subject_id not in subj_tsr_files.keys():
         subj_tsr_files[subject_id] = []
      subj_tsr_files[subject_id].append(tsr_file)

   # Compute and store information about each TSR file per subject
   subj_tsr_error_dict = {}
   for subj_id in tqdm(subj_tsr_files.keys(), desc='Gathering TSR info'):
      if not subj_id in subj_signal_files.keys():
         pdb.set_trace()
         print("Skipping subject %s because no source signal file was found."%(subj_id))
         continue

      #print("Extracting TSR info for subject: %s"%(subj_id))
      signal_df = pd.read_csv(subj_signal_files[subj_id])
      signal_df = signal_df.set_index('Time_seconds')
      subj_tsr_info = []
      for tsr_file in subj_tsr_files[subj_id]:
         tsr_df = pd.read_csv(tsr_file)
         tsr_df = tsr_df.set_index(tsr_df.columns[0])
         subject_id = re.search(subj_id_tsr_regex, tsr_file).group(1)

         num_tsr_segments = int(re.search(seg_tsr_regex, tsr_file).group(1))
         constant_intervals = ExtractConstIntervalsSignal(tsr_df)
         num_const_intervals = len(constant_intervals)#tsr_df.shape[0]-1
         sse = GetTSRSumSquareError(tsr_df, signal_df)

         subj_tsr_info.append((num_tsr_segments, num_const_intervals, sse, tsr_file))
      subj_tsr_info = sorted(subj_tsr_info, key=lambda info_item: info_item[0]) # Sort by number of TSR segments
      subj_tsr_info_num_tsr_segs, subj_tsr_info_num_const_intervals, subj_tsr_info_sse, subj_tsr_info_files = zip(*subj_tsr_info)
      subj_tsr_error_dict[subj_id] = {'num_tsr_segments': subj_tsr_info_num_tsr_segs, 'num_const_intervals': subj_tsr_info_num_const_intervals, 'sse': subj_tsr_info_sse, 'tsr_files': subj_tsr_info_files}

   # Setup for optimization
   subject_ids = [*subj_tsr_error_dict.keys()]
   total_num_tsr_seg_files = 0
   for subj_id in subject_ids:
      total_num_tsr_seg_files += len(subj_tsr_error_dict[subj_id]['num_tsr_segments'])
   opt_df = pd.DataFrame(data=np.zeros((total_num_tsr_seg_files-len(subject_ids), 3)), columns=['Total TSR Segments', 'Total Const Intervals', 'Total SSE'], index=range(total_num_tsr_seg_files-len(subject_ids)))

   # Perform a greedy search, adding TSR segments one at a time to minimize the total SSE
   subj_tsr_idx = len(subject_ids)*[0] # Index to track TSR information per subject
   for i in tqdm(range(opt_df.shape[0]), desc='Optimizing'):
      # Update the output optimization dataframe
      num_tsr_segs = 0
      num_const_intervals = 0
      sse = 0
      for j in range(len(subject_ids)):
         subj_id = subject_ids[j]
         subj_id_idx = subj_tsr_idx[j]
         num_tsr_segs += subj_tsr_error_dict[subj_id]['num_tsr_segments'][subj_id_idx]
         num_const_intervals += subj_tsr_error_dict[subj_id]['num_const_intervals'][subj_id_idx]
         sse += subj_tsr_error_dict[subj_id]['sse'][subj_id_idx]
      opt_df.loc[i,'Total TSR Segments'] = num_tsr_segs
      opt_df.loc[i,'Total Const Intervals'] = num_const_intervals
      opt_df.loc[i,'Total SSE'] = sse
      
      # Compute the SSE/num_tsr_segs gradient
      sse_tsr_segs_grads = len(subject_ids)*[np.nan]
      for j in range(len(subject_ids)):
         subj_id = subject_ids[j]
         cur_subj_info_idx = subj_tsr_idx[j]
         if cur_subj_info_idx+1 < len(subj_tsr_error_dict[subj_id]['num_tsr_segments']):
            subj_tsr_seg_diff = subj_tsr_error_dict[subj_id]['num_tsr_segments'][cur_subj_info_idx+1] - subj_tsr_error_dict[subj_id]['num_tsr_segments'][cur_subj_info_idx]
            subj_sse_diff = subj_tsr_error_dict[subj_id]['sse'][cur_subj_info_idx+1] - subj_tsr_error_dict[subj_id]['sse'][cur_subj_info_idx]
         else:
            subj_tsr_seg_diff = 1
            subj_sse_diff = np.inf

         sse_tsr_seg_grad = float(subj_sse_diff)/subj_tsr_seg_diff
         sse_tsr_segs_grads[j] = sse_tsr_seg_grad

      best_subj_idx = np.argmin(sse_tsr_segs_grads)
      subj_tsr_idx[best_subj_idx] += 1

   # Save output
   opt_df.to_csv(output_file_path, index=False, header=True)

   if show_final_plots:
      plt.ion()
      plt.figure()
      plt.plot(opt_df['Total TSR Segments'], opt_df['Total SSE'])
      plt.xlabel('Total TSR Segments')
      plt.ylabel('Total SSE')
      plt.show()
      plt.figure()
      plt.plot(opt_df['Total Const Intervals'], opt_df['Total SSE'])
      plt.xlabel('Total Const Intervals')
      plt.ylabel('Total SSE')
      plt.ioff()
      plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_signal_path', dest='input_signal_path', required=True, help='Path to the folder containing the fused signals for each task')
   parser.add_argument('--input_tsr_path', dest='input_tsr_path', required=True, help='Path to the folder containing many TSR approximations to the signal for each task')
   parser.add_argument('--output_file_path', dest='output_file_path', required=True, help='File path for optimization output')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ComputeTSRBudgetCurve(args.input_signal_path, args.input_tsr_path, args.output_file_path)
