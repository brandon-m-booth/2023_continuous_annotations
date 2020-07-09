#!/usr/bin/env python3
# Author: Brandon M Booth

import os
import re
import sys
import pdb
import glob
import math
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def FixNans(signal_path, tsr_path, output_path, subj_id_signal_regex='(\d+)_average\.csv', subj_id_tsr_regex='(\d+)_tsr_T.*\.csv', seg_tsr_regex='.*_tsr_T(\d+)\.csv', show_plots=False):
   signal_files = glob.glob(os.path.join(signal_path, '*.csv'))
   if len(signal_files) == 0:
      print("No signal files found for path: %s.  Please check the path and try again"%(signal_path))
      return

   tsr_files = glob.glob(os.path.join(tsr_path, '*.csv'))
   if len(tsr_files) == 0:
      print("No TSR files found for path: %s.  Please check the path and try again"%(tsr_path))
      return

   if not os.path.exists(output_path):
      os.makedirs(output_path)

   subj_signal_files = {}
   # Store the signal files by subject id
   for signal_file in signal_files:
      subject_id = re.search(subj_id_signal_regex, os.path.basename(signal_file)).group(1)
      if subject_id not in subj_signal_files.keys():
         subj_signal_files[subject_id] = signal_file
      else:
         print("Error! Multiple signal files detected for subject %s"%(subject_id))
         pdb.set_trace()

   # Group the TSR files per subject
   subj_tsr_files = {}
   for tsr_file in tsr_files:
      subject_id = re.search(subj_id_tsr_regex, os.path.basename(tsr_file)).group(1)
      if subject_id not in subj_tsr_files.keys():
         subj_tsr_files[subject_id] = []
      subj_tsr_files[subject_id].append(tsr_file)

   # Fix NaNs for all TSR files each subject
   subj_tsr_error_dict = {}
   for subj_id in tqdm(subj_tsr_files.keys(), desc='Gathering TSR info'):
      if not subj_id in subj_signal_files.keys():
         print("Skipping subject %s because no source signal file was found."%(subj_id))
         continue

      signal_df = pd.read_csv(subj_signal_files[subj_id])
      signal_df = signal_df.set_index(signal_df.columns[0])

      if show_plots:
         fig, ax = plt.subplots(1,1)
         ax.plot(signal_df.index, signal_df.values, label='Signal')

      # Get list of domain intervals with NaN values
      nan_intervals = []
      start_nan_time = None
      for idx in range(signal_df.shape[0]):
         if start_nan_time is None:
            if np.isnan(signal_df.iloc[idx,0]):
               if idx > 0:
                  start_nan_time = signal_df.index[idx-1]
               else:
                  start_nan_time = signal_df.index[0]
         else:
            if not np.isnan(signal_df.iloc[idx,0]):
               end_nan_time = signal_df.index[idx]
               nan_intervals.append((start_nan_time, end_nan_time))
               start_nan_time = None
      if start_nan_time is not None:
         end_nan_time = signal_df.index[-1]
         nan_intervals.append((start_nan_time, end_nan_time))

      for tsr_file in subj_tsr_files[subj_id]:
         tsr_df = pd.read_csv(tsr_file)

         # Insert NaN intervals and interpolate
         nan_interval_times = np.array(nan_intervals).flatten()
         nan_df = pd.DataFrame(data=np.vstack((nan_interval_times,len(nan_interval_times)*[np.nan])).T, columns=tsr_df.columns)
         out_df = pd.concat((nan_df, tsr_df))
         out_df.sort_values(by=out_df.columns[0], inplace=True)
         out_df.index = out_df.iloc[:,0]
         out_df.interpolate(method='index', inplace=True)

         # Insert NaN values just inside each NaN interval in the domain
         time_eps = 1e-5
         pinched_nan_intervals = [(x+time_eps,y-time_eps) for x,y in nan_intervals]
         pinched_nan_interval_times = np.array(pinched_nan_intervals).flatten()
         pinched_nan_df = pd.DataFrame(data=np.vstack((pinched_nan_interval_times,len(pinched_nan_interval_times)*[np.nan])).T, columns=tsr_df.columns)
         out_df = pd.concat((pinched_nan_df, out_df))
         out_df.sort_values(by=out_df.columns[0], inplace=True)

         # Fill in NaN values within the pinched intervals
         idx = 0
         for pinched_nan_interval in pinched_nan_intervals:
            while idx < out_df.shape[0]:
               time = out_df.iloc[idx,0]
               if time >= pinched_nan_interval[0] and time <= pinched_nan_interval[1]:
                  out_df.iloc[idx,1] = np.nan
               if time >= pinched_nan_interval[1]:
                  break
               idx += 1
         
         # Output fixed TSR file
         out_df.to_csv(os.path.join(output_path, os.path.basename(tsr_file)), index=False, header=True)

         if show_plots:
            num_segments = re.search(seg_tsr_regex, os.path.basename(tsr_file)).group(1)
            ax.plot(out_df.iloc[:,0], out_df.iloc[:,1], label=str(num_segments))

      if show_plots:
         ax.legend()
         ax.set_title(subj_id + ' TSR Plots')
         plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_signal_path', dest='input_signal_path', required=True, help='Path to the folder containing the fused signals for each task')
   parser.add_argument('--input_tsr_path', dest='input_tsr_path', required=True, help='Path to the folder containing many TSR approximations to the signal for each task')
   parser.add_argument('--output_path', dest='output_path', required=True, help='Output folder for the optimal TSR signals (one per subject)')
   parser.add_argument('--subj_id_regex', required=True, help='Regex to use on the input signal path\'s file names, including parenthesis denoting the group containing the subject ID')
   parser.add_argument('--subj_id_tsr_regex', required=True, help='Regex to use on the input tsr path\s file names, including parenthesis denoting the group containing the subject ID')
   parser.add_argument('--segment_num_regex', required=True, help='Regex to use on the input tsr path\s file names, including parenthesis denoting the group containing the number of TSR segments')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Displays plots of each subject\'s TSR approximations, reconstruction error, and agreement')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   FixNans(args.input_signal_path, args.input_tsr_path, args.output_path, args.subj_id_regex, args.subj_id_tsr_regex, args.segment_num_regex, args.show_plots)
