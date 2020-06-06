#!/usr/bin/env python3
# Author: Brandon M. Booth

import io
import os
import re
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

show_overlay_plot = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'util')))
import util

def GetUserIDFromFilename(filename):
   #id = os.path.basename(filename).split('_')[0]
   user_id = os.path.basename(filename).split('.')[0].split('_')[-1]
   if user_id.startswith('T') and user_id[1:].isnumeric():
      user_id = os.path.basename(filename).split('.')[0].split('_')[-2]
   return user_id

def CheckTSRError(target_signals_path, tsr_approxs_path):
   target_signal_files = glob.glob(os.path.join(target_signals_path, '*.csv'))
   tsr_approx_files = glob.glob(os.path.join(tsr_approxs_path, '*.csv'))

   # Get the id's of the annotation tasks
   task_ids = []
   for target_signal_file in target_signal_files:
      task_ids.append(GetUserIDFromFilename(target_signal_file))
   task_ids = sorted(task_ids)

   has_error = False
   for task_id in task_ids:
      # Find the target signal file for this task ID
      signal_df = None
      for target_signal_file in target_signal_files:
         target_signal_id = GetUserIDFromFilename(target_signal_file)
         if target_signal_id == task_id:
            signal_df = pd.read_csv(target_signal_file)
            time_col = [x for x in signal_df.columns if 'time' in x.lower()]
            if len(time_col) != 1:
               print("ERROR: Could not find a single time column in file %s"%(target_signal_file))
               return
            signal_df = signal_df.set_index(time_col[0])
      if signal_df is None:
         print("FIXME - Could not find the target signal file for task ID: %d"%(task_id))
         pdb.set_trace()
      
      tsr_files = []
      # Find the relevant TSR files for this target signal
      for tsr_approx_file in tsr_approx_files:
         tsr_task_id = GetUserIDFromFilename(tsr_approx_file)
         if tsr_task_id == task_id:
            tsr_files.append(tsr_approx_file)

      if not tsr_files:
         continue

      if show_overlay_plot:
         plt.figure()
         plt.plot(signal_df.index, signal_df.iloc[:,0], 'r-', label='Fused')
      # Compute and store the reconstruction error (SSE)
      t_vs_sse = []
      for tsr_file in tsr_files:
         tsr_df = pd.read_csv(tsr_file)
         #num_segments = tsr_df.shape[0]-1
         num_segs_regex = re.search(".+_T([0-9]+)\.csv", tsr_file)
         num_segments = int(num_segs_regex.group(1))
         time_col = [x for x in tsr_df.columns if 'time' in x.lower()]
         if len(time_col) != 1:
            print("ERROR: Could not find a single time column in file %s"%(tsr_file))
            return
         tsr_df = tsr_df.set_index(time_col[0])
         sse = util.GetTSRSumSquareError(tsr_df, signal_df)
         t_vs_sse.append((num_segments, sse))
         if show_overlay_plot and num_segments > 30:
            plt.plot(tsr_df.index, tsr_df.iloc[:,0], label=str(num_segments))
      if show_overlay_plot:
         plt.title('TSR Overlay Plot')
         plt.legend()

      t, sse = zip(*t_vs_sse)
      sort_idx = np.argsort(t)
      t = np.array(t)[sort_idx]
      sse = np.array(sse)[sort_idx]

      # Check for SSE monotonicity
      tol = 1e-10
      if np.any(np.diff(sse)-tol > 0):
         print("Task %d does not have a monotonic error function vs. num segments"%(task_id))
         has_error = True

      if show_overlay_plot:
         plt.figure()
         plt.plot(t,sse,'r-')
         plt.title('Task ID: %s'%(task_id))
         plt.xlabel('T')
         plt.ylabel('SSE')
         plt.show()

   if has_error:
      print("Error detected.  Check previous output")
   else:
      print("No TSR errors detected!")
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_avg_path', required=True, help='Path to folder containing the averaged (fused) annotation signals')
   parser.add_argument('--input_tsr_path', required=True, help='Path to folder containing the collection of tsr approximations to for varying numbers of segments for each annotation task')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CheckTSRError(args.input_avg_path, args.input_tsr_path)
