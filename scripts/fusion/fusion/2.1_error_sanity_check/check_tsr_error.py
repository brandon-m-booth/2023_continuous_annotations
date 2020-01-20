#!/usr/bin/env python3
# Author: Brandon M. Booth

import io
import os
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

def CheckTSRError(target_signals_path, tsr_approxs_path):
   target_signal_files = glob.glob(os.path.join(target_signals_path, '*.csv'))
   tsr_approx_files = glob.glob(os.path.join(tsr_approxs_path, '*.csv'))

   # Get the id's of the annotation tasks
   task_ids = []
   for target_signal_file in target_signal_files:
      task_ids.append(int(os.path.basename(target_signal_file).split('_')[0]))
   task_ids = sorted(task_ids)

   for task_id in task_ids:
      # Find the target signal file for this task ID
      signal_df = None
      for target_signal_file in target_signal_files:
         target_signal_id = int(os.path.basename(target_signal_file).split('_')[0])
         if target_signal_id == task_id:
            signal_df = pd.read_csv(target_signal_file)
            signal_df = signal_df.set_index('Time_seconds')
      if signal_df is None:
         print("FIXME - Could not find the target signal file for task ID: %d"%(task_id))
         pdb.set_trace()
      
      tsr_files = []
      # Find the relevant TSR files for this target signal
      for tsr_approx_file in tsr_approx_files:
         tsr_task_id = int(os.path.basename(tsr_approx_file).split('_')[0])
         if tsr_task_id == task_id:
            tsr_files.append(tsr_approx_file)

      if show_overlay_plot:
         plt.figure()
         plt.plot(signal_df.index, signal_df.iloc[:,0], 'r-', label='Fused')
      # Compute and store the reconstruction error (MSE)
      t_vs_mse = []
      for tsr_file in tsr_files:
         tsr_df = pd.read_csv(tsr_file)
         num_segments = tsr_df.shape[0]-1
         tsr_df = tsr_df.set_index('Time')
         mse = util.GetTSRSumSquareError(tsr_df, signal_df)
         t_vs_mse.append((num_segments, mse))
         if show_overlay_plot and num_segments > 8 and num_segments < 13:
            plt.plot(tsr_df.index, tsr_df.iloc[:,0], label=str(num_segments))
      plt.title('TSR Overlay Plot')
      plt.legend()

      t, mse = zip(*t_vs_mse)
      sort_idx = np.argsort(t)
      t = np.array(t)[sort_idx]
      mse = np.array(mse)[sort_idx]
      plt.figure()
      plt.plot(t,mse,'r-')
      plt.title('Task ID: %d'%(task_id))
      plt.xlabel('T')
      plt.ylabel('MSE')
      plt.show()

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
