import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt_threshold_ms = 1000
show_plots = False

def CleanPaganLog(input_file_path, output_file_path):
   input_files = [input_file_path]
   if os.path.isdir(input_file_path):
      input_files = glob.glob(os.path.join(input_file_path, '*.csv'))

   output_files = []
   if os.path.isfile(output_file_path):
      if len(input_files) == 1:
         output_files.append(output_file_path)
      else:
         raise ValueError('Output path points to a single file while input points to a folder.  These must match.')
   else:
      if not os.path.isdir(output_file_path):
         os.makedirs(output_file_path)
      for input_file in input_files:
         output_files.append(os.path.join(output_file_path, os.path.basename(input_file)))
      
   for file_idx in range(len(input_files)):
      input_file = input_files[file_idx]
      output_file = output_files[file_idx]
      df = pd.read_csv(input_file)
      unique_experiment_names = np.unique(df['OriginalName'])
      output_df = None
      for project_entry_name in unique_experiment_names:
         proj_df = df.loc[df['OriginalName'] == project_entry_name, :]
         proj_pids = np.unique(proj_df['Participant'])
         for i in range(len(proj_pids)):
            proj_pid = proj_pids[i]
            proj_pid_df = proj_df.loc[proj_df['Participant']==proj_pid,:]
            proj_pid_df = proj_pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable sort alg.
            anno_time = proj_pid_df.loc[:,'VideoTime'].values
            anno_vals = proj_pid_df.loc[:,'Value'].values
            diff_anno_time = np.diff(anno_time)
            dt_threshold_mask = diff_anno_time > dt_threshold_ms
            add_nan_rows = []
            for dt_mask_idx in range(len(dt_threshold_mask)):
               if dt_threshold_mask[dt_mask_idx] and anno_vals[dt_mask_idx] != anno_vals[dt_mask_idx+1]:
                  start_row = proj_pid_df.iloc[dt_mask_idx,:]
                  end_row = proj_pid_df.iloc[dt_mask_idx+1,:]
                  start_row['VideoTime'] += 1
                  end_row['VideoTime'] -= 1
                  start_row['Value'] = end_row['Value'] = np.nan
                  add_nan_rows.append(start_row)
                  add_nan_rows.append(end_row)
            for new_row in add_nan_rows:
               proj_pid_df = proj_pid_df.append(new_row)
            proj_pid_df = proj_pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable

            if show_plots:
               plt.plot(anno_time, anno_vals, 'r--')
               plt.plot(proj_pid_df.loc[:,'VideoTime'].values, proj_pid_df.loc[:,'Value'].values, 'b')
               plt.show()

            if output_df is None:
               output_df = proj_pid_df
            else:
               output_df = pd.concat((output_df, proj_pid_df), axis=0, ignore_index=True)
      output_df.to_csv(output_file, index=False, header=True)
                  
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_log', required=True, help='Path to log file or folder containing such files exported from PAGAN')
   parser.add_argument('--output_log', required=True, help='Output folder for cleaned logs')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CleanPaganLog(args.input_log, args.output_log)
