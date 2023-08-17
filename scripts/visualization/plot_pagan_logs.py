import os
import sys
import glob
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def PlotPaganLog(input_file_path):
   input_files = [input_file_path]
   if os.path.isdir(input_file_path):
      input_files = glob.glob(os.path.join(input_file_path, '*.csv'))
   for input_file in input_files:
      df = pd.read_csv(input_file)
      unique_experiment_names = np.unique(df['OriginalName'])
      palette = itertools.cycle(sns.color_palette())
      for project_entry_name in unique_experiment_names:
         proj_df = df.loc[df['OriginalName'] == project_entry_name, :]
         pids = np.unique(proj_df['Participant'])
         external_pids = []
         num_rows_plot = int(math.floor(math.sqrt(len(pids))))
         num_cols_plot = int(math.ceil(math.sqrt(len(pids))))
         if num_rows_plot*num_cols_plot < len(pids):
            num_rows_plot = num_cols_plot
         anno_fig, axs = plt.subplots(num_rows_plot, num_cols_plot)
         comb_fig, comb_ax = plt.subplots()
         for i in range(len(pids)):
            pid = pids[i]
            pid_df = proj_df.loc[proj_df['Participant']==pid,:]
            pid_df = pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable sort alg.
            anno_time = pid_df.loc[:,'VideoTime'].values
            anno_vals = pid_df.loc[:,'Value'].values
            external_pid = pid_df['ExternalPID'].iloc[0]
            external_pids.append(external_pid)
            c = next(palette)
            if hasattr(axs, '__iter__'):
               subplot_ax = axs[int(i/num_cols_plot),i%num_cols_plot]
            else:
               subplot_ax = axs
            subplot_ax.plot(anno_time, anno_vals, 'o-', color=c)
            subplot_ax.set_ylim([-100,100])
            subplot_ax.set_title(external_pid)
            comb_ax.plot(anno_time, anno_vals, 'o-', color=c)
         comb_ax.set_title(project_entry_name)
         comb_ax.legend(external_pids)
         comb_ax.set_xlabel('Video Time')
         comb_ax.set_ylabel('Annotation Value')
         comb_ax.set_ylim([-100,100])
         plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_log', required=True, help='Path to log file or folder containing such files exported from PAGAN')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   PlotPaganLog(args.input_log)
