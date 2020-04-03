import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PlotPaganLog(input_file_path):
   df = pd.read_csv(input_file_path)
   unique_experiment_names = np.unique(df['OriginalName'])
   for project_entry_name in unique_experiment_names:
      proj_df = df.loc[df['OriginalName'] == project_entry_name, :]
      pids = np.unique(proj_df['Participant'])
      for pid in pids:
         pid_df = proj_df.loc[proj_df['Participant']==pid,:]
         pid_df = pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable sort alg.
         anno_time = pid_df.loc[:,'VideoTime'].values
         anno_vals = pid_df.loc[:,'Value'].values
         plt.plot(anno_time, anno_vals)
      plt.title(project_entry_name)
      plt.legend(pids)
      plt.xlabel('Video Time')
      plt.ylabel('Annotation Value')
      plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_log', required=True, help='Path to log file exported from PAGAN')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   PlotPaganLog(args.input_log)
