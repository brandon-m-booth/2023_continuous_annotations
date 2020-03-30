import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PlotPaganLog(input_file_path):
   df = pd.read_csv(input_file_path)
   project_entry_name = df['OriginalName'][0]
   pids = np.unique(df['Participant'])
   for pid in pids:
      anno_time = df.loc[df['Participant']==pid,'VideoTime'].values
      anno_vals = df.loc[df['Participant']==pid,'Value'].values
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
