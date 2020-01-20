#!/usr/bin/env python3
# Author: Brandon M Booth

import os
import sys
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_extract_const_intervals')))
from extract_const_intervals import ExtractConstIntervalsSignal

def OptimizeTSR(input_regexp):
   bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]

   tsr_files = glob.glob(input_regexp)
   if len(tsr_files) == 0:
      print("No input files found for path: %s.  Please check the path and try again"%(input_regexp))
      return

   tsr_info = {}
   for tsr_file in tsr_files:
      tsr_df = pd.read_csv(tsr_file)
      tsr_df = tsr_df.set_index(tsr_df.columns[0])

      constant_intervals = ExtractConstIntervalsSignal(tsr_df)
      interval_durations = [x[1]-x[0] for x in constant_intervals]

      num_segments = tsr_df.shape[0]-1
      tsr_info[num_segments] = {'interval_durations': interval_durations}

   num_segments_tested = len(tsr_info.keys())
   num_plot_rows = int(math.ceil(math.sqrt(num_segments_tested)))
   num_plot_cols = int(math.ceil(num_segments_tested/float(num_plot_rows)))
   fig, ax = plt.subplots(num_plot_rows, num_plot_cols)
   plt.title("Histogram of interval durations per number of segments")
   sorted_num_segments = sorted(tsr_info.keys())
   segment_idx = 0
   for row_idx in range(num_plot_rows):
      for col_idx in range(num_plot_cols):
         if segment_idx >= num_segments_tested:
            continue
         tsr_info_key = sorted_num_segments[segment_idx]
         ax[row_idx, col_idx].hist(tsr_info[tsr_info_key]['interval_durations'], bins=bins)
         ax[row_idx, col_idx].title.set_text('T=%s'%(str(tsr_info_key)))
         segment_idx += 1
   plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_path_regexp', dest='input_regexp', required=True, help='A path to the input files each containing a TSR approximation of a signal for different numbers of segments.  The path should contain a wildcard (*) so it matches all TSR approximation files for a single signal.  For example: "/mypath/<subject_id>_T*.csv"')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_regexp = args.input_regexp
   OptimizeTSR(input_csv_regexp)
