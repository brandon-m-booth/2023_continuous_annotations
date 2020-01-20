#!/usr/bin/env python
#Author: Brandon M. Booth

import os
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd

CONST_THRESHOLD = 1e-4

def ExtractConstIntervalsSignal(tsr_df):
   # Save a list of constant interval segment pairs:
   # (start time, end time)
   constant_intervals = []
   const_start_idx = None
   for idx in range(tsr_df.shape[0]):
      if const_start_idx is None:
         const_start_idx = idx
      else:
         if abs(tsr_df.iloc[idx,0]-tsr_df.iloc[const_start_idx,0]) < CONST_THRESHOLD:
            start_time = tsr_df.index[const_start_idx]
            end_time = tsr_df.index[idx]
            constant_intervals.append((start_time, end_time))
            const_start_idx = None
         else:
            const_start_idx = idx
   return constant_intervals

def ExtractConstIntervals(input_csv_path, output_path):
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   tsr_files = glob.glob(os.path.join(input_csv_path, '*.csv'))

   # Extract the constant intervals from each TSR file
   for tsr_file in tsr_files:
      tsr_df = pd.read_csv(tsr_file)
      tsr_df = tsr_df.set_index(tsr_df.columns[0])

      constant_intervals = ExtractConstIntervalsSignal(tsr_df)

      out_file_path = os.path.join(output_path, os.path.basename(tsr_file)[:-4]+'_const_intervals.csv')
      out_df = pd.DataFrame(data=np.array(constant_intervals), columns=['Start Time', 'End Time'])
      out_df.to_csv(out_file_path, header=True, index=False)
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='Folder containing CSV-formatted TSR signals (first column: time, second: signal value)')
   parser.add_argument('--output_path', dest='output_path', required=True, help='Output path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   output_path = args.output_path
   ExtractConstIntervals(input_csv_path, output_path)
