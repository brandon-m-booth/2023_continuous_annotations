#!/usr/bin/env python
#Author: Brandon M. Booth

import os
import sys
import pdb
import copy
import math
import glob
import decimal
import argparse
import statprof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib2tikz
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# For debugging
show_final_plot = True
show_debug_plots = False

def ComputeTrapezoidalFusion(input_csv_path, output_csv_path, target_hz=1.0, ground_truth_path=None, tikz_file=None):
   CONST_VAL = 1
   CHANGE_VAL = -1
   const_threshold = 1e-4

   # Get the largest time index in any file
   tsr_files = glob.glob(os.path.join(input_csv_path, '*.csv'))
   max_time = 0.0
   for tsr_file in tsr_files:
      signal_df = pd.read_csv(tsr_file)
      time = signal_df.iloc[:,0]
      max_time = max(max_time, time.iloc[-1])

   # Get the TSR segment sequence of each annotation
   tsr_segment_seqs = []
   time_index = np.arange(0,max_time,1.0/target_hz)
   tsr_template = pd.Series(data=len(time_index)*[np.nan], index=time_index)
   for tsr_file in tsr_files:
      signal_df = pd.read_csv(tsr_file)
      time = signal_df.iloc[:,0]
      signal = signal_df.iloc[:,1]
      signal.index = time

      # Quantize the signal to remove meaningless tiny differences between values
      #quantized_signal= [decimal.Decimal(x).quantize(decimal.Decimal('0.00001'), rounding=decimal.ROUND_HALF_UP) for x in signal]
      #signal = pd.Series(np.array(quantized_signal).astype(float), index=signal.index)

      # Compute the TSR segment sequence
      tsr_segment_seq = copy.deepcopy(tsr_template)
      tsr_segment_seq.iloc[:] = CHANGE_VAL
      cur_time_index = 0
      next_time_index = 1
      while next_time_index < len(time):
         if abs(signal.iloc[next_time_index]-signal.iloc[cur_time_index]) < const_threshold:
            while abs(signal.iloc[next_time_index]-signal.iloc[next_time_index-1]) < const_threshold:
               next_time_index += 1
               if next_time_index == len(time):
                  break
            mask1 = tsr_segment_seq.index >= signal.index[cur_time_index]
            mask2 = tsr_segment_seq.index <= signal.index[next_time_index-1]
            tsr_segment_seq.iloc[mask1 & mask2] = CONST_VAL
         else:
            while abs(signal.iloc[next_time_index]-signal.iloc[next_time_index-1]) >= const_threshold:
               next_time_index += 1
               if next_time_index == len(time):
                  break
            mask1 = tsr_segment_seq.index > signal.index[cur_time_index]
            mask2 = tsr_segment_seq.index < signal.index[next_time_index-1]
            tsr_segment_seq.iloc[mask1 & mask2] = CHANGE_VAL
         cur_time_index = next_time_index - 1
         next_time_index = cur_time_index + 1

      if show_debug_plots:
         half_sample_interval = 0.5/target_hz
         fig, ax = plt.subplots()
         ax.plot(time, signal, 'b-')
         plt.title("Segmented TSR: %s"%(os.path.basename(tsr_file)))
         const_type_mask = tsr_segment_seq == CONST_VAL
         const_times = tsr_segment_seq.index[const_type_mask]
         change_times = tsr_segment_seq.index[~const_type_mask]
         for const_time in const_times:
            ax.axvspan(const_time-half_sample_interval, const_time+half_sample_interval, alpha=0.2, color='red')
         for change_time in change_times:
            ax.axvspan(change_time-half_sample_interval, change_time+half_sample_interval, alpha=0.2, color='green')
         plt.show()

      tsr_segment_seqs.append(tsr_segment_seq)
   
   # Compute final segment sequence via majority voting
   # TODO: time alignment
   final_tsr_segment_seq = tsr_segment_seqs[0]
   for i in range(1,len(tsr_segment_seqs)):
      final_tsr_segment_seq += tsr_segment_seqs[i]
   final_tsr_segment_seq[final_tsr_segment_seq >= 0] = CONST_VAL
   final_tsr_segment_seq[final_tsr_segment_seq < 0] = CHANGE_VAL

   # Plot
   half_sample_interval = 0.5/target_hz
   fig, ax = plt.subplots()
   if ground_truth_path:
      gt_df = pd.read_csv(ground_truth_path)
      ax.plot(gt_df.iloc[:,0], gt_df.iloc[:,1], 'b-')
   else:
      ax.plot(time, signal, 'b-') # TODO: Plot the average?
   plt.title("Final Segmented TSR")
   const_type_mask = final_tsr_segment_seq == CONST_VAL
   const_times = final_tsr_segment_seq.index[const_type_mask]
   change_times = final_tsr_segment_seq.index[~const_type_mask]
   for const_time in const_times:
      ax.axvspan(const_time-half_sample_interval, const_time+half_sample_interval, alpha=0.2, color='red')
   for change_time in change_times:
      ax.axvspan(change_time-half_sample_interval, change_time+half_sample_interval, alpha=0.2, color='green')

   if show_final_plot:
      plt.show()

   if tikz_file is not None:
      matplotlib2tikz.save(tikz_file)

   # Save final TSR segment sequence
   out_df = pd.DataFrame({'Time': final_tsr_segment_seq.index, 'Value': final_tsr_segment_seq})
   out_df.to_csv(output_csv_path, header=True, index=False)

   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='Folder containing CSV-formatted TSR signals (first column: time, second: signal value)')
   parser.add_argument('--output', dest='output_csv', required=True, help='Output csv fused signal path')
   parser.add_argument('--target_hz', dest='target_hz', required=False, help='Frequency of the desired fusion')
   parser.add_argument('--ground_truth', dest='ground_truth', required=False, help='CSV file containing the ground truth data')
   parser.add_argument('--tikz', dest='tikz', required=False, help='Output path for TikZ PGF plot code') 
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   target_hz = 1.0
   if args.target_hz:
      target_hz = float(args.target_hz)
   ground_truth_path = None
   if args.ground_truth:
      ground_truth_path = args.ground_truth
   tikz_file = args.tikz
   output_csv_path = args.output_csv
   ComputeTrapezoidalFusion(input_csv_path, output_csv_path, target_hz=target_hz, ground_truth_path=ground_truth_path, tikz_file=tikz_file)
