#!/usr/bin/env python
import os
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FileIO import GetCsvData, SaveCsvData
from PrettyPlotter import pretty
from matplotlib2tikz import save as tikz_save

def RemoveIntervalOverlap(intervals):
   last_end_idx = -1
   for i in range(intervals.shape[0]):
      if intervals[i,0] <= last_end_idx:
         intervals[i,0] = last_end_idx+1
      last_end_idx = intervals[i,1]
   return intervals


def GetWildcardMatch(wildcard_str, inflated_str):
   if '*' in wildcard_str:
      match_str = inflated_str
      idx = wildcard_str.index('*')
      match_str = match_str.replace(wildcard_str[idx+1:], '')
      match_str = match_str.replace(wildcard_str[0:idx], '')
      return match_str
   else:
      return wildcard_str


def DoConstructSignal(seg_seq_csv, intervals_csv, interval_values_glob, output_file, objective_csv=None, tikz_path=None, do_show_plot=True):
   if not os.path.isdir(os.path.split(output_file)[0]):
      os.makedirs(os.path.split(output_file)[0])

   interval_values_globs = glob.glob(interval_values_glob)
   for interval_values_csv in interval_values_globs:
      seg_seq_header, seg_seq = GetCsvData(seg_seq_csv)
      dummy_header, intervals = GetCsvData(intervals_csv, first_line_header=False)
      dummy_header, interval_values = GetCsvData(interval_values_csv, first_line_header=False)
      intervals = intervals.astype(int)

      # Separate times from segment sequence
      times = seg_seq[:,0]
      segments = seg_seq[:,1]
      sampling_rate = 1.0/(times[1]-times[0])

      constructed_signal = pd.DataFrame(data={'Time_sec': times, 'Data': len(segments)*[np.nan]})
      constructed_signal = constructed_signal.reindex(columns=['Time_sec', 'Data'])
      
      # Fill in the constant interval values
      for interval_idx in range(intervals.shape[0]):
         interval = intervals[interval_idx,:]
         interval_value = interval_values[interval_idx]
         constructed_signal.iloc[interval[0]:interval[1]+1,1] = interval_value

      # Linearly interpolate between constant intervals
      constructed_signal.interpolate(method='linear', inplace=True)

      # Plot the results
      if do_show_plot:
         if objective_csv is not None:
            ot_header, ot_signal = GetCsvData(objective_csv)
            plt.plot(ot_signal[:,0], ot_signal[:,1], 'm-', linewidth=4)

         plt.xlabel('Time(s)', fontsize=24)
         plt.ylabel('Green Intensity', fontsize=24)

         dummy_header, intervals = GetCsvData(intervals_csv, first_line_header=False)
         for i in range(intervals.shape[0]):
            interval = intervals[i]/sampling_rate
            values = 2*[interval_values[i]]
            if i > 0:
               plt.plot(interval, values, 'g-o', label='_nolegend_')
            else:
               plt.plot(interval, values, 'g-o')

         plt.plot(constructed_signal.iloc[:,0], constructed_signal.iloc[:,1], 'r-')
         
         pretty(plt)

         plt.axis([times[0],times[-1],-1.5,1.5])
         legend_list = []
         if objective_csv is not None:
            legend_list.append('Objective Truth')
         legend_list.extend(['Embedded Intervals', 'Constructed Signal'])
         plt.legend(legend_list, loc='upper left', bbox_to_anchor=(1,1), frameon=False, prop={'size':24})
         if tikz_path is not None:
            tikz_save(tikz_path)
         plt.show()

      if '*' in output_file:
         wildcard_match = GetWildcardMatch(interval_values_glob, interval_values_csv)
         outfile = output_file.replace('*', wildcard_match)
      else:
         outfile = output_file

      constructed_signal.to_csv(outfile, index=False, header=True)

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--seg_seq_path', dest='seg_seq', required=True, help='Path to CSV containing the fused segment sequence')
   parser.add_argument('--intervals_path', dest='intervals_path', required=True, help='Path to CSV containing the constant interval indices')
   parser.add_argument('--interval_embedding_path', dest='interval_embedding_path', required=True, help='Path to CSV containing the constant interval embedding values')
   parser.add_argument('--output_file_path', dest='output_file_path', required=True, help='Output CSV file path for the constructed signal')
   parser.add_argument('--ground_truth_path', dest='ground_truth_path', required=False, help='CSV file containing the ground truth data (used only for plotting)')
   parser.add_argument('--tikz', dest='tikz', required=False, help='Output path for TikZ PGF plot code') 
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ground_truth_path = None if not args.ground_truth_path else args.ground_truth_path
   tikz_path = None if not args.tikz else args.tikz
   DoConstructSignal(args.seg_seq, args.intervals_path, args.interval_embedding_path, args.output_file_path, ground_truth_path, tikz_path)
