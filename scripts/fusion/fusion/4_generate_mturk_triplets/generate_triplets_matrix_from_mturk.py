#!/usr/bin/env python

import os
import sys
import pdb
import numpy as np
from FileIO import GetCsvData, SaveCsvData

def GetIntervalIndexFromTimeRange(intervals_times, start_time, end_time):
   eps = 0.0000001
   for idx in range(intervals_times.shape[0]):
      if np.sum(np.abs(intervals_times[idx] - [start_time, end_time])) < eps:
         return idx

   print('ERROR: Could not find an interval with the specified time range. FIX ME!')
   pdb.set_trace()
   return np.nan

def GenerateTripletsMatrixFromMTurk(output_file_path, mturk_results_csv, intervals_csv, intervals_sampling_rate):
   (mturk_header, mturk_results) = GetCsvData(mturk_results_csv, first_line_header=True)
   (dummy_header, intervals_data) = GetCsvData(intervals_csv, first_line_header=False)

   mturk_header = mturk_header.tolist()
   ref_start_idx = mturk_header.index('Input.ref_start')
   ref_end_idx = mturk_header.index('Input.ref_end')
   a_start_idx = mturk_header.index('Input.a_start')
   a_end_idx = mturk_header.index('Input.a_end')
   b_start_idx = mturk_header.index('Input.b_start')
   b_end_idx = mturk_header.index('Input.b_end')
   result_idx = mturk_header.index('Answer.choice')

   intervals_times = intervals_data.astype(float)/intervals_sampling_rate

   triplets_mat = np.zeros((mturk_results.shape[0],3)).astype(int)
   for i in range(mturk_results.shape[0]):
      mturk_result = mturk_results[i]
      ref_start_time = float(mturk_result[ref_start_idx])
      ref_end_time =  float(mturk_result[ref_end_idx])
      a_start_time = float(mturk_result[a_start_idx])
      a_end_time =  float(mturk_result[a_end_idx])
      b_start_time = float(mturk_result[b_start_idx])
      b_end_time =  float(mturk_result[b_end_idx])
      result_str = mturk_result[result_idx]

      ref_interval_idx = GetIntervalIndexFromTimeRange(intervals_times, ref_start_time, ref_end_time)
      a_interval_idx = GetIntervalIndexFromTimeRange(intervals_times, a_start_time, a_end_time)
      b_interval_idx = GetIntervalIndexFromTimeRange(intervals_times, b_start_time, b_end_time)

      # Generate triplet (ref,a,b) if ||ref-a|| < ||ref-b||
      if result_str == 'optionA':
         triplets_mat[i] = [ref_interval_idx, a_interval_idx, b_interval_idx]
      elif result_str == 'optionB':
         triplets_mat[i] = [ref_interval_idx, b_interval_idx, a_interval_idx]
      else:
         print('Unknown choice option in MTurk results file. FIX ME!')
         pdb.set_trace()
         return

   print('Done!')
   SaveCsvData(output_file_path, None, triplets_mat)

if __name__ == '__main__':
   if len(sys.argv) > 4:
      output_file_path = sys.argv[1]
      mturk_results_csv = sys.argv[2]
      intervals_csv_path = sys.argv[3]
      intervals_sampling_rate = float(sys.argv[4])
      GenerateTripletsMatrixFromMTurk(output_file_path, mturk_results_csv, intervals_csv_path, intervals_sampling_rate)
   else:
      print 'Please provide the following command line arguments:\n1) Output triplets file path\n2) Path to Mechanical Turk results csv file\n3) Path to constant intervals csv used in MTurk experiment\n4) The sampling rate of the constant intervals file'
