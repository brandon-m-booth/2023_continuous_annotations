#!/usr/bin/env python3
# Author: Brandon M Booth

import os
import re
import sys
import pdb
import glob
import math
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, '4_extract_const_intervals')))
from extract_const_intervals import ExtractConstIntervalsSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'util')))
from util import GetTSRSumSquareError, NormedDiff
import agreement_metrics as agree

def OptimizeTSR(signal_path, tsr_path, output_path, subj_id_signal_regex='(\d+)_average\.csv', subj_id_tsr_regex='(\d+)_tsr_T.*\.csv', seg_tsr_regex='.*_tsr_T(\d+)\.csv', show_plots=False):
   signal_files = glob.glob(os.path.join(signal_path, '*.csv'))
   if len(signal_files) == 0:
      print("No signal files found for path: %s.  Please check the path and try again"%(signal_path))
      return

   tsr_files = glob.glob(os.path.join(tsr_path, '*.csv'))
   if len(tsr_files) == 0:
      print("No TSR files found for path: %s.  Please check the path and try again"%(tsr_path))
      return

   if not os.path.exists(output_path):
      os.makedirs(output_path)

   subj_signal_files = {}
   # Store the signal files by subject id
   for signal_file in signal_files:
      re_match = re.search(subj_id_signal_regex, os.path.basename(signal_file))
      if re_match is None:
         continue
      subject_id = re_match.group(1)
      if subject_id not in subj_signal_files.keys():
         subj_signal_files[subject_id] = signal_file
      else:
         print("Error! Multiple signal files detected for subject %s"%(subject_id))
         pdb.set_trace()

   # Group the TSR files per subject
   subj_tsr_files = {}
   for tsr_file in tsr_files:
      subject_id = re.search(subj_id_tsr_regex, os.path.basename(tsr_file)).group(1)
      if subject_id not in subj_tsr_files.keys():
         subj_tsr_files[subject_id] = []
      subj_tsr_files[subject_id].append(tsr_file)

   # Compute and store information about each TSR file per subject
   subj_tsr_error_dict = {}
   for subj_id in tqdm(subj_tsr_files.keys(), desc='Gathering TSR info'):
      if not subj_id in subj_signal_files.keys():
         print("Skipping subject %s because no source signal file was found."%(subj_id))
         continue

      #print("Extracting TSR info for subject: %s"%(subj_id))
      signal_df = pd.read_csv(subj_signal_files[subj_id])
      time_cols = [x for x in signal_df.columns if 'time' in x.lower()]
      if len(time_cols) != 1:
         print("ERROR: Could not find time column")
         return
      signal_df = signal_df.set_index(time_cols[0])
      subj_tsr_info = {}
      for tsr_file in subj_tsr_files[subj_id]:
         tsr_df = pd.read_csv(tsr_file)
         tsr_df = tsr_df.set_index(tsr_df.columns[0])
         subject_id = re.search(subj_id_tsr_regex, tsr_file).group(1)

         num_tsr_segments = int(re.search(seg_tsr_regex, tsr_file).group(1))
         constant_intervals = ExtractConstIntervalsSignal(tsr_df)
         num_const_intervals = len(constant_intervals)#tsr_df.shape[0]-1
         sse = GetTSRSumSquareError(tsr_df, signal_df)

         tsr_interp_df = pd.merge(tsr_df, signal_df, how='outer', left_index=True, right_index=True)
         tsr_interp_df.interpolate(method='index', inplace=True)
         pearson_corr = agree.PearsonCorr(tsr_interp_df).iloc[0,1]
         kendalls_tau = agree.KendallTauCorr(tsr_interp_df).iloc[0,1]
         tsr_norm_diff_interp_df = NormedDiff(tsr_interp_df)
         sda = agree.SDACorrected(tsr_norm_diff_interp_df).iloc[0,1]

         subj_tsr_info[num_tsr_segments] = {'num_const_intervals': num_const_intervals, 'sse':  sse, 'pearson': pearson_corr, 'kendalls_tau': kendalls_tau, 'sda': sda, 'signal_df': signal_df, 'tsr_df': tsr_df, 'tsr_files': tsr_file}

      subj_tsr_error_dict[subj_id] = {'num_tsr_segments':[], 'num_const_intervals':[], 'sse':[], 'pearson':[], 'kendalls_tau':[], 'sda':[], 'signal_dfs':[], 'tsr_dfs':[], 'tsr_files':[]}
      for num_tsr_segs in sorted(subj_tsr_info.keys()):
         subj_tsr_error_dict[subj_id]['num_tsr_segments'].append(num_tsr_segs)
         subj_tsr_error_dict[subj_id]['num_const_intervals'].append(subj_tsr_info[num_tsr_segs]['num_const_intervals'])
         subj_tsr_error_dict[subj_id]['sse'].append(subj_tsr_info[num_tsr_segs]['sse'])
         subj_tsr_error_dict[subj_id]['pearson'].append(subj_tsr_info[num_tsr_segs]['pearson'])
         subj_tsr_error_dict[subj_id]['kendalls_tau'].append(subj_tsr_info[num_tsr_segs]['kendalls_tau'])
         subj_tsr_error_dict[subj_id]['sda'].append(subj_tsr_info[num_tsr_segs]['sda'])
         subj_tsr_error_dict[subj_id]['signal_dfs'].append(subj_tsr_info[num_tsr_segs]['signal_df'])
         subj_tsr_error_dict[subj_id]['tsr_dfs'].append(subj_tsr_info[num_tsr_segs]['tsr_df'])
         subj_tsr_error_dict[subj_id]['tsr_files'].append(subj_tsr_info[num_tsr_segs]['tsr_files'])

   if show_plots:
      plt.ion()

   for subj_id in subj_tsr_error_dict.keys():
      num_segments = subj_tsr_error_dict[subj_id]['num_tsr_segments']
      sda = subj_tsr_error_dict[subj_id]['sda']
      kendalls_tau = subj_tsr_error_dict[subj_id]['kendalls_tau']
      sda_diff_idx = np.argmax(np.diff(sda))
      if max(np.diff(sda)) < 0:
         sda_diff_idx = 1
      else:
         sda_diff_idx += 1

      kendalls_tau_diff_idx = np.argmax(np.diff(kendalls_tau[sda_diff_idx-1:]))
      if max(np.diff(kendalls_tau[sda_diff_idx-1:])) < 0:
         optimal_idx = sda_diff_idx
      else:
         optimal_idx = sda_diff_idx + kendalls_tau_diff_idx

      # Output optimal TSR set, one file per subject, using a heuristic
      opt_subj_tsr_file = subj_tsr_error_dict[subj_id]['tsr_files'][optimal_idx]
      shutil.copy(opt_subj_tsr_file, output_path)

      if show_plots:
         fig, ax = plt.subplots(1,3)
         for i in range(len(subj_tsr_error_dict[subj_id]['num_tsr_segments'])):
            tsr_df = subj_tsr_error_dict[subj_id]['tsr_dfs'][i]
            T = subj_tsr_error_dict[subj_id]['num_tsr_segments'][i]
            ax[0].plot(tsr_df.index, tsr_df.values, label=str(T))
         signal_df = subj_tsr_error_dict[subj_id]['signal_dfs'][0]
         ax[0].plot(signal_df.index, signal_df.values, label='Anno')
         ax[0].legend()
         ax[0].set_title(subj_id + ' TSR Plots')

         ax[1].plot(subj_tsr_error_dict[subj_id]['num_tsr_segments'], subj_tsr_error_dict[subj_id]['num_const_intervals'], label='Num Const Intervals')
         ax[1].plot(subj_tsr_error_dict[subj_id]['num_tsr_segments'], subj_tsr_error_dict[subj_id]['sse'], label='SSE')
         ax[1].set_title(subj_id + ' Info Plots')
         ax[1].legend()

         ax[2].plot(subj_tsr_error_dict[subj_id]['num_tsr_segments'], subj_tsr_error_dict[subj_id]['pearson'], label='Pearson')
         ax[2].plot(subj_tsr_error_dict[subj_id]['num_tsr_segments'], subj_tsr_error_dict[subj_id]['kendalls_tau'], label='Kendall\'s Tau')
         ax[2].plot(subj_tsr_error_dict[subj_id]['num_tsr_segments'], subj_tsr_error_dict[subj_id]['sda'], label='SDA')
         ax[2].plot(2*[subj_tsr_error_dict[subj_id]['num_tsr_segments'][optimal_idx]], [0,1], color='black', label='')
         ax[2].set_title(subj_id + ' Agreement Plots')
         ax[2].legend()

         tikz_output_file = os.path.join(output_path, subj_id+'_opt_tsr_plots.tex')
         tikzplotlib.save(tikz_output_file)

   if show_plots:
      plt.ioff()
      plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_signal_path', dest='input_signal_path', required=True, help='Path to the folder containing the fused signals for each task')
   parser.add_argument('--input_tsr_path', dest='input_tsr_path', required=True, help='Path to the folder containing many TSR approximations to the signal for each task')
   parser.add_argument('--output_path', dest='output_path', required=True, help='Output folder for the optimal TSR signals (one per subject)')
   parser.add_argument('--subj_id_regex', required=True, help='Regex to use on the input signal path\'s file names, including parenthesis denoting the group containing the subject ID')
   parser.add_argument('--subj_id_tsr_regex', required=True, help='Regex to use on the input tsr path\s file names, including parenthesis denoting the group containing the subject ID')
   parser.add_argument('--segment_num_regex', required=True, help='Regex to use on the input tsr path\s file names, including parenthesis denoting the group containing the number of TSR segments')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Displays plots of each subject\'s TSR approximations, reconstruction error, and agreement')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   OptimizeTSR(args.input_signal_path, args.input_tsr_path, args.output_path, args.subj_id_regex, args.subj_id_tsr_regex, args.segment_num_regex, args.show_plots)
