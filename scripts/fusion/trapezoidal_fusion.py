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
import numpy.matlib
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# For debugging
show_final_plot = True
show_debug_plots = False

# Maximum the alignment of the segment sequences via temporal shift
def TimeAlignSegmentSeqs(segment_seqs, sample_rate):
   max_time_shift = int(math.ceil(3.0*sample_rate)) # 0-5 seconds
   segment_seqs_mat = np.array(segment_seqs)
   num_annotators = len(segment_seqs)
   num_samples = len(segment_seqs[0])

   print("Building matrix of all possible time shifts per annotator")
   # Build matrix of all possible time shifts
   shifts = np.zeros(((max_time_shift+1)**num_annotators,num_annotators))
   for i in range(num_annotators):
      rep_shift_mat = []
      for shift_amount in range(max_time_shift+1):
         rep_shift_mat.extend(((max_time_shift+1)**i)*[shift_amount])
      num_repeats = shifts.shape[0]/len(rep_shift_mat)
      shifts[:,-1-i] = np.matlib.repmat(rep_shift_mat, 1, num_repeats).T.reshape(-1,)

   print("Examining agreement over all possible temporal shifts")
   best_segment_seq_mat = None
   best_shift = None
   max_agreement = -np.inf
   min_shift_sum = np.inf
   for i in range(shifts.shape[0]):
      if (i%100000) == 0:
         print("%f%% complete"%(100*float(i)/shifts.shape[0]))
      shift = shifts[i,:]
      shifted_seqs_mat = np.zeros_like(segment_seqs_mat)
      for annotator_idx in range(num_annotators):
         annotator_shift = shift[annotator_idx]
         shifted_segment_seq = segment_seqs[annotator_idx][annotator_shift:]
         shifted_seqs_mat[annotator_idx,0:len(shifted_segment_seq)] = shifted_segment_seq
      agreement = np.sum(np.abs(np.sum(shifted_seqs_mat, axis=0)))
      if agreement >= max_agreement:
         max_agreement = agreement
         if np.sum(shift) < min_shift_sum:
            best_segment_seq_mat = shifted_seqs_mat
            best_shift = shift
            min_shift_sum = np.sum(shift)
      
   col_names = []
   for i in range(num_annotators):
      col_names.append('s'+str(i))
   aligned_segment_seq = pd.DataFrame(data=best_segment_seq_mat.T, index=segment_seqs[0].index, columns=col_names)
   return aligned_segment_seq, best_shift

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

   tsr_segment_seqs_aligned, best_shift = TimeAlignSegmentSeqs(tsr_segment_seqs, target_hz)
   print("Best temporal shift: "+str(best_shift))
   
   # Compute final segment sequence via majority voting
   final_tsr_segment_seq = np.sum(tsr_segment_seqs_aligned, axis=1)
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
