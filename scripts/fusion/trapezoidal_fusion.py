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
import sklearn.metrics
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# For debugging
show_final_plot = True
show_debug_plots = False

MAX_SHIFT_SECONDS = 5
CONST_VAL = 1
CHANGE_VAL = -1
CONST_THRESHOLD = 1e-4

# Maximize the alignment of the segment sequence to feature time series using NMI
def TimeOffsetSegmentSequence(segment_seq, features_df):
   time_index = segment_seq.index
   trap_seg_features = ComputeTrapezoidalSegmentSequence(features_df, time_index)
   best_shift = None
   best_nmi_total = -np.inf
   for shift in range(MAX_SHIFT_SECONDS+1):
      shifted_seg_seq = segment_seq.iloc[shift:]
      shifted_trap_seg_features = trap_seg_features.iloc[0:trap_seg_features.shape[0]-shift,:]
      nmi_total = 0
      for trap_seg_feat_idx in range(trap_seg_features.shape[1]):
         nmi_total += sklearn.metrics.normalized_mutual_info_score(shifted_trap_seg_features.iloc[:,trap_seg_feat_idx], shifted_seg_seq)
      if nmi_total > best_nmi_total:
         best_nmi_total = nmi_total
         best_shift = shift
   shifted_seg_seq = segment_seq.iloc[best_shift:]
   shifted_seg_seq.index = segment_seq.index[0:-best_shift]
   return shifted_seg_seq, best_shift

# Maximize the alignment of the segment sequences via temporal shift
def TimeAlignSegmentSeqs(segment_seqs, sample_rate, max_shift_seconds):
   max_time_shift = int(math.ceil(float(max_shift_seconds)*sample_rate))
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
   
   # HACK - TaskA green intensity experiment
   #shifts = np.zeros((1,num_annotators))

   print("Examining agreement over all possible temporal shifts")
   best_segment_seq_mat = None
   best_shift = None
   max_agreement = -np.inf
   min_shift_sum = np.inf
   for i in range(shifts.shape[0]):
      if (i%100000) == 0:
         print("%f%% complete"%(100*float(i)/shifts.shape[0]))
      shift = shifts[i,:]
      shifted_seqs_mat = np.zeros((num_annotators, num_samples))
      for annotator_idx in range(num_annotators):
         annotator_shift = shift[annotator_idx]
         shifted_segment_seq = segment_seqs[annotator_idx][annotator_shift:]
         shifted_seqs_mat[annotator_idx,0:len(shifted_segment_seq)] = shifted_segment_seq.values.reshape(-1,)
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

# Get the trapezoidal segment sequence of each annotation
def ComputeTrapezoidalSegmentSequence(signal_df, time_index):
   trap_segment_seq = pd.DataFrame(data=CHANGE_VAL*np.ones(len(time_index)), index=time_index)
   cur_time_index = 0
   next_time_index = 1
   for i in range(signal_df.shape[1]):
      signal = signal_df.iloc[:,i]
      while next_time_index < signal_df.shape[0]:
         if abs(signal.iloc[next_time_index]-signal.iloc[cur_time_index]) < CONST_THRESHOLD:
            while abs(signal.iloc[next_time_index]-signal.iloc[next_time_index-1]) < CONST_THRESHOLD:
               next_time_index += 1
               if next_time_index == signal_df.shape[0]:
                  break
            mask1 = trap_segment_seq.index >= signal.index[cur_time_index]
            mask2 = trap_segment_seq.index <= signal.index[next_time_index-1]
            trap_segment_seq.iloc[mask1 & mask2,i] = CONST_VAL
         else:
            while abs(signal.iloc[next_time_index]-signal.iloc[next_time_index-1]) >= CONST_THRESHOLD:
               next_time_index += 1
               if next_time_index == signal_df.shape[0]:
                  break
            mask1 = trap_segment_seq.index > signal.index[cur_time_index]
            mask2 = trap_segment_seq.index < signal.index[next_time_index-1]
            trap_segment_seq.iloc[mask1 & mask2,i] = CHANGE_VAL
         cur_time_index = next_time_index - 1
         next_time_index = cur_time_index + 1

   return trap_segment_seq

def ComputeTrapezoidalFusion(input_csv_path, output_csv_path, target_hz=1.0, do_time_alignment=False, ground_truth_path=None, tikz_file=None):
   # Load the ground truth
   gt_df = pd.read_csv(ground_truth_path)

   # Get the largest time index in any file
   trap_seg_files = glob.glob(os.path.join(input_csv_path, '*.csv'))
   max_time = 0.0
   for trap_seg_file in trap_seg_files:
      signal_df = pd.read_csv(trap_seg_file)
      max_time = max(max_time, signal_df.iloc[-1,0])

   # Get the trapezoidal segment sequence of each annotation
   trap_segment_seqs = []
   time_index = np.arange(0,max_time,1.0/target_hz)
   for trap_seg_file in trap_seg_files:
      signal_df = pd.read_csv(trap_seg_file)
      signal_df = signal_df.set_index(signal_df.columns[0])

      # Quantize the signal to remove meaningless tiny differences between values
      #quantized_signal= [decimal.Decimal(x).quantize(decimal.Decimal('0.00001'), rounding=decimal.ROUND_HALF_UP) for x in signal]
      #signal = pd.Series(np.array(quantized_signal).astype(float), index=signal.index)

      trap_segment_seq = ComputeTrapezoidalSegmentSequence(signal_df, time_index)

      if show_debug_plots:
         half_sample_interval = 0.5/target_hz
         fig, ax = plt.subplots()
         ax.plot(signal_df.index, signal_df.values, 'b-')
         plt.title("Segmented TSR: %s"%(os.path.basename(trap_seg_file)))
         const_type_mask = trap_segment_seq == CONST_VAL
         const_times = trap_segment_seq.index[const_type_mask]
         change_times = trap_segment_seq.index[~const_type_mask]
         for const_time in const_times:
            ax.axvspan(const_time-half_sample_interval, const_time+half_sample_interval, alpha=0.2, color='red')
         for change_time in change_times:
            ax.axvspan(change_time-half_sample_interval, change_time+half_sample_interval, alpha=0.2, color='green')
         plt.show()

      trap_segment_seqs.append(trap_segment_seq)

   # Align the trapezoidal segment seguences to each other
   if do_time_alignment:
      trap_segment_seqs_aligned, best_annotator_shifts = TimeAlignSegmentSeqs(trap_segment_seqs, target_hz, MAX_SHIFT_SECONDS)
      print("Best temporal shifts per annotator: "+str(best_annotator_shifts))
   else:
      trap_segment_seqs_aligned, best_annotator_shifts = TimeAlignSegmentSeqs(trap_segment_seqs, target_hz, 0)

   # Fuse the segment sequences via majority voting
   fused_trap_segment_seq = np.sum(trap_segment_seqs_aligned, axis=1)
   fused_trap_segment_seq[fused_trap_segment_seq >= 0] = CONST_VAL
   fused_trap_segment_seq[fused_trap_segment_seq < 0] = CHANGE_VAL

   # Offset the trapezoidal segment sequence in time to align with stimulus features
   if do_time_alignment:
      # TODO: compute features instead of using ground truth
      fused_trap_seg_seq_aligned, best_time_offset = TimeOffsetSegmentSequence(fused_trap_segment_seq, gt_df)
      print("Best time alignment offset: "+str(best_time_offset))
   else:
      fused_trap_seg_seq_aligned = fused_trap_segment_seq
   
   # Plot
   half_sample_interval = 0.5/target_hz
   fig, ax = plt.subplots()
   ax.plot(gt_df.iloc[:,0], gt_df.iloc[:,1], 'b-')
   plt.title("Fused Segmented Trapezoidal Sequence")
   const_type_mask = fused_trap_seg_seq_aligned == CONST_VAL
   const_times = fused_trap_seg_seq_aligned.index[const_type_mask]
   change_times = fused_trap_seg_seq_aligned.index[~const_type_mask]
   for const_time in const_times:
      ax.axvspan(const_time-half_sample_interval, const_time+half_sample_interval, alpha=0.2, color='red')
   for change_time in change_times:
      ax.axvspan(change_time-half_sample_interval, change_time+half_sample_interval, alpha=0.2, color='green')

   if show_final_plot:
      plt.show()

   if tikz_file is not None:
      matplotlib2tikz.save(tikz_file)

   # Save final TSR segment sequence
   out_df = pd.DataFrame({'Time': fused_trap_seg_seq_aligned.index, 'Value': fused_trap_seg_seq_aligned})
   out_df.to_csv(output_csv_path, header=True, index=False)

   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='Folder containing CSV-formatted TSR signals (first column: time, second: signal value)')
   parser.add_argument('--output', dest='output_csv', required=True, help='Output csv fused signal path')
   parser.add_argument('--target_hz', dest='target_hz', required=False, help='Frequency of the desired fusion')
   parser.add_argument('--do_time_alignment', dest='do_time_alignment', required=False, action="store_true", help='Flag indicating that trapezoidal segment sequence time alignment should be performed')
   parser.add_argument('--ground_truth', dest='ground_truth', required=True, help='CSV file containing the ground truth data')
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
   do_time_alignment = False
   if args.do_time_alignment:
      do_time_alignment = True
   ground_truth_path = args.ground_truth
   tikz_file = args.tikz
   output_csv_path = args.output_csv
   ComputeTrapezoidalFusion(input_csv_path, output_csv_path, target_hz=target_hz, do_time_alignment=do_time_alignment, ground_truth_path=ground_truth_path, tikz_file=tikz_file)
