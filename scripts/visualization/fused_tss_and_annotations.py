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

# TSS encoding constant variables
CONST_VAL = 0

def VisualizeFusedTSSAndAnnotations(annotations_folder, fused_tss_file_path, file_name_filter_str='', tikz_file_out_path=None):
   # Load the annotations
   annotations = {}
   anno_files = glob.glob(os.path.join(annotations_folder, '*.csv'))
   for anno_file in anno_files:
      if file_name_filter_str == '' or file_name_filter_str in os.path.basename(anno_file):
          df = pd.read_csv(anno_file)
          annotations[os.path.basename(anno_file)] = df

   # Load the fused TSS
   fused_tss = pd.read_csv(fused_tss_file_path)
   const_type_mask = fused_tss.iloc[:,1] == CONST_VAL
   const_times = fused_tss.index[const_type_mask]
   change_times = fused_tss.index[~const_type_mask]
   
   # Plot
   fig, ax = plt.subplots()
   for anno_key in annotations.keys():
      df = annotations[anno_key]
      ax.plot(df.iloc[:,0], df.iloc[:,1])

   half_sample_interval = 0.5*(fused_tss.iloc[1,0]-fused_tss.iloc[0,0])
   for const_time in const_times:
      ax.axvspan(const_time-half_sample_interval, const_time+half_sample_interval, alpha=0.2, color='red')
   for change_time in change_times:
      ax.axvspan(change_time-half_sample_interval, change_time+half_sample_interval, alpha=0.2, color='green')
   plt.xlim(135,167)

   if tikz_file_out_path is not None:
      matplotlib2tikz.save(tikz_file_out_path)

   plt.show()

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--annotations_path', dest='annotations_path', required=True, help='Path to folder containing annotations')
   parser.add_argument('--fused_tss_path', dest='fused_tss_path', required=True, help='Path to file containing the fused TSS for the annotations')
   parser.add_argument('--file_name_filter_str', required=False, type=str, default='', help='A substring used to filter the view of files living in the annotations_path.  Only files containing this substring will be used during execution')
   parser.add_argument('--tikz', dest='tikz', required=False, help='Output path for TikZ PGF plot code')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   VisualizeFusedTSSAndAnnotations(args.annotations_path, args.fused_tss_path, args.file_name_filter_str, args.tikz)
