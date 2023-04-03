import re
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib

def GetAnnotationDataFrame(aligned_anno_movie_files, movie_cut_times_df, lag_correction_secs=0.0):
   cut_anno_dict = {}
   for aligned_anno_movie_file in aligned_anno_movie_files:
      cut_num = int(os.path.basename(aligned_anno_movie_file).split('_cut')[1].split('_')[0])
      if not cut_num in cut_anno_dict.keys():
         cut_anno_dict[cut_num] = []
      anno_df = pd.read_csv(aligned_anno_movie_file)
      cut_anno_dict[cut_num].append(anno_df)

   merged_df = None
   for cut_num in sorted(cut_anno_dict.keys()):
      anno_dfs = cut_anno_dict[cut_num]
      time_df = anno_dfs[0].iloc[:,0]
      cut_anno_df = pd.concat([time_df]+[x.iloc[:,1] for x in anno_dfs], axis=1)

      # Find the corresponding movie cut times entry
      cut_name = 'cut'+str(cut_num)
      cut_time_df = movie_cut_times_df.loc[movie_cut_times_df['Cut Name']==cut_name, :]
      start_time_sec = cut_time_df['Start Time Sec'].iloc[0]
      end_time_sec = cut_time_df['End Time Sec'].iloc[0]
      cut_anno_df.iloc[:,0] += start_time_sec - lag_correction_secs

      if merged_df is None:
         merged_df = cut_anno_df
      else:
         merged_df = merged_df.merge(cut_anno_df, on=merged_df.columns[0], how='outer', sort=False)

   merged_df.sort_values(by=merged_df.columns[0], inplace=True)
   return merged_df

def CompareAnnotations(warped_annos_path, baseline_annos_path, lag_correction_secs=0.0, tikz_output_path=None):
   if tikz_output_path is not None and not os.path.isdir(tikz_output_path):
      os.makedirs(tikz_output_path)

   warped_anno_files = glob.glob(os.path.join(warped_annos_path, '*.csv'))
   baseline_anno_files = glob.glob(os.path.join(baseline_annos_path, '*.csv'))

   # Min-max normalize all warped annotations between 0 and 1
   warped_annos_dict = {}
   min_val = float("inf")
   max_val = -float("inf")
   for warped_anno_file in warped_anno_files:
      merged_warped_df = pd.read_csv(warped_anno_file)
      movie_name = os.path.basename(warped_anno_file).split('.')[0]
      movie_name = movie_name[len('warped_annotation_'):]
      min_val = min(np.min(merged_warped_df.iloc[:,1]),min_val)
      max_val = max(np.max(merged_warped_df.iloc[:,1]),max_val)
      warped_annos_dict[movie_name] = merged_warped_df
   for movie_name in warped_annos_dict.keys():
      df = warped_annos_dict[movie_name]
      df.iloc[:,1] = (df.iloc[:,1] - min_val)/(max_val-min_val)
      warped_annos_dict[movie_name] = df

   # Plot the baseline and warped annotations
   for movie_name in warped_annos_dict.keys():
      merged_warped_df = warped_annos_dict[movie_name]

      baseline_anno_file = [x for x in baseline_anno_files if movie_name in os.path.basename(x)][0]
      baseline_df = pd.read_csv(baseline_anno_file)
      baseline_df.iloc[:,1] += lag_correction_secs # lag correction
      baseline_df.iloc[:,1] = (100+baseline_df.iloc[:,1])/200.0 # 0-1 normalization

      plt.figure()
      plt.plot(baseline_df.iloc[:,0], baseline_df.iloc[:,1], 'r--')
      plt.plot(merged_warped_df.iloc[:,0], merged_warped_df.iloc[:,1], 'b-')
      plt.legend(['Baseline', 'Signal Warped'])
      plt.title(movie_name)

      if tikz_output_path is not None:
         tikz_output_file = os.path.join(tikz_output_path, movie_name+'_anno_compare.tex')
         tikzplotlib.save(tikz_output_file)
   plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--merged_warped_annotations_path', required=True, help='Folder containing merged warped annotations, one file per movie')
   parser.add_argument('--baseline_annotations_path', required=True, help='Folder containing baseline annotations, one file per movie')
   parser.add_argument('--lag_correction_secs', required=False, default=0.0, type=float, help='Number of seconds to shift the baseline annotations to correct for annotator lag.  Typical values range from 1-5 seconds')
   parser.add_argument('--tikz_output_path', required=False, help='Output folder for tikz plots')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CompareAnnotations(args.merged_warped_annotations_path, args.baseline_annotations_path, args.lag_correction_secs, args.tikz_output_path)
