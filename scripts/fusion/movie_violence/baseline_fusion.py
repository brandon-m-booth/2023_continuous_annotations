# Computes the average annotation as a baseline from cleaned, aligned annotations
# Author: Brandon M Booth
import os
import re
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def GetClipTimeDataFrame(clip_times_csv_path):
   df = pd.read_csv(clip_times_csv_path)
   df['movie_name'] = [os.path.basename(x)[:-4] for x in df['Movie Path']]
   df['clip_num'] = [int(x[3:]) for x in df['Cut Name']] # "Cut10" -> 10
   df['start_time'] = df['Start Time Sec']
   return df

def ComputeBaselineAnnotations(input_annos_path, clip_times_csv_path, output_annos_path, movie_name_regex, clip_number_regex, show_plots=False):
   if not os.path.isdir(output_annos_path):
      os.makedirs(output_annos_path)

   clip_start_time_df = GetClipTimeDataFrame(clip_times_csv_path)

   anno_files = glob.glob(os.path.join(input_annos_path, '*.csv'))
   if len(anno_files) == 0:
      print("No annotation files found for path: %s.  Please check the path and try again"%(input_annos_path))
      return

   movie_clip_dict = {}
   for anno_file in anno_files:
      re_movie_name = re.search(movie_name_regex, os.path.basename(anno_file))
      re_clip_num = re.search(clip_number_regex, os.path.basename(anno_file))
      if re_movie_name is None:
         print("Error: Could not match movie name for %s"%(anno_file))
         pdb.set_trace()
         continue
      if re_clip_num is None:
         print("Error: Could not match clip number for %s"%(anno_file))
         pdb.set_trace()
         continue
      movie_name = re_movie_name.group(1)
      clip_num = re_clip_num.group(1)
      if movie_name not in movie_clip_dict.keys():
         movie_clip_dict[movie_name] = {}
      if clip_num not in movie_clip_dict[movie_name].keys():
         movie_clip_dict[movie_name][clip_num] = []
      movie_clip_dict[movie_name][clip_num].append(anno_file)

   for movie_name in movie_clip_dict.keys():
      clip_nums = np.array(sorted([int(x) for x in movie_clip_dict[movie_name].keys()])).astype(str)

      # Compute the baseline per clip and assemble
      combined_movie_df = None
      for clip_num in clip_nums:
         clip_anno_files = movie_clip_dict[movie_name][clip_num]
         combined_clip_df = None
         for clip_anno_file in clip_anno_files:
            anno_clip_df = pd.read_csv(clip_anno_file)
            if combined_clip_df is None:
               combined_clip_df = anno_clip_df
            else:
               combined_clip_df = pd.concat((combined_clip_df, anno_clip_df.iloc[:,1]), axis=1)

         # Average the annotations for this clip
         combined_clip_df['average'] = combined_clip_df.iloc[:,1:].mean(axis=1, skipna=True)

         # Offset the clip timestamps by the start time
         row_mask = (clip_start_time_df['movie_name'] == movie_name) & (clip_start_time_df['clip_num'] == int(clip_num))
         if np.sum(row_mask) != 1:
            print("Failed to find just one match in the clip start times dataframe")
            pdb.set_trace()
         start_time = clip_start_time_df.loc[row_mask,'start_time'].values[0]
         combined_clip_df.iloc[:,0] += start_time
         
         ts_col = combined_clip_df.columns[0]
         if combined_movie_df is None:
            combined_movie_df = combined_clip_df.loc[:,[ts_col, 'average']]
         else:
            combined_movie_df = pd.concat((combined_movie_df, combined_clip_df.loc[:,[ts_col, 'average']]), axis=0)

      output_file_path = os.path.join(output_annos_path, movie_name+'_fused_baseline.csv')
      combined_movie_df.to_csv(output_file_path, index=False, header=True)

      if show_plots:
         plt.figure()
         plt.plot(combined_movie_df.iloc[:,0], combined_movie_df.iloc[:,1])
         plt.xlabel('Time(s)')
         plt.ylabel('Annotated Value')
         plt.title(movie_name)
         plt.show()

   return

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_clip_annotations_path', required=True, help='Path to the folder containing cleaned and filtered csv-formatted movie clip annotations')
   parser.add_argument('--clip_times_csv_path', required=True, help='Path to CSV file containing the start and stop times for each movie clip')
   parser.add_argument('--movie_name_regex', required=True, help='Regex with parenthesis to pull the movie name from each annotation file name')
   parser.add_argument('--clip_number_regex', required=True, help='Regex with parenthesis to pull the movie clip number from each annotation file name')
   parser.add_argument('--output_path', required=True, help='Path the output folder for baseline annotations')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Enables display of the aligned annotations')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ComputeBaselineAnnotations(args.input_clip_annotations_path, args.clip_times_csv_path, args.output_path, args.movie_name_regex, args.clip_number_regex, args.show_plots)
