#!/usr/bin/env python
import os
import sys
import pdb
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
from FileIO import GetCsvData, SaveCsvData
from PrettyPlotter import pretty


def MergeClipSignals(json_config_path, do_show_plots=True):
   with open(json_config_path) as config_fp:
      config = json.load(config_fp)

   if not os.path.isdir(config['merged_output_path']):
      os.makedirs(config['merged_output_path'])

   movie_cut_times_df = pd.read_csv(config['movie_cut_times_path'])
   movie_cut_times_df['Movie Name'] = [os.path.basename(x).split('.')[0] for x in movie_cut_times_df['Movie Path']]

   movie_names = [x['match_name'].split('_cut')[0] for x in config['clip_data']]
   movie_names = np.unique(movie_names)

   # Prepare dict of movie clips grouped by movie name
   signals_dict = {}
   for movie_name in movie_names:
      signals_dict[movie_name] = []

   # Find the clips belonging to each movie by name
   constructed_signal_paths = glob.glob(os.path.join(config['output_path'], '*.csv'))
   for constructed_signal_path in constructed_signal_paths:
      found_movie = False
      for movie_name in movie_names:
         if movie_name in os.path.basename(constructed_signal_path):
            found_movie = True
            break
      if not found_movie:
         print("ERROR: Could not find parent movie for clip %s"%(constructed_signal_path))
         pdb.set_trace()

      signals_dict[movie_name].append(constructed_signal_path)

   # Sort the clips for each movie by cut order (e.g., cut1, cut2, etc.) then merge them
   for movie_name in movie_names:
      movie_clip_paths = signals_dict[movie_name]
      sorted_movie_clip_paths = sorted(movie_clip_paths, key=lambda clip_path: int(os.path.basename(clip_path).split('_cut')[1].split('.')[0]))

      merged_df = None
      for movie_clip_path in sorted_movie_clip_paths:
         signal_df = pd.read_csv(movie_clip_path)

         # Find the corresponding movie cut times entry
         cut_name = os.path.basename(movie_clip_path).split('_')[-1].split('.')[0]
         cut_time_df = movie_cut_times_df.loc[(movie_cut_times_df['Movie Name']==movie_name).values & (movie_cut_times_df['Cut Name']==cut_name).values, :]
         start_time_sec = cut_time_df['Start Time Sec'].iloc[0]
         end_time_sec = cut_time_df['End Time Sec'].iloc[0]
         signal_df.iloc[:,0] += start_time_sec

         if merged_df is None:
            merged_df = signal_df
         else:
            merged_df = pd.concat((merged_df,signal_df), axis=0, ignore_index=True)

      merged_df.sort_values(by='Time(sec)', inplace=True)
      merged_df.to_csv(os.path.join(config['merged_output_path'], 'warped_annotation_'+movie_name+'.csv'), index=False)

      if do_show_plots:
         plt.figure()
         norm_merged_df = merged_df.copy()
         norm_merged_df.iloc[:,1] = (norm_merged_df.iloc[:,1] + 100)/200.0
         plt.plot(norm_merged_df.iloc[:,0], norm_merged_df.iloc[:,1], 'r-')
         pretty(plt)
         plt.xlabel('Time(s)', fontsize=24)
         plt.ylabel('Movie Violence', fontsize=24)
         plt.axis([norm_merged_df.iloc[0,0], norm_merged_df.iloc[-1,0],0,1])
         plt.title(movie_name)

         output_tikz_path = os.path.join(config['merged_output_path'], 'warped_annotation_'+movie_name+'.tex')
         tikzplotlib.save(output_tikz_path)
   return

def DoConstructSignal(json_config_path, do_show_plots=True):
   if do_show_plots:
      plt.ion()

   with open(json_config_path) as config_fp:
      config = json.load(config_fp)

   if not (config['interpolation_method'] == 'average' or config['interpolation_method'] == 'trapezoidal'):
      print("Error: Interpolation method must be 'average' or 'trapezoidal'")
      pdb.set_trace()
      return

   if not os.path.isdir(config['output_path']):
      os.makedirs(config['output_path'])

   embedding_df = pd.read_csv(config['embedding_path'], header=None)
   embedding_code_df = pd.read_csv(config['embedding_code_path'])

   # Invert and renormalize the embedding
   if config['flip_embedding']:
      embedding_df = 1.0-embedding_df
   embedding_df = embedding_df*(config['norm_scale'][1]-config['norm_scale'][0]) + config['norm_scale'][0]

   for clip_data in config['clip_data']:
      if '.' in clip_data['annotations_folder']: # Assume path points to a file(s)
         annotation_files = glob.glob(clip_data['annotations_folder'])
      else:
         annotation_files = glob.glob(os.path.join(clip_data['annotations_folder'], '*.csv'))
      annotations_df = None
      for annotation_file in annotation_files:
         anno_df = pd.read_csv(annotation_file, index_col=0)
         if annotations_df is None:
            annotations_df = anno_df
         else:
            annotations_df = annotations_df.merge(anno_df, how='outer', left_index=True, right_index=True)
      average_annotation = annotations_df.mean(axis=1)
      
      seg_seq_df = pd.read_csv(clip_data['fused_seg_seq_path'])
      intervals_df = pd.read_csv(clip_data['const_intervals_path'], header=None)

      # Gather all the embedding values for this clip
      match_embedding_indices = []
      for i in range(embedding_code_df.shape[0]):
         if clip_data['match_name'] in embedding_code_df['clip_name'][i]:
            match_embedding_indices.append(i)
      embedding_code_clip_df = embedding_code_df.iloc[match_embedding_indices,:]
      sorted_match_embedding_indices = np.argsort(embedding_code_clip_df['start'])
      embedding_clip_df = embedding_df.iloc[np.array(match_embedding_indices)[sorted_match_embedding_indices],:]

      constructed_signal = pd.DataFrame(data=np.nan*np.zeros((len(average_annotation.index), 2)), index=average_annotation.index, columns=['Time(sec)', 'Data'])
      constructed_signal.iloc[:,0] = average_annotation.index

   #interval_values_globs = glob.glob(interval_values_glob)
   #for interval_values_csv in interval_values_globs:
   #   seg_seq_header, seg_seq = GetCsvData(seg_seq_csv)
   #   dummy_header, intervals = GetCsvData(intervals_csv, first_line_header=False)
   #   dummy_header, interval_values = GetCsvData(interval_values_csv, first_line_header=False)
   #   intervals = intervals.astype(int)

   #   # Separate times from segment sequence
   #   times = seg_seq[:,0]
   #   segments = seg_seq[:,1]
   #   sampling_rate = 1.0/(times[1]-times[0])

   #   constructed_signal = pd.DataFrame(data={'Time_sec': times, 'Data': len(segments)*[np.nan]})
   #   constructed_signal = constructed_signal.reindex(columns=['Time_sec', 'Data'])

      if config['interpolation_method'] == 'average':
         # Interpolate between constant intervals using the average annotation signal
         current_frame = 0
         interval_idx = 0
         while current_frame < constructed_signal.shape[0]:
            # Shift the next interval
            if interval_idx < intervals_df.shape[0]:
               interval = intervals_df.iloc[interval_idx, :].values
               annotation_interval_average = average_annotation.iloc[interval[0]:interval[1]+1].mean()
               interval_shift = embedding_clip_df.iloc[interval_idx,0] - annotation_interval_average
               constructed_signal.iloc[interval[0]:interval[1]+1,1] = average_annotation[interval[0]:interval[1]+1] + interval_shift
               end_frame = intervals_df.iloc[interval_idx,0]
            else:
               interval_shift = 0
               end_frame = len(constructed_signal)-1

            # If not inside the interval just shifted, skew the frames
            # leading up to it
            if current_frame <= end_frame:
               scale = constructed_signal.iloc[end_frame,1] - constructed_signal.iloc[current_frame,1]
               avg_annotation_scale = average_annotation[end_frame] - average_annotation[current_frame]
               #if avg_annotation_scale*scale > 0:
               if avg_annotation_scale*scale > 0 and abs(avg_annotation_scale) > abs(scale):
                  if end_frame > current_frame:
                     constructed_signal.iloc[current_frame:end_frame+1,1] = (average_annotation[current_frame:end_frame+1] - average_annotation[current_frame])/(average_annotation[end_frame]-average_annotation[current_frame])*scale + constructed_signal.iloc[current_frame,1]
                     #constructed_signal.iloc[current_frame:end_frame+1,1] = (average_annotation[current_frame:end_frame+1] - average_annotation[current_frame])/(average_annotation[end_frame]-average_annotation[current_frame])*scale + average_annotation[current_frame]+bias
                  else:
                     pass
               else:
                  centered_average_annotation = (average_annotation.iloc[current_frame:end_frame+1]-average_annotation[current_frame])+constructed_signal.iloc[current_frame,1]
                  end_frame_diff = constructed_signal.iloc[end_frame,1] - centered_average_annotation.iloc[-1]
                  frame_diffs = np.linspace(0,end_frame_diff,end_frame-current_frame+1)
                  constructed_signal.iloc[current_frame:end_frame+1,1] = centered_average_annotation + frame_diffs
               #if current_frame > 0:
               #   bias = constructed_signal.iloc[current_frame,1] - average_annotation[current_frame]
               #else:
               #   bias = 0
               #scale = constructed_signal[end_frame]+interval_shift - (constructed_signal[current_frame]+bias)
               #scale = average_annotation[end_frame]+interval_shift - (average_annotation[current_frame]+bias)
                  #constructed_signal.iloc[current_frame:end_frame+1,1] = average_annotation[current_frame]+(bias+interval_shift)/2.0

            if interval_idx < intervals_df.shape[0]:
               #last_value = warped_signal[intervals[interval_idx,1]]
               current_frame = intervals_df.iloc[interval_idx,1]
               interval_idx += 1
            else:
               current_frame = len(constructed_signal)

      elif config['interpolation_method'] == 'trapezoidal':
         # Fill in the constant interval values
         for interval_idx in range(intervals_df.shape[0]):
            interval = intervals_df.iloc[interval_idx,:]
            embedding_value = embedding_clip_df.iloc[interval_idx,0]
            constructed_signal.iloc[interval[0]:interval[1]+1,1] = embedding_value

         # Linearly interpolate between constant intervals
         constructed_signal.interpolate(method='linear', inplace=True)

      # Plot the results
      if do_show_plots:
         plt.figure()
         for i in range(intervals_df.shape[0]):
            interval = intervals_df.iloc[i,:]
            interval_times = [average_annotation.index[interval[0]], average_annotation.index[interval[1]]]
            values = 2*[embedding_clip_df.iloc[i,0]]
            if i > 0:
               plt.plot(interval_times, values, 'g-o', label='_nolegend_')
            else:
               plt.plot(interval_times, values, 'g-o')

         plt.plot(constructed_signal.iloc[:,0], constructed_signal.iloc[:,1], 'r-')
         if config['interpolation_method'] == 'average':
            plt.plot(average_annotation.index, average_annotation.values, 'b--')
         
         pretty(plt)

         plt.xlabel('Time(s)', fontsize=24)
         plt.ylabel('Movie Violence', fontsize=24)
         plt.axis([constructed_signal.iloc[0,0],constructed_signal.iloc[-1,0],config['norm_scale'][0],config['norm_scale'][1]])
         legend_list = []
         legend_list.extend(['Embedded Intervals', 'Constructed Signal'])
         if config['interpolation_method'] == 'average':
            legend_list.extend(['Average Annotation'])
         plt.legend(legend_list, loc='upper left', bbox_to_anchor=(1,1), frameon=False, prop={'size':24})
         plt.title(clip_data['match_name'])

         output_tikz_path = os.path.join(config['output_path'], 'warped_annotation_'+clip_data['match_name']+'.tex')
         tikzplotlib.save(output_tikz_path)

      output_signal_path = os.path.join(config['output_path'], 'warped_annotation_'+clip_data['match_name']+'.csv')
      constructed_signal.to_csv(output_signal_path, index=False, header=True)

   MergeClipSignals(json_config_path, do_show_plots)

   if do_show_plots:
      plt.ioff()
      plt.show()

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--json_config_path', required=True, help='JSON config file')
   parser.add_argument('--show_plots', required=False, action='store_true')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   DoConstructSignal(args.json_config_path, args.show_plots)
