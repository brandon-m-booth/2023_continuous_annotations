import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'util')))
import util
import agreement_metrics as agree

def AlignDTW(input_anno_dir, output_anno_dir, max_warp_seconds=5, resample_hz=None, show_plots=False):
   if os.path.isdir(input_anno_dir):
      input_annos = glob.glob(os.path.join(input_anno_dir, '*.csv'))
   elif os.path.isfile(input_anno_dir):
      input_annos = [input_anno_dir]
   else:
      input_annos = glob.glob(input_anno_dir)

   if not os.path.isdir(output_anno_dir):
      os.makedirs(output_anno_dir)

   for input_anno in input_annos:
      if show_plots:
         fig, ax = plt.subplots(1,1, figsize=(11,9))

      df = pd.read_csv(input_anno)

      # Use time column as an index
      df = df.set_index(df.columns[0])
      #for cidx in range(df.shape[1]):
      #   plt.plot(df.index, df.iloc[:,cidx])
      #plt.show()
      #continue
      #df.index = [t.timestamp() for t in df.index]

      # Use the most agreeable annotation as the reference
      norm_diff_df = util.NormedDiff(df)
      sda_mat = agree.SDACorrected(norm_diff_df)
      np.fill_diagonal(sda_mat.values, 0)
      mean_agree = sda_mat.mean(axis=0)
      ref_idx = np.argmax(mean_agree)

      # Resample for even spacing
      df.index = [datetime.fromtimestamp(t/1000.0, pytz.utc) for t in df.index]
      df_interp = df.resample('100ms').mean().interpolate()
      df_ref = df_interp.iloc[:,ref_idx]
      
      dtw_df = agree.DTWReference(df_interp, df_ref, max_warp_distance=max_warp_seconds*10)

      # Add the NaN values back in where they existed in the original signal
      dummy_df = df_interp.copy()
      for col in dummy_df.columns:
         dummy_df[col].values[:] = np.nan
      merge_nan_df = df.replace(to_replace=np.nan, value=np.inf)
      for col in df.columns:
         merge_nan_col_df = merge_nan_df[col].combine(dummy_df[col], min)
         merge_nan_col_df.interpolate(method='nearest', inplace=True)
         inf_mask = np.isinf(merge_nan_col_df.loc[dummy_df.index])
         dtw_df[col][inf_mask] = np.nan

      if resample_hz is not None:
         #dtw_df = dtw_df.resample(pd.to_timedelta(str(1.0/resample_hz)+'s')).mean().interpolate()
         dtw_df = dtw_df.resample(pd.to_timedelta(str(1.0/resample_hz)+'s')).mean()
         time_scale = 1.0
      else:
         time_scale = 10.0 # 100ms -> seconds

      dtw_df.index = [t.timestamp() for t in dtw_df.index]
      dtw_df_cols = dtw_df.columns
      dtw_df_cols = dtw_df_cols.insert(0, 'Time(sec)')
      dtw_df['Time(sec)'] = dtw_df.index*time_scale
      dtw_df = dtw_df[dtw_df_cols]

      if show_plots:
         for anno_col in dtw_df.columns[1:]:
            ax.plot(dtw_df['Time(sec)'], dtw_df[anno_col], label=anno_col)
         ax.legend()
         plt.title(os.path.basename(input_anno) + ' DTW annotations')
         plt.show()

      # Write separate files for each annotation
      for dtw_df_col in dtw_df.columns[1:]:
         output_file_name = os.path.basename(input_anno).split('.')[0]+'_'+dtw_df_col+'.csv'
         out_df = dtw_df[['Time(sec)', dtw_df_col]]
         out_df.to_csv(os.path.join(output_anno_dir, output_file_name), index=False, header=True)

   return

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_annos', required=True, help='Path to the (filtered) annotation csv file or folder containing them')
   parser.add_argument('--output_path', required=True, help='Path the folder for aligned annotations')
   parser.add_argument('--max_warp_seconds', default=5, type=int, required=False, help='Maximum number of seconds that the annotations are allowed to be locally distorted')
   parser.add_argument('--resample_hz', required=False, type=int, help='Sampling rate of output')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Enables display of the aligned annotations')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   AlignDTW(args.input_annos, args.output_path, args.max_warp_seconds, args.resample_hz, args.show_plots)
