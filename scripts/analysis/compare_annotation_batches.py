import re
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import agreement_metrics as agree
from util import TikzplotlibFixNCols

def CompareAnnotationBatches(json_config_path):
   with open(json_config_path) as config_fp:
      config = json.load(config_fp)

   movie_name_regprog = re.compile(config['movie_name_regex'])

   movie_batch_dict = {}
   for batch_item in config['batch_data']:
      batch_name = batch_item['name']
      movie_batch_files = glob.glob(os.path.join(batch_item['annos_path'], '*.csv'))
      for movie_batch_file in movie_batch_files:
         movie_name = movie_name_regprog.match(os.path.basename(movie_batch_file)).group(1)
         #movie_name = movie_name.replace('__', ': ')
         #movie_name = movie_name.replace('_', ' ')
         if not movie_name in movie_batch_dict.keys():
            movie_batch_dict[movie_name] = []

         df = pd.read_csv(movie_batch_file)
         movie_batch_dict[movie_name].append(df)

   for movie_name in movie_batch_dict.keys():
      spearman_corrs = []
      kendall_taus = []
      prev_df = None
      for df in movie_batch_dict[movie_name]:
         if prev_df is not None:
            corr_df = pd.DataFrame(data=pd.concat((prev_df.iloc[:,1], df.iloc[:,1]), axis=1))
            spearman_corrs.append(agree.SpearmanCorr(corr_df).iloc[0,1])
            kendall_taus.append(agree.KendallTauCorr(corr_df).iloc[0,1])
         prev_df = df

      print("Movie: "+movie_name)
      print("  Spearman: "+str(spearman_corrs))
      print("  Kendall\'s Tau: "+str(kendall_taus))

      batches = range(1,len(spearman_corrs)+1)
      fig = plt.figure()
      plt.plot(batches, spearman_corrs, label='Spearman')
      plt.plot(batches, kendall_taus, label='Kendall\'s Tau')
      plt.legend()
      plt.xlabel('Batch')
      plt.ylabel('Agreement Value')
      plt.title(movie_name)

      if not os.path.exists(config['tikz_out_path']):
         os.makedirs(config['tikz_out_path'])

      TikzplotlibFixNCols(fig)
      tikz_out_file_path = os.path.join(config['tikz_out_path'], 'batch_comparison_'+movie_name+'.tex')
      tikzplotlib.save(tikz_out_file_path, figure=fig)

      plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--json_config', required=True, help='Path to json config file')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CompareAnnotationBatches(args.json_config)
