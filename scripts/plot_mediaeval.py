#!/usr/bin/env python3

import os
import pdb
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

do_show_plots = True
show_plots_threshold = 0.75

# TODO:
# Setup SVM classifier for the baseline? Or is it the shallow NN?
# Verify I can reproduce the paper results from 2015
# Compare my GT to the average GT using the SVM method on the highly correlated annotations

def GetCombinations(n, r):
   return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

def GetStatistic(df, stat_str):
   num_annos = df.shape[1]
   num_combos = GetCombinations(num_annos, 2)
   stat = ((df.corr(method=stat_str).sum().sum()-num_annos)/2.0)/float(num_combos)
   return stat

def PlotMediaeval():
   cur_file_path = os.path.dirname(os.path.realpath(__file__))
   mediaeval_anno_path = os.path.join(cur_file_path, '..', 'datasets', 'mediaeval', 'annotations', 'annotations per each rater', 'dynamic (per second annotations)')
   arousal_anno_path = os.path.join(mediaeval_anno_path, 'arousal')
   valence_anno_path = os.path.join(mediaeval_anno_path, 'valence')

   for anno_dimension in [('arousal', arousal_anno_path), ('valence', valence_anno_path)]:
      cur_dimension = anno_dimension[0]
      anno_files_path = anno_dimension[1]
      anno_files = glob.glob(os.path.join(anno_files_path, '*.csv'))

      corr_dict = {'task': [], 'pearson': []}
      for anno_file in anno_files:
         df = pd.read_csv(anno_file)
         if 'WorkerId' in df.columns:
            df = df.drop(columns='WorkerId')
         num_annos = df.shape[0]
         times_str = [x[7:-2] for x in df.columns]
         times_sec = [float(t)/1000.0 for t in times_str]
         new_df = df.T
         new_df.index = times_sec
         new_df.columns = ['A%0d'%(i) for i in range(new_df.shape[1])]

         avg_pearson = GetStatistic(new_df, 'pearson')
         corr_dict['task'].append(os.path.basename(anno_file))
         corr_dict['pearson'].append(avg_pearson)

         if do_show_plots and avg_pearson > show_plots_threshold:
            plt.figure()
            ax = plt.gca()
            new_df.plot(ax=ax)
            plt.title(cur_dimension + ': '+ os.path.basename(anno_file))
            plt.show()

      corr_df = pd.DataFrame(corr_dict)
      corr_df = corr_df.reindex(columns=['task', 'pearson'])
      corr_df = corr_df.sort_values(by=['pearson'], ascending=False)
      corr_df.to_csv(os.path.join(cur_file_path, '%s_corr.csv'%(cur_dimension)), index=False)

   return

if __name__=='__main__':
   PlotMediaeval()
