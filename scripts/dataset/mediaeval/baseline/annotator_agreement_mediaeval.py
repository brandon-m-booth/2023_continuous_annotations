#!/usr/bin/env python3

import os
import pdb
import glob
import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

output_file_path = './annotator_agreement.csv'

def cmp_file_names(x, y):
   x_number_str = os.path.basename(x).split('.')[0]
   y_number_str = os.path.basename(y).split('.')[0]
   if int(x_number_str) < int(y_number_str):
      return -1
   else:
      return 1

def NormalizeAnnotations(df):
   norm_df = df.copy()
   for col_idx in range(df.shape[1]):
      norm_df.iloc[:,col_idx] += df.mean(axis=0)[col_idx]
   return norm_df - df.mean().mean()

# DF columns are annotators, row are time indices
def ComputeCronbachAlpha(df):
   vars_vals = df.var(axis=0, ddof=1)
   sum_vals = df.sum(axis=1)
   N = df.shape[1]
   return (N/(N-1))*(1.0 - vars_vals.sum() / sum_vals.var(ddof=1))

def MediaevalAnnotatorAgreement():
   cur_file_path = os.path.dirname(os.path.realpath(__file__))
   mediaeval_anno_path = os.path.join(cur_file_path, '..', '..', 'datasets', 'mediaeval', 'annotations', 'annotations per each rater', 'dynamic (per second annotations)')
   arousal_anno_path = os.path.join(mediaeval_anno_path, 'arousal')
   valence_anno_path = os.path.join(mediaeval_anno_path, 'valence')

   results_dict = {}
   for anno_dimension in [('arousal', arousal_anno_path), ('valence', valence_anno_path)]:
      print("Computing %s annotator agreement..."%(anno_dimension[0]))
      cur_dimension = anno_dimension[0]
      anno_files_path = anno_dimension[1]
      anno_files = glob.glob(os.path.join(anno_files_path, '*.csv'))

      # Use all but the last 58 songs, which are the 2015 evaluation full-length songs
      anno_files.sort(key=functools.cmp_to_key(cmp_file_names))
      anno_files = anno_files[:-58]

      # Compute annotator agreement
      anno_agree_dict = {}
      for anno_file in anno_files:
         task_name = os.path.basename(anno_file)

         # Read and format annotation data
         anno_df = pd.read_csv(anno_file)
         if 'WorkerId' in anno_df.columns:
            anno_df = anno_df.drop(columns='WorkerId')
         num_annos = anno_df.shape[0]
         times_str = [x[7:-2] for x in anno_df.columns]
         times_sec = [float(t)/1000.0 for t in times_str]
         anno_df = anno_df.T
         anno_df.index = times_sec
         anno_df.columns = ['A%0d'%(i) for i in range(anno_df.shape[1])]

         # Normalize annotations
         anno_df = NormalizeAnnotations(anno_df)

         # Compute annotator agreement measure(s)
         anno_agree_dict[task_name] = {'name': task_name}

         # Cronbach's Alpha
         alpha = ComputeCronbachAlpha(anno_df)
         alpha = max(0,alpha) # Clamping to positive values, as per the paper
         anno_agree_dict[task_name]['cronbach_alpha_'+cur_dimension] = alpha

         # Pearson correlation
         num_pairs = ((anno_df.shape[1]-1)*anno_df.shape[1])/2
         avg_pearson = (anno_df.corr(method='pearson').sum().sum()-anno_df.shape[1])/(2*num_pairs)
         anno_agree_dict[task_name]['annotator_pearson_mean_'+cur_dimension] = avg_pearson

         # Store the sample length per task
         anno_agree_dict[task_name]['num_samples_'+cur_dimension] = anno_df.shape[0]

      # Save results
      anno_agree_df = pd.DataFrame(anno_agree_dict).T
      results_dict[cur_dimension] = anno_agree_df

   results_df = None
   for result in results_dict.keys():
      if results_df is None:
         results_df = results_dict[result]
      else:
         results_df = results_df.merge(results_dict[result], on='name', how='outer')
   results_df['cronbach_alpha_sum'] = results_df.loc[:,'cronbach_alpha_arousal'] + results_df.loc[:,'cronbach_alpha_valence']
   results_df['cronbach_alpha_prod'] = results_df.loc[:,'cronbach_alpha_arousal']*results_df.loc[:,'cronbach_alpha_valence']
   anno_agree_df = results_df.reindex(columns=['name', 'num_samples_arousal', 'num_samples_valence', 'annotator_pearson_mean_arousal', 'annotator_pearson_mean_valence', 'cronbach_alpha_arousal', 'cronbach_alpha_valence', 'cronbach_alpha_sum', 'cronbach_alpha_prod'])
   anno_agree_df = anno_agree_df.sort_values(by=['cronbach_alpha_prod'], ascending=False)
   anno_agree_df.to_csv(os.path.join(cur_file_path, output_file_path), index=False)
   return

if __name__=='__main__':
   MediaevalAnnotatorAgreement()
