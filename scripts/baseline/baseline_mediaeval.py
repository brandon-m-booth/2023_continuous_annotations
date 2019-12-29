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

import annotator_agreement

do_show_plots = False

def cmp_file_names(x, y):
   x_number_str = os.path.basename(x).split('.')[0]
   y_number_str = os.path.basename(y).split('.')[0]
   if int(x_number_str) < int(y_number_str):
      return -1
   else:
      return 1

def MediaevalMLBaseline():
   cur_file_path = os.path.dirname(os.path.realpath(__file__))
   mediaeval_anno_path = os.path.join(cur_file_path, '..', '..', 'datasets', 'mediaeval', 'annotations', 'annotations per each rater', 'dynamic (per second annotations)')
   arousal_anno_path = os.path.join(mediaeval_anno_path, 'arousal')
   valence_anno_path = os.path.join(mediaeval_anno_path, 'valence')
   mediaeval_features_path = os.path.join(cur_file_path, '..', '..', 'datasets', 'mediaeval', 'features')

   # Create dictionary of features files
   features_files = glob.glob(os.path.join(mediaeval_features_path, '*.csv'))
   features_dict = {}
   for features_file in features_files:
      features_dict[os.path.basename(features_file)] = features_file

   for anno_dimension in [('arousal', arousal_anno_path), ('valence', valence_anno_path)]:
      print("Computing %s baseline..."%(anno_dimension[0]))
      cur_dimension = anno_dimension[0]
      anno_files_path = anno_dimension[1]
      anno_files = glob.glob(os.path.join(anno_files_path, '*.csv'))

      # Use the last 58 songs for evaluation, which are the full-length ones from the 2015 data set
      anno_files.sort(key=functools.cmp_to_key(cmp_file_names))
      #task_train = [os.path.basename(x) for x in anno_files[:-58]]
      task_test = [os.path.basename(x) for x in anno_files[-58:]]

      # Use 431 clips with the top annotator agreement for training (per the 2015 data set)
      if not os.path.exists(annotator_agreement.output_file_path):
         print("Annotator agreement file not found: %s"%(annotator_agreement.output_file_path))
         print("Running annotator agreement script to generate this file...")
         print("===========================================================")
         annotator_agreement.MediaevalAnnotatorAgreement()
         print("Finished computing annotator agreement")
         print("===========================================================")
      anno_agree_df = pd.read_csv(annotator_agreement.output_file_path)
      anno_agree_df.sort_values(by=['cronbach_alpha_prod'], ascending=False)
      task_train = anno_agree_df.loc[0:431,'name'].tolist()

      # Assemble features and annotations
      task_dict = {}
      for anno_file in anno_files:
         task_name = os.path.basename(anno_file)
         if not task_name in features_dict.keys():
            print("Unable to find features for annotation task: %s"%(task_name))
            continue

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

         # Read and format the features data
         feature_df = pd.read_csv(features_dict[task_name], sep=';')
         feature_df = feature_df.set_index('frameTime')

         # Compute the baseline average annotation
         anno_avg_df = pd.DataFrame({'annotation_mean': anno_df.mean(axis=1)})

         # Join the annotations and features to remove excess rows
         combined_df = anno_avg_df.join(feature_df, how='inner')

         # Add the features and labels to the task dict
         task_dict[task_name] = {}
         task_dict[task_name]['features_df'] = combined_df.loc[:, combined_df.columns != 'annotation_mean']
         task_dict[task_name]['labels_df'] = combined_df.loc[:, 'annotation_mean']

      #train_dict = {}
      #for task in task_train:
      #   train_dict[task] = task_dict[task]
      #test_dict = {}
      #for task in task_test:
      #   test_dict[task] = task_dict[task]

      # Construct train/test folds without splitting up songs (similar to LOSO-CV)
      #k_fold = KFold(n_splits=5, shuffle=True)
      #task_keys = np.array(task_dict.keys())
      #num_tasks = len(task_keys)
      #for train_task_indices, test_task_indices in k_fold.split(range(num_tasks)):
      task_keys = np.array(task_dict.keys())
      train_task_indices = [i for i in range(len(task_keys)) if task_keys[i] in task_train ]
      test_task_indices = [i for i in range(len(task_keys)) if task_keys[i] in task_test]
      if True:

         # Create train/test dataframes for features and labels
         features_df_train = None
         labels_df_train = None
         for task_name in task_keys[train_task_indices]:
            if features_df_train is None:
               features_df_train = task_dict[task_name]['features_df']
               labels_df_train = task_dict[task_name]['labels_df']
            else:
               features_df_train = features_df_train.append(task_dict[task_name]['features_df'])
               labels_df_train = labels_df_train.append(task_dict[task_name]['labels_df'])

         features_df_test = None
         labels_df_test = None
         for task_name in task_keys[test_task_indices]:
            if features_df_test is None:
               features_df_test = task_dict[task_name]['features_df']
               labels_df_test = task_dict[task_name]['labels_df']
            else:
               features_df_test = features_df_test.append(task_dict[task_name]['features_df'])
               labels_df_test = labels_df_test.append(task_dict[task_name]['labels_df'])

         # Normalize the features
         norm_scaler = MinMaxScaler()
         features_df_train_norm = norm_scaler.fit_transform(features_df_train)
         features_df_test_norm = norm_scaler.transform(features_df_test)

         #print('Performing grid search for SVR model...')
         #param_grid = [{'C': [1e2, 1e1, 1e-1, 1e-2], 'gamma': [0.250, 0.125, 0.0625], 'kernel': ['rbf']}]
         #cv_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=3, n_jobs=-1)
         #cv_search.fit(features_df_train_norm, labels_df_train)
         #best_c = cv_search.best_estimator_.C
         #best_gamma = cv_search.best_estimator_.gamma
         #print('Best C: ', best_c)
         #print('Best Gamma: ', best_gamma)
         # From the paper:
         best_c = 1e2
         best_gamma = 0.125
         # From my experiments:
         #best_c = 1e-1
         #best_gamma = 0.25

         # Retrain on all training data, then predict the test data
         print("Retraining on all training data...")
         model = SVR(C=best_c, gamma=best_gamma, kernel='rbf')
         model.fit(features_df_train_norm, labels_df_train)
         pred = model.predict(features_df_test_norm)

         # Compute the evaluation metric per test set song
         cur_pred_idx = 0
         for task_name in task_keys[test_task_indices]:
            labels_test = task_dict[task_name]['labels_df']
            pred_song = pred[cur_pred_idx:cur_pred_idx+len(labels_test)]
            rmse = math.sqrt(mean_squared_error(labels_test, pred_song))
            corr_pearson = pearsonr(labels_test, pred_song)[0]
            task_dict[task_name]['pred_rmse'] = rmse
            task_dict[task_name]['pred_pearson'] = corr_pearson

            if do_show_plots:
               plt.figure()
               plt.plot(labels_test, 'r-')
               plt.plot(labels_test.index, pred_song, 'b--')
               plt.title("Prediction vs average annotation for %s"%(task_name))
               plt.show()
            cur_pred_idx += len(labels_test)

      # Save results
      task_df = pd.DataFrame(task_dict).T
      task_df = task_df.reindex(columns=['name', 'pred_rmse', 'pred_pearson'])
      task_df = task_df.sort_values(by=['pred_pearson'], ascending=False)
      task_df.to_csv(os.path.join(cur_file_path, '%s_corr.csv'%(cur_dimension)), index=False)

   return

if __name__=='__main__':
   MediaevalMLBaseline()
