import re
import os
import sys
import glob
import pickle
import argparse
import tikzplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
import agreement_metrics as agree

functionals = [np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nansum]

def CompareAnnotationRatings(baseline_annos_folder_path, warped_annos_folder_path, csm_ratings_path, output_folder, baseline_anno_movie_name_regex='(.*)_fused_baseline.csv', warped_anno_movie_name_regex='warped_annotation_(.*).csv'):
   # Gather CSM movie titles with available ratings information
   baseline_movie_name_regprog = re.compile(baseline_anno_movie_name_regex)
   warped_movie_name_regprog = re.compile(warped_anno_movie_name_regex)
   with open(csm_ratings_path, 'rb') as csm_ratings_file:
      movie_csm_data_list = pickle.load(csm_ratings_file)
   titles = [x['title'] for x in movie_csm_data_list]

   # Build dictionary associating movie violence ratings with their respective annotations
   anno_rating_dict = {}
   warped_anno_files = glob.glob(os.path.join(warped_annos_folder_path, '*.csv'))
   for warped_anno_file in warped_anno_files:
      movie_name = warped_movie_name_regprog.match(os.path.basename(warped_anno_file)).group(1)
      movie_name = movie_name.replace('__', ': ')
      movie_name = movie_name.replace('_', ' ')

      if not movie_name in anno_rating_dict.keys():
         ratings_list_idx = titles.index(movie_name)
         if ratings_list_idx < 0:
            print("ERROR: Movie name not found in the CSM ratings file")
            pdb.set_trace()
         anno_rating_dict[movie_name] = {}

      movie_csm_data = movie_csm_data_list[ratings_list_idx]
      violence_rating = movie_csm_data['contentGrid']['violence']['rating']
      warped_anno_df = pd.read_csv(warped_anno_file)
      anno_rating_dict[movie_name] = {'violence': violence_rating, 'warped': warped_anno_df}

   # Add baseline annotations to the dict
   baseline_anno_files = glob.glob(os.path.join(baseline_annos_folder_path, '*.csv'))
   for baseline_anno_file in baseline_anno_files:
      movie_name = baseline_movie_name_regprog.match(os.path.basename(baseline_anno_file)).group(1)
      movie_name = movie_name.replace('__', ': ')
      movie_name = movie_name.replace('_', ' ')

      if not movie_name in anno_rating_dict.keys():
         print("ERROR: Movie name not found in the CSM ratings file")
         pdb.set_trace()

      baseline_anno_df = pd.read_csv(baseline_anno_file)
      anno_rating_dict[movie_name]['baseline'] = baseline_anno_df
   
   # Apply functionals to the annotations to reduce their dimension for comparison with violence ratings
   func_dict = {}
   for movie_name in anno_rating_dict.keys():
      func_dict[movie_name] = {'baseline': {}, 'warped': {}}
      baseline_anno = anno_rating_dict[movie_name]['baseline'].iloc[:,1]
      warped_anno = anno_rating_dict[movie_name]['warped'].iloc[:,1]

      # Apply each functional to the annotation per movie
      for functional in functionals:
         func_name = functional.__name__
         func_value_baseline = functional(baseline_anno)
         func_value_warped = functional(warped_anno)
         func_dict[movie_name]['baseline'][func_name] = func_value_baseline
         func_dict[movie_name]['warped'][func_name] = func_value_warped

   # Apply different evaluation metrics comparing the functional values to the violence ratings
   movie_names_list = anno_rating_dict.keys()
   violence_ratings = []
   for movie_name in movie_names_list:
      violence_ratings.append(anno_rating_dict[movie_name]['violence'])
   
   func_names = []
   for functional in functionals:
      func_names.append(functional.__name__)

   ratings_mat = np.array(violence_ratings)
   for func_name in func_names:
      func_values_baseline = []
      func_values_warped = []
      for movie_name in movie_names_list:
         func_values_baseline.append(func_dict[movie_name]['baseline'][func_name])
         func_values_warped.append(func_dict[movie_name]['warped'][func_name])
      ratings_mat = np.vstack((ratings_mat, func_values_baseline, func_values_warped))
   ratings_df = pd.DataFrame(data=ratings_mat.T, columns=['CSM']+np.array([[x+' baseline', x+' warped'] for x in func_names]).flatten().tolist(), index=movie_names_list)

   spearman = pd.Series(agree.SpearmanCorr(ratings_df).iloc[0,1:], name='Spearman')
   tau = pd.Series(agree.KendallTauCorr(ratings_df).iloc[0,1:], name='Kendall\'s Tau')
   corr_df = pd.concat((spearman, tau), axis=1)
   sns_corr_df = pd.DataFrame(data=np.zeros((corr_df.shape[0]*corr_df.shape[1],4)), columns=['Metric', 'Functional', 'Annotation', 'Correlation'])

   sns_idx = 0
   for metric in corr_df.columns:
      for func_name in corr_df.index:
         func, anno_name = func_name.split(' ')
         sns_corr_df.iloc[sns_idx,:] = [metric, func, anno_name, corr_df.loc[func_name, metric]]
         sns_idx += 1

   if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
   output_file_path = os.path.join(output_folder, 'csm_annotation_comparison.csv')
   sns_corr_df.to_csv(output_file_path, index=False, header=True)

   # TODO - multi bar plot with each metrics grouped and each group having many functionals
   #plt.bar([pearson,spearman,ccc,kappa,alpha,icc])
   #ax = sns.barplot(x="Metric", y="Correlation", hue="Functional", data=sns_corr_df)
   ax = sns.barplot(x="Annotation", y="Correlation", hue="Functional", data=sns_corr_df[sns_corr_df['Metric'] == 'Spearman'])
   #plt.xlabel(['Pearson','Spearman','CCC','Kappa','Alpha','ICC'])
   #plt.ylabel('Correlation, CSM vs. ratings')
   plt.title('Movie Violence Validation')

   output_tikz_path = os.path.join(output_folder, 'csm_annotation_comparison.tex')
   tikzplotlib.save(output_tikz_path)

   plt.show()


   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--baseline_annotations_path', required=True, help='Folder containing baseline annotations, one file per movie')
   parser.add_argument('--warped_annotations_path', required=True, help='Folder containing warped annotations, one file per movie')
   parser.add_argument('--csm_ratings_path', required=True, help='Path to CSM pkl containing movie ratings')
   parser.add_argument('--output_folder', required=True, help='Folder output path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CompareAnnotationRatings(args.baseline_annotations_path, args.warped_annotations_path, args.csm_ratings_path, args.output_folder)
