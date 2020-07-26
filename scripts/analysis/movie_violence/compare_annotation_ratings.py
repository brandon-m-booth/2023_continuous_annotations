import re
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
import agreement_metrics as agree

functionals = [np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nansum]

def CompareAnnotationRatings(warped_annos_folder_path, csm_ratings_path, movie_name_regex='warped_annotation_(.*).csv'):
   # Gather CSM movie titles with available ratings information
   movie_name_regprog = re.compile(movie_name_regex)
   with open(csm_ratings_path, 'rb') as csm_ratings_file:
      movie_csm_data_list = pickle.load(csm_ratings_file)
   titles = [x['title'] for x in movie_csm_data_list]

   # Build dictionary associating movie violence ratings with their respective annotations
   anno_rating_dict = {}
   warped_anno_files = glob.glob(os.path.join(warped_annos_folder_path, '*.csv'))
   for warped_anno_file in warped_anno_files:
      movie_name = movie_name_regprog.match(os.path.basename(warped_anno_file)).group(1)
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
      anno_rating_dict[movie_name] = {'violence': violence_rating, 'annotation': warped_anno_df}
   
   # Apply functionals to the annotations to reduce their dimension for comparison with violence ratings
   func_dict = {}
   for movie_name in anno_rating_dict.keys():
      func_dict[movie_name] = {}
      annotation = anno_rating_dict[movie_name]['annotation'].iloc[:,1]

      # Apply each functional to the annotation per movie
      for functional in functionals:
         func_name = functional.__name__
         func_value = functional(annotation)
         func_dict[movie_name][func_name] = func_value

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
      func_values = []
      for movie_name in movie_names_list:
         func_values.append(func_dict[movie_name][func_name])
      ratings_mat = np.vstack((ratings_mat, func_values))
   ratings_df = pd.DataFrame(data=ratings_mat.T, columns=['CSM']+func_names, index=movie_names_list)

   pearson = pd.Series(agree.PearsonCorr(ratings_df).iloc[0,1:], name='Pearson')
   spearman = pd.Series(agree.SpearmanCorr(ratings_df).iloc[0,1:], name='Spearman')
   tau = pd.Series(agree.KendallTauCorr(ratings_df).iloc[0,1:], name='Kendall\'s Tau')
   corr_df = pd.concat((pearson, spearman, tau), axis=1)
   sns_corr_df = pd.DataFrame(data=np.zeros((corr_df.shape[0]*corr_df.shape[1],3)), columns=['Metric', 'Functional', 'Correlation'])

   sns_idx = 0
   for metric in corr_df.columns:
      for func_name in corr_df.index:
         sns_corr_df.iloc[sns_idx,:] = [metric, func_name, corr_df.loc[func_name, metric]]
         sns_idx += 1

   # TODO - multi bar plot with each metrics grouped and each group having many functionals
   #plt.bar([pearson,spearman,ccc,kappa,alpha,icc])
   ax = sns.barplot(x="Metric", y="Correlation", hue="Functional", data=sns_corr_df)
   #plt.xlabel(['Pearson','Spearman','CCC','Kappa','Alpha','ICC'])
   #plt.ylabel('Correlation, CSM vs. ratings')
   plt.title('Movie Violence Validation')
   plt.show()


   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--warped_annotations_path', required=True, help='Folder containing warped annotations, one file per movie')
   parser.add_argument('--csm_ratings_path', required=True, help='Path to CSM pkl containing movie ratings')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   CompareAnnotationRatings(args.warped_annotations_path, args.csm_ratings_path)
