import re
import os
import sys
import pdb
import glob
import pickle
import argparse
import tikzplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import agreement_metrics as agree
from util import TikzplotlibFixNCols

import warnings
warnings.filterwarnings("ignore") # Reenable if you wish to debug!!

do_plot_debug=False # WILL NOT OUTPUT VALID DATA IF ENABLED

def ComputeSelectedAnnotationAgreements(agreement_filtered_anno_folder, output_folder, agreement_anno_movie_name_regex='(.*)_cut\d+_opt_annotations_.*\.csv', agreement_anno_movie_cut_regex='.*_cut(\d+)_opt_annotations_.*\.csv'):
    
    movie_name_regprog = re.compile(agreement_anno_movie_name_regex)
    movie_cut_regprog = re.compile(agreement_anno_movie_cut_regex)   

    # Get the list of unique movie names and cuts
    movie_names = []
    movie_cuts = []
    agreement_annotations_files = glob.glob(os.path.join(agreement_filtered_anno_folder, "*.csv"))
    for agreement_anno_file in agreement_annotations_files:
        movie_name = movie_name_regprog.match(os.path.basename(agreement_anno_file)).group(1)
        movie_cut = movie_cut_regprog.match(os.path.basename(agreement_anno_file)).group(1)

        movie_names.append(movie_name)
        movie_cuts.append(movie_cut)

    movie_names = sorted(np.unique(movie_names))
    movie_cuts = sorted(np.unique(movie_cuts), key=lambda x: int(x)-1)

    agree_metrics = ['Pearson mean', 'Pearson sd', 'Spearman mean', 'Spearman sd', 'Kendalls Tau mean', 'Kendalls Tau sd', 'CCC mean', 'CCC sd', 'Cronbachs Alpha', 'ICC[1,k]', 'ICC[1,k] lower bound', 'ICC[1,k] upper bound', 'Krippendorffs Alpha', 'SDA mean', 'SDA sd']
    movie_names_and_cuts = [name+'_cut'+cut for name in movie_names for cut in movie_cuts]
    agree_df = pd.DataFrame(data=np.zeros((len(movie_names_and_cuts), len(agree_metrics))), index=movie_names_and_cuts, columns=agree_metrics)

    for movie_name in tqdm(movie_names):
        for movie_cut in movie_cuts:
            movie_name_and_cut = movie_name+'_cut'+movie_cut
            movie_and_cut_agreement_files = glob.glob(os.path.join(agreement_filtered_anno_folder, movie_name_and_cut+'*.csv'))
            if len(movie_and_cut_agreement_files) == 0:
                agree_df.loc[movie_name_and_cut, :] = np.nan
                continue

            annos_df = None
            for movie_and_cut_csv_file in movie_and_cut_agreement_files:
                anno_df = pd.read_csv(movie_and_cut_csv_file)
                if annos_df is None:
                    annos_df = anno_df
                else:
                    annos_df = pd.merge(annos_df, anno_df, how='outer', on='Time(sec)')

            annos_df = annos_df.iloc[:, 1:]

            if do_plot_debug:
                #if movie_name == "Good_Boys" and movie_cut == "3":
                #if movie_name == "The_Possession_of_Hannah_Grace" and movie_cut == "1":
                if movie_name == "The_Hustle" and movie_cut == "1":
                    annos_df.plot()
                    plt.show()
                else:
                    continue

            pearson_corr = agree.PearsonCorr(annos_df)
            upper_tri_idx = np.triu_indices(pearson_corr.shape[0], k=1)
            pearson_mean  = np.nanmean(pearson_corr.values[upper_tri_idx])
            pearson_sd = np.nanstd(pearson_corr.values[upper_tri_idx])

            spearman_corr = agree.SpearmanCorr(annos_df)
            spearman_mean = np.nanmean(spearman_corr.values[upper_tri_idx])
            spearman_sd = np.nanstd(spearman_corr.values[upper_tri_idx])

            tau_corr = agree.KendallTauCorr(annos_df)
            tau_mean = np.nanmean(tau_corr.values[upper_tri_idx])
            tau_sd = np.nanstd(tau_corr.values[upper_tri_idx])

            ccc_corr = agree.ConcordanceCorrelationCoef(annos_df)
            ccc_mean = np.nanmean(ccc_corr.values[upper_tri_idx])
            ccc_sd = np.nanstd(ccc_corr.values[upper_tri_idx])

            icc_mat = agree.ICC(annos_df)
            icc1k = icc_mat.loc[icc_mat['type'] == 'ICC1k', 'ICC'][0]
            icc1k_lower = icc_mat.loc[icc_mat['type'] == 'ICC1k', 'lower bound'][0]
            icc1k_upper = icc_mat.loc[icc_mat['type'] == 'ICC1k', 'upper bound'][0]

            sda_corr = agree.SDAFromRatings(annos_df)
            sda_mean = np.nanmean(sda_corr.values[upper_tri_idx])
            sda_sd = np.nanstd(sda_corr.values[upper_tri_idx])
            
            agree_df.loc[movie_name_and_cut, 'Pearson mean'] = pearson_mean
            agree_df.loc[movie_name_and_cut, 'Pearson sd'] = pearson_sd
            agree_df.loc[movie_name_and_cut, 'Spearman mean'] = spearman_mean
            agree_df.loc[movie_name_and_cut, 'Spearman sd'] = spearman_sd
            agree_df.loc[movie_name_and_cut, 'Kendalls Tau mean'] = tau_mean
            agree_df.loc[movie_name_and_cut, 'Kendalls Tau sd'] = tau_sd
            agree_df.loc[movie_name_and_cut, 'CCC mean'] = ccc_mean
            agree_df.loc[movie_name_and_cut, 'CCC sd'] = ccc_sd
            agree_df.loc[movie_name_and_cut, 'Cronbachs Alpha'] = agree.CronbachsAlphaCorr(annos_df)
            agree_df.loc[movie_name_and_cut, 'ICC[1,k]']= icc1k
            agree_df.loc[movie_name_and_cut, 'ICC[1,k] lower bound']= icc1k_lower
            agree_df.loc[movie_name_and_cut, 'ICC[1,k] upper bound']= icc1k_upper
            agree_df.loc[movie_name_and_cut, 'Krippendorffs Alpha'] = agree.KrippendorffsAlpha(annos_df)
            agree_df.loc[movie_name_and_cut, 'SDA mean'] = sda_mean
            agree_df.loc[movie_name_and_cut, 'SDA sd'] = sda_sd

    agree_df.to_csv(os.path.join(output_folder, "selected_anno_agreements.csv"), index=True, header=True)
    return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--cleaned_selected_aligned_annotations_path', required=True, help='Folder containing cleaned, selected, and aligned annotations, one file per annotation')
   parser.add_argument('--output_folder', required=True, help='Folder output path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)

   ComputeSelectedAnnotationAgreements(args.cleaned_selected_aligned_annotations_path, args.output_folder)
