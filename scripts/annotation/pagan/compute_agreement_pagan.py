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
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
import util
import agreement_metrics as agree

valid_time_percentage = 0.20
dtw_cache_folder_path = '../../../results/dtw_cache'

# Sort labels so 0 is the group with the highest similarity, followed by 1,2,3...
def FixLabels(sim_mat, labels):
   ulabels = sorted(np.unique(labels).tolist())
   avg_sim_per_label = []
   for ulabel in ulabels:
      label_mask = labels == ulabel
      mask_sim_mat = sim_mat.loc[label_mask, label_mask]
      avg_sim =  np.mean(util.GetUpperTri(mask_sim_mat.values))
      avg_sim_per_label.append(avg_sim)
   new_label_order = np.argsort(avg_sim_per_label)[::-1]

   fixed_labels = np.zeros(len(labels)).astype(int)
   for i in range(len(ulabels)):
      fixed_labels[labels == ulabels[i]] = new_label_order[i]

   return fixed_labels

def GetAgglomerativeClusters(sim_mat, k, method='average'):
   ac = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage=method)
   labels = ac.fit(1.0-sim_mat).labels_
   labels = FixLabels(sim_mat, labels)
   return labels

def GetSpectralClusters(sim_mat, k):
   sc = SpectralClustering(n_clusters=k, assign_labels="discretize", affinity='precomputed')
   labels = sc.fit(sim_mat).labels_
   labels = FixLabels(sim_mat, labels)
   return labels

def GetHierarchyClusters(link, k, method='maxclust'):
   clusters = hcluster.fcluster(link, k, method)
   clusters -= 1 # Zero-based indexing
   return clusters

# Outputs the optimal clustering given a collection of cluster labels from different methods
def ComputeOptClusters(cluster_df):
   #opt_cluster_cols = ['SDA Spectral', 'Kendall\'s Tau Spectral']
   #opt_cluster_cols = ['SDA Spectral']
   opt_cluster_cols = ['DTW SDA Spectral']
   opt_df = cluster_df[opt_cluster_cols]
   opt_mask = opt_df.sum(axis=1) < len(opt_cluster_cols)/2.0
   return opt_df.index[opt_mask]
   

def ClusterSimilarityMatrix(sim_mat, method='average'):
   n = len(sim_mat)
   flat_dist_mat = ssd.squareform(1.0-sim_mat)
   res_linkage = hcluster.linkage(flat_dist_mat, method=method)
   res_order = hcluster.leaves_list(res_linkage)
   seriated_sim = np.zeros((n,n))
   a,b = np.triu_indices(n,k=1)
   seriated_sim[a,b] = sim_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
   seriated_sim[b,a] = seriated_sim[a,b]
   for i in range(n):
      seriated_sim[i,i] = sim_mat[i,i]

   return seriated_sim, res_order, res_linkage


def ComputePaganAgreement(input_file_path, output_path, do_show_plots=False):
   input_files = [input_file_path]
   if os.path.isdir(input_file_path):
      input_files = glob.glob(os.path.join(input_file_path, '*.csv'))
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   for input_file in input_files:
      df = pd.read_csv(input_file)
      unique_experiment_names = np.unique(df['OriginalName'])
      for project_entry_name in unique_experiment_names:
         print('Processing '+project_entry_name)
         proj_df = df.loc[df['OriginalName'] == project_entry_name, :]
         pids = np.unique(proj_df['Participant'])
         external_pids = []

         if do_show_plots:
            fig_anno, axs_anno = plt.subplots(1,1, figsize=(11,9))
            fig_agree, axs_agree = plt.subplots(1,1, figsize=(11,9), tight_layout=True)
            fig_corr, axs_corr = plt.subplots(4,5, figsize=(11,9))
            fig_agg, axs_agg = plt.subplots(4,5, figsize=(11,9))
            fig_spec, axs_spec= plt.subplots(4,5, figsize=(11,9))
            fig_opt, axs_opt = plt.subplots(1,1, figsize=(11,9))

         end_time = max(proj_df['VideoTime'])
         combined_anno_df = None
         for pid in pids:
            pid_df = proj_df.loc[proj_df['Participant']==pid,:]
            pid_df = pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable sort alg.
            external_pid = str(pid_df['ExternalPID'].iloc[0])
            external_pids.append(external_pid)
            anno_time = pid_df.loc[:,'VideoTime'].values
            anno_vals = pid_df.loc[:,'Value'].values
            anno_df = pd.DataFrame(data=anno_vals, index=anno_time, columns=[external_pid])
            if do_show_plots:
               axs_anno.plot(anno_time, anno_vals)

            if not max(anno_time) >= valid_time_percentage*end_time:
               print('Ignoring annotation '+external_pid+' because end time is: '+str(max(anno_time))+'/'+str(end_time))
               continue
            #if len(np.unique(anno_vals)) == 1:
            #   print('Ignoring annotation '+external_pid+' because it has no variability')
            #   continue

            # Replace NaN values with inf values temporarily. NaNs in the data indicate missing values
            # and interfere with the interpolation later
            anno_df = anno_df.replace(to_replace=np.nan, value=np.inf)
            if combined_anno_df is None:
               combined_anno_df = anno_df
            else:
               combined_anno_df = pd.merge(combined_anno_df, anno_df, how='outer', left_index=True, right_index=True, sort=False)
         combined_anno_df = combined_anno_df.sort_index()
         combined_anno_df = combined_anno_df.interpolate(method='linear', axis=0)
         combined_anno_df = combined_anno_df.groupby(level=0).mean()
         
         # Undo the temporary replacement of NaN values with Inf
         combined_anno_df = combined_anno_df.replace(to_replace=np.inf, value=np.nan)

         # Compute time-warped variant of the combined annotations.  We resample the annotations every 100ms
         # so a maximum time window can be applied by DTW
         combined_anno_dtw_df = combined_anno_df.copy()
         combined_anno_dtw_df.index = [datetime.fromtimestamp(t/1000.0, pytz.utc) for t in combined_anno_dtw_df.index]
         resampled_dtw_df = combined_anno_dtw_df.resample('100ms').mean().interpolate()
         resampled_dtw_df.index = [t.timestamp() for t in resampled_dtw_df.index]
         dtw_cache_file = os.path.join(dtw_cache_folder_path, project_entry_name+'_dtw_cache.pkl')
         if os.path.isfile(dtw_cache_file):
            combined_anno_dtw_df = pickle.load(open(dtw_cache_file, 'rb'))
         else:
            combined_anno_dtw_df = agree.DTWPairwise(resampled_dtw_df, max_warp_distance=5*10) # 5-second max
            if not os.path.isdir(os.path.dirname(dtw_cache_file)):
               os.makedirs(os.path.dirname(dtw_cache_file))
            pickle.dump(combined_anno_dtw_df, open(dtw_cache_file, 'wb'))

         ##################
         #### Compute different pairwise agreement measures
         ##################

         ### Value-based metrics ###
         # Pearson
         pearson_corr_mat = agree.PearsonCorr(combined_anno_df)
         pearson_dtw_corr_mat = agree.DTWPairwiseAgreement(combined_anno_dtw_df, combined_anno_df.columns, agree.PearsonCorr)

         # Spearman
         spearman_corr_mat = agree.SpearmanCorr(combined_anno_df)
         spearman_dtw_corr_mat = agree.DTWPairwiseAgreement(combined_anno_dtw_df, combined_anno_df.columns, agree.SpearmanCorr)

         # Kendall's Tau
         kendall_corr_mat = agree.KendallTauCorr(combined_anno_df)
         kendall_dtw_corr_mat = agree.DTWPairwiseAgreement(combined_anno_dtw_df, combined_anno_df.columns, agree.KendallTauCorr)

         # MSE
         mse_mat = agree.MeanSquaredErrorMat(combined_anno_df)
         mse_dtw_mat = agree.DTWPairwiseAgreement(combined_anno_dtw_df, combined_anno_df.columns, agree.MeanSquaredErrorMat)

         # CCC
         ccc_corr_mat = agree.ConcordanceCorrelationCoef(combined_anno_df)
         ccc_dtw_corr_mat = agree.DTWPairwiseAgreement(combined_anno_dtw_df, combined_anno_df.columns, agree.ConcordanceCorrelationCoef)

         ###
         ### Derivative-based methods ###
         norm_diff_df = util.NormedDiff(combined_anno_df)
         abs_norm_diff_df = norm_diff_df.abs()
         accum_norm_diff_df = util.AccumNormedDiff(combined_anno_df)
         norm_diff_dtw_df = util.NormedDiff(combined_anno_dtw_df)
         abs_norm_diff_dtw_df = norm_diff_dtw_df.abs()
         accum_norm_diff_dtw_df = util.AccumNormedDiff(combined_anno_dtw_df)

         # Cohen's Kappa normed diff
         cohens_kappa_norm_diff_corr_mat = agree.CohensKappaCorr(norm_diff_df, labels=[-1,0,1])
         cohens_kappa_norm_diff_dtw_corr_mat = agree.DTWPairwiseAgreement(norm_diff_dtw_df, combined_anno_df.columns, agree.CohensKappaCorr, agreement_func_args={'labels': [-1,0,1]})

         # Cohen's Kappa abs normed diff
         cohens_kappa_abs_norm_diff_corr_mat = agree.CohensKappaCorr(abs_norm_diff_df, labels=[0,1])
         cohens_kappa_abs_norm_diff_dtw_corr_mat = agree.DTWPairwiseAgreement(abs_norm_diff_dtw_df, combined_anno_df.columns, agree.CohensKappaCorr, agreement_func_args={'labels':[0,1]})

         # SDA
         sda_mat = agree.SDACorrected(norm_diff_df)
         #norm_sum_delta_mat = agree.NormedSumDelta(norm_diff_df)
         sda_dtw_mat = agree.DTWPairwiseAgreement(norm_diff_dtw_df, combined_anno_df.columns, agree.SDACorrected)
         #norm_sum_delta_dtw_mat = agree.DTWPairwiseAgreement(norm_diff_dtw_df, combined_anno_df.columns, agree.NormedSumDelta)

         # TSS absolute diff, T>=N-1
         #abs_norm_sum_delta_mat = agree.NormedSumDelta(abs_norm_diff_df)
         abs_sda_mat = agree.SDACorrected(abs_norm_diff_df)
         #abs_norm_sum_delta_dtw_mat = agree.DTWPairwiseAgreement(abs_norm_diff_dtw_df, combined_anno_df.columns, agree.NormedSumDelta)
         abs_sda_dtw_mat = agree.DTWPairwiseAgreement(abs_norm_diff_dtw_df, combined_anno_df.columns, agree.SDACorrected)

         ###############
         # Compute summary agreement measures
         ###############
         # Cronbach's alpha
         cronbachs_alpha = agree.CronbachsAlphaCorr(combined_anno_df)
         
         # Cronbach's alpha normed diff
         cronbachs_alpha_norm_diff = agree.CronbachsAlphaCorr(norm_diff_df)

         # Cronbach's alpha abs normed diff
         cronbachs_alpha_abs_norm_diff = agree.CronbachsAlphaCorr(abs_norm_diff_df)

         # Cronbach's alpha abs normed diff
         cronbachs_alpha_accum_norm_diff = agree.CronbachsAlphaCorr(accum_norm_diff_df)

         # ICC(2,1)
         #icc_df = agree.ICC(combined_anno_df)
         #icc21_df = icc_df.loc[icc_df['type'] == 'ICC2',:]
         #icc21 = icc21_df['ICC'].iloc[0]
         icc21 = 0.5

         # SAGR (signed agreement)
         # BB - Doesn't make sense for scales where zero isn't the center

         # Krippendorff's alpha
         #krippendorffs_alpha = agree.KrippendorffsAlpha(combined_anno_df)

         # Krippendorff's alpha of normed diff
         #krippendorffs_alpha_norm_diff = agree.KrippendorffsAlpha(norm_diff_df)

         # Krippendorff's alpha of abs normed diff
         #krippendorffs_alpha_abs_norm_diff = agree.KrippendorffsAlpha(abs_norm_diff_df)

         # Accumulated Normed Rank-based Krippendorff's Alpha
         #krippendorffs_alpha_accum_norm_diff = agree.KrippendorffsAlpha(accum_norm_diff_df)
         krippendorffs_alpha = 1.0
         krippendorffs_alpha_norm_diff = 1.0
         krippendorffs_alpha_abs_norm_diff = 1.0
         krippendorffs_alpha_accum_norm_diff = 1.0

         ###############

         # Put global agreement measures into a dataframe
         global_agreement_df = pd.DataFrame(data=[[icc21, cronbachs_alpha, cronbachs_alpha_norm_diff, cronbachs_alpha_abs_norm_diff, cronbachs_alpha_accum_norm_diff, krippendorffs_alpha, krippendorffs_alpha_norm_diff, krippendorffs_alpha_abs_norm_diff, krippendorffs_alpha_accum_norm_diff]], columns=['ICC(2)', 'Cronbach\'s Alpha', 'Cronbach\'s Alpha Norm Diff', 'Cronbach\'s Alpha Abs Norm Diff', 'Cronbach\'s Alpha Accum Norm Diff', 'Krippendorff\'s Alpha', 'Krippendorff\'s Alpha Norm Diff', 'Krippendorff\'s Alpha Abs Norm Diff', 'Krippendorff\'s Alpha Accum Norm Diff'])

         # Max-normalize the MSE and convert to a correlation-like matrix
         mse_corr_mat = 1.0 - mse_mat/np.max(mse_mat.values)
         np.fill_diagonal(mse_corr_mat.values, 1)
         mse_dtw_corr_mat = 1.0 - mse_dtw_mat/np.max(mse_dtw_mat.values)
         np.fill_diagonal(mse_dtw_corr_mat.values, 1)

         # Force symmetry for corr matrices and normalize into [0,1] range
         pearson_corr_mat[pd.isna(pearson_corr_mat)] = 0
         np.fill_diagonal(pearson_corr_mat.values, 1)
         pearson_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         pearson_norm_corr_mat = 0.5*pearson_corr_mat+ 0.5
         #pearson_norm_corr_mat = pearson_corr_mat.abs()
         pearson_dtw_corr_mat[pd.isna(pearson_dtw_corr_mat)] = 0
         np.fill_diagonal(pearson_dtw_corr_mat.values, 1)
         pearson_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         pearson_norm_dtw_corr_mat = 0.5*pearson_dtw_corr_mat + 0.5
         #pearson_norm_dtw_corr_mat = pearson_dtw_corr_mat.abs()

         spearman_corr_mat[pd.isna(spearman_corr_mat)] = 0
         np.fill_diagonal(spearman_corr_mat.values, 1)
         spearman_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         spearman_norm_corr_mat = 0.5*spearman_corr_mat+ 0.5
         #spearman_norm_corr_mat = spearman_corr_mat.abs()
         spearman_dtw_corr_mat[pd.isna(spearman_dtw_corr_mat)] = 0
         np.fill_diagonal(spearman_dtw_corr_mat.values, 1)
         spearman_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         spearman_norm_dtw_corr_mat = 0.5*spearman_dtw_corr_mat + 0.5
         #spearman_norm_dtw_corr_mat = spearman_dtw_corr_mat.abs()

         kendall_corr_mat[pd.isna(kendall_corr_mat)] = 0
         np.fill_diagonal(kendall_corr_mat.values, 1)
         kendall_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         kendall_norm_corr_mat = 0.5*kendall_corr_mat+ 0.5
         #kendall_norm_corr_mat = kendall_corr_mat.abs()
         kendall_dtw_corr_mat[pd.isna(kendall_dtw_corr_mat)] = 0
         np.fill_diagonal(kendall_dtw_corr_mat.values, 1)
         kendall_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         kendall_norm_dtw_corr_mat = 0.5*kendall_dtw_corr_mat + 0.5
         #kendall_norm_dtw_corr_mat = kendall_dtw_corr_mat.abs()

         ccc_corr_mat[pd.isna(ccc_corr_mat)] = 0
         np.fill_diagonal(ccc_corr_mat.values, 1)
         ccc_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         ccc_norm_corr_mat = 0.5*ccc_corr_mat + 0.5
         #ccc_norm_corr_mat = ccc_corr_mat.abs()
         ccc_dtw_corr_mat[pd.isna(ccc_dtw_corr_mat)] = 0
         np.fill_diagonal(ccc_dtw_corr_mat.values, 1)
         ccc_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         ccc_dtw_norm_corr_mat = 0.5*ccc_dtw_corr_mat + 0.5
         #ccc_dtw_norm_corr_mat = ccc_dtw_corr_mat.abs()

         cohens_kappa_norm_diff_corr_mat[pd.isna(cohens_kappa_norm_diff_corr_mat)] = 0
         np.fill_diagonal(cohens_kappa_norm_diff_corr_mat.values, 1)
         cohens_kappa_norm_diff_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         cohens_kappa_norm_diff_corr_mat = 0.5*cohens_kappa_norm_diff_corr_mat + 0.5
         cohens_kappa_norm_diff_dtw_corr_mat[pd.isna(cohens_kappa_norm_diff_dtw_corr_mat)] = 0
         np.fill_diagonal(cohens_kappa_norm_diff_dtw_corr_mat.values, 1)
         cohens_kappa_norm_diff_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         cohens_kappa_norm_diff_dtw_corr_mat = 0.5*cohens_kappa_norm_diff_dtw_corr_mat + 0.5

         cohens_kappa_abs_norm_diff_corr_mat[pd.isna(cohens_kappa_abs_norm_diff_corr_mat)] = 0
         np.fill_diagonal(cohens_kappa_abs_norm_diff_corr_mat.values, 1)
         cohens_kappa_abs_norm_diff_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         cohens_kappa_abs_norm_diff_corr_mat = 0.5*cohens_kappa_abs_norm_diff_corr_mat + 0.5
         cohens_kappa_abs_norm_diff_dtw_corr_mat[pd.isna(cohens_kappa_abs_norm_diff_dtw_corr_mat)] = 0
         np.fill_diagonal(cohens_kappa_abs_norm_diff_dtw_corr_mat.values, 1)
         cohens_kappa_abs_norm_diff_dtw_corr_mat.clip(lower=-1.0, upper=1.0, inplace=True)
         cohens_kappa_abs_norm_diff_dtw_corr_mat = 0.5*cohens_kappa_abs_norm_diff_dtw_corr_mat + 0.5

         np.fill_diagonal(sda_mat.values, 1)
         sda_norm_mat = 0.5*sda_mat + 0.5
         np.fill_diagonal(sda_dtw_mat.values, 1)
         sda_norm_dtw_mat = 0.5*sda_dtw_mat + 0.5
         np.fill_diagonal(abs_sda_mat.values, 1)
         abs_sda_norm_mat = 0.5*abs_sda_mat + 0.5
         np.fill_diagonal(abs_sda_dtw_mat.values, 1)
         abs_sda_norm_dtw_mat = 0.5*abs_sda_dtw_mat + 0.5

         # Print correlation statistics
         pearson_tri = util.GetUpperTri(pearson_corr_mat.values)
         spearman_tri = util.GetUpperTri(spearman_corr_mat.values)
         kendall_tau_tri = util.GetUpperTri(kendall_corr_mat.values)
         ccc_tri = util.GetUpperTri(ccc_corr_mat.values)
         mse_tri = util.GetUpperTri(mse_mat.values)
         sda_tri = util.GetUpperTri(sda_mat.values)
         combined_tri = np.vstack((pearson_tri, spearman_tri, kendall_tau_tri, ccc_tri, mse_tri, sda_tri)).T
         corr_tri_df = pd.DataFrame(data=combined_tri, columns=['Pearson', 'Spearman', 'Kendall\'s Tau', 'CCC', 'MSE', 'SDA'])

         # Agglomerative clustering
         #(agg_pearson_sim, pearson_cluster_order_idx, pearson_agg_link) = ClusterSimilarityMatrix(pearson_norm_corr_mat.values, method='centroid')
         #agg_pearson_sim = pd.DataFrame(data=agg_pearson_sim, index=pearson_norm_corr_mat.columns[pearson_cluster_order_idx], columns=pearson_norm_corr_mat.columns[pearson_cluster_order_idx])
         #(agg_spearman_sim, spearman_cluster_order_idx, spearman_agg_link) = ClusterSimilarityMatrix(spearman_norm_corr_mat.values, method='centroid')
         #agg_spearman_sim = pd.DataFrame(data=agg_spearman_sim, index=spearman_norm_corr_mat.columns[spearman_cluster_order_idx], columns=spearman_norm_corr_mat.columns[spearman_cluster_order_idx])
         #(agg_kendall_sim, kendall_cluster_order_idx, kendall_agg_link) = ClusterSimilarityMatrix(kendall_norm_corr_mat.values, method='centroid')
         #agg_kendall_sim = pd.DataFrame(data=agg_kendall_sim, index=kendall_norm_corr_mat.columns[kendall_cluster_order_idx], columns=kendall_norm_corr_mat.columns[kendall_cluster_order_idx])
         #(agg_mse_sim, mse_cluster_order_idx, mse_agg_link) = ClusterSimilarityMatrix(mse_corr_mat.values, method='centroid')
         #agg_mse_sim = pd.DataFrame(data=agg_mse_sim, index=mse_corr_mat.columns[mse_cluster_order_idx], columns=mse_corr_mat.columns[mse_cluster_order_idx])
         #(agg_ccc_sim, ccc_cluster_order_idx, ccc_agg_link) = ClusterSimilarityMatrix(ccc_norm_corr_mat.values, method='centroid')
         #agg_ccc_sim = pd.DataFrame(data=agg_ccc_sim, index=ccc_norm_corr_mat.columns[ccc_cluster_order_idx], columns=ccc_norm_corr_mat.columns[ccc_cluster_order_idx])
         #(agg_cohens_kappa_norm_diff_sim, cohens_kappa_norm_diff_cluster_order_idx, cohens_kappa_norm_diff_agg_link) = ClusterSimilarityMatrix(cohens_kappa_norm_diff_corr_mat.values, method='centroid')
         #agg_cohens_kappa_norm_diff_sim = pd.DataFrame(data=agg_cohens_kappa_norm_diff_sim, index=cohens_kappa_norm_diff_corr_mat.columns[cohens_kappa_norm_diff_cluster_order_idx], columns=cohens_kappa_norm_diff_corr_mat.columns[cohens_kappa_norm_diff_cluster_order_idx])
         #(agg_cohens_kappa_abs_norm_diff_sim, cohens_kappa_abs_norm_diff_cluster_order_idx, cohens_kappa_abs_norm_diff_agg_link) = ClusterSimilarityMatrix(cohens_kappa_abs_norm_diff_corr_mat.values, method='centroid')
         #agg_cohens_kappa_abs_norm_diff_sim = pd.DataFrame(data=agg_cohens_kappa_abs_norm_diff_sim, index=cohens_kappa_abs_norm_diff_corr_mat.columns[cohens_kappa_abs_norm_diff_cluster_order_idx], columns=cohens_kappa_abs_norm_diff_corr_mat.columns[cohens_kappa_abs_norm_diff_cluster_order_idx])
         #(agg_sda_sim, sda_cluster_order_idx, sda_agg_link) = ClusterSimilarityMatrix(sda_norm_mat.values, method='centroid')
         #agg_sda_sim = pd.DataFrame(data=agg_sda_sim, index=sda_norm_mat.columns[sda_cluster_order_idx], columns=sda_norm_mat.columns[sda_cluster_order_idx])
         #(agg_abs_sda_sim, abs_sda_cluster_order_idx, abs_sda_agg_link) = ClusterSimilarityMatrix(abs_sda_norm_mat.values, method='centroid')
         #agg_abs_sda_sim = pd.DataFrame(data=agg_abs_sda_sim, index=abs_sda_norm_mat.columns[abs_sda_cluster_order_idx], columns=abs_sda_norm_mat.columns[abs_sda_cluster_order_idx])
         #(agg_pearson_dtw_sim, pearson_cluster_order_dtw_idx, pearson_agg_dtw_link) = ClusterSimilarityMatrix(pearson_norm_dtw_corr_mat.values, method='centroid')
         #agg_pearson_dtw_sim = pd.DataFrame(data=agg_pearson_dtw_sim, index=pearson_norm_dtw_corr_mat.columns[pearson_cluster_order_dtw_idx], columns=pearson_norm_dtw_corr_mat.columns[pearson_cluster_order_dtw_idx])
         #(agg_spearman_dtw_sim, spearman_cluster_order_dtw_idx, spearman_agg_dtw_link) = ClusterSimilarityMatrix(spearman_norm_dtw_corr_mat.values, method='centroid')
         #agg_spearman_dtw_sim = pd.DataFrame(data=agg_spearman_dtw_sim, index=spearman_norm_dtw_corr_mat.columns[spearman_cluster_order_dtw_idx], columns=spearman_norm_dtw_corr_mat.columns[spearman_cluster_order_dtw_idx])
         #(agg_kendall_dtw_sim, kendall_cluster_order_dtw_idx, kendall_agg_dtw_link) = ClusterSimilarityMatrix(kendall_norm_dtw_corr_mat.values, method='centroid')
         #agg_kendall_dtw_sim = pd.DataFrame(data=agg_kendall_dtw_sim, index=kendall_norm_dtw_corr_mat.columns[kendall_cluster_order_dtw_idx], columns=kendall_norm_dtw_corr_mat.columns[kendall_cluster_order_dtw_idx])
         #(agg_mse_dtw_sim, mse_cluster_order_dtw_idx, mse_agg_dtw_link) = ClusterSimilarityMatrix(mse_dtw_corr_mat.values, method='centroid')
         #agg_mse_dtw_sim = pd.DataFrame(data=agg_mse_dtw_sim, index=mse_dtw_corr_mat.columns[mse_cluster_order_dtw_idx], columns=mse_dtw_corr_mat.columns[mse_cluster_order_dtw_idx])
         #(agg_ccc_dtw_sim, ccc_cluster_order_dtw_idx, ccc_agg_dtw_link) = ClusterSimilarityMatrix(ccc_dtw_norm_corr_mat.values, method='centroid')
         #agg_ccc_dtw_sim = pd.DataFrame(data=agg_ccc_dtw_sim, index=ccc_dtw_norm_corr_mat.columns[ccc_cluster_order_dtw_idx], columns=ccc_dtw_norm_corr_mat.columns[ccc_cluster_order_dtw_idx])
         #(agg_cohens_kappa_norm_diff_dtw_sim, cohens_kappa_norm_diff_cluster_order_dtw_idx, cohens_kappa_norm_diff_agg_dtw_link) = ClusterSimilarityMatrix(cohens_kappa_norm_diff_dtw_corr_mat.values, method='centroid')
         #agg_cohens_kappa_norm_diff_dtw_sim = pd.DataFrame(data=agg_cohens_kappa_norm_diff_dtw_sim, index=cohens_kappa_norm_diff_dtw_corr_mat.columns[cohens_kappa_norm_diff_cluster_order_dtw_idx], columns=cohens_kappa_norm_diff_dtw_corr_mat.columns[cohens_kappa_norm_diff_cluster_order_dtw_idx])
         #(agg_cohens_kappa_abs_norm_diff_dtw_sim, cohens_kappa_abs_norm_diff_cluster_order_dtw_idx, cohens_kappa_abs_norm_diff_agg_dtw_link) = ClusterSimilarityMatrix(cohens_kappa_abs_norm_diff_dtw_corr_mat.values, method='centroid')
         #agg_cohens_kappa_abs_norm_diff_dtw_sim = pd.DataFrame(data=agg_cohens_kappa_abs_norm_diff_dtw_sim, index=cohens_kappa_abs_norm_diff_dtw_corr_mat.columns[cohens_kappa_abs_norm_diff_cluster_order_dtw_idx], columns=cohens_kappa_abs_norm_diff_dtw_corr_mat.columns[cohens_kappa_abs_norm_diff_cluster_order_dtw_idx])
         #(agg_sda_dtw_sim, sda_cluster_order_dtw_idx, sda_agg_dtw_link) = ClusterSimilarityMatrix(sda_norm_dtw_mat.values, method='centroid')
         #agg_sda_dtw_sim = pd.DataFrame(data=agg_sda_dtw_sim, index=sda_norm_dtw_mat.columns[sda_cluster_order_dtw_idx], columns=sda_norm_dtw_mat.columns[sda_cluster_order_dtw_idx])
         #(agg_abs_sda_dtw_sim, abs_sda_cluster_order_dtw_idx, abs_sda_agg_dtw_link) = ClusterSimilarityMatrix(abs_sda_norm_dtw_mat.values, method='centroid')
         #agg_abs_sda_dtw_sim = pd.DataFrame(data=agg_abs_sda_dtw_sim, index=abs_sda_norm_dtw_mat.columns[abs_sda_cluster_order_dtw_idx], columns=abs_sda_norm_dtw_mat.columns[abs_sda_cluster_order_dtw_idx])
         agg_pearson_labels = GetAgglomerativeClusters(pearson_norm_corr_mat, k=2)
         agg_spearman_labels = GetAgglomerativeClusters(spearman_norm_corr_mat, k=2)
         agg_kendall_labels = GetAgglomerativeClusters(kendall_norm_corr_mat, k=2)
         agg_mse_labels = GetAgglomerativeClusters(mse_corr_mat, k=2)
         agg_ccc_labels = GetAgglomerativeClusters(ccc_norm_corr_mat, k=2)
         agg_cohens_kappa_norm_diff_labels = GetAgglomerativeClusters(cohens_kappa_norm_diff_corr_mat, k=2)
         agg_cohens_kappa_abs_norm_diff_labels = GetAgglomerativeClusters(cohens_kappa_abs_norm_diff_corr_mat, k=2)
         agg_sda_labels = GetAgglomerativeClusters(sda_norm_mat, k=2)
         agg_abs_sda_labels = GetAgglomerativeClusters(abs_sda_norm_mat, k=2)
         agg_pearson_dtw_labels = GetAgglomerativeClusters(pearson_norm_dtw_corr_mat, k=2)
         agg_spearman_dtw_labels = GetAgglomerativeClusters(spearman_norm_dtw_corr_mat, k=2)
         agg_kendall_dtw_labels = GetAgglomerativeClusters(kendall_norm_dtw_corr_mat, k=2)
         agg_mse_dtw_labels = GetAgglomerativeClusters(mse_dtw_corr_mat, k=2)
         agg_ccc_dtw_labels = GetAgglomerativeClusters(ccc_dtw_norm_corr_mat, k=2)
         agg_cohens_kappa_norm_diff_dtw_labels = GetAgglomerativeClusters(cohens_kappa_norm_diff_dtw_corr_mat, k=2)
         agg_cohens_kappa_abs_norm_diff_dtw_labels = GetAgglomerativeClusters(cohens_kappa_abs_norm_diff_dtw_corr_mat, k=2)
         agg_sda_dtw_labels = GetAgglomerativeClusters(sda_norm_dtw_mat, k=2)
         agg_abs_sda_dtw_labels = GetAgglomerativeClusters(abs_sda_norm_dtw_mat, k=2)


         # Spectral clustering
         spec_pearson_labels = GetSpectralClusters(pearson_norm_corr_mat, k=2)
         spec_spearman_labels = GetSpectralClusters(spearman_norm_corr_mat, k=2)
         spec_kendall_labels = GetSpectralClusters(kendall_norm_corr_mat, k=2)
         spec_mse_labels = GetSpectralClusters(mse_corr_mat, k=2)
         spec_ccc_labels = GetSpectralClusters(ccc_norm_corr_mat, k=2)
         spec_cohens_kappa_norm_diff_labels = GetSpectralClusters(cohens_kappa_norm_diff_corr_mat, k=2)
         spec_cohens_kappa_abs_norm_diff_labels = GetSpectralClusters(cohens_kappa_abs_norm_diff_corr_mat, k=2)
         spec_sda_labels = GetSpectralClusters(sda_norm_mat, k=2)
         spec_abs_sda_labels = GetSpectralClusters(abs_sda_norm_mat, k=2)
         spec_pearson_dtw_labels = GetSpectralClusters(pearson_norm_dtw_corr_mat, k=2)
         spec_spearman_dtw_labels = GetSpectralClusters(spearman_norm_dtw_corr_mat, k=2)
         spec_kendall_dtw_labels = GetSpectralClusters(kendall_norm_dtw_corr_mat, k=2)
         spec_mse_dtw_labels = GetSpectralClusters(mse_dtw_corr_mat, k=2)
         spec_ccc_dtw_labels = GetSpectralClusters(ccc_dtw_norm_corr_mat, k=2)
         spec_cohens_kappa_norm_diff_dtw_labels = GetSpectralClusters(cohens_kappa_norm_diff_dtw_corr_mat, k=2)
         spec_cohens_kappa_abs_norm_diff_dtw_labels = GetSpectralClusters(cohens_kappa_abs_norm_diff_dtw_corr_mat, k=2)
         spec_sda_dtw_labels = GetSpectralClusters(sda_norm_dtw_mat, k=2)
         spec_abs_sda_dtw_labels = GetSpectralClusters(abs_sda_norm_dtw_mat, k=2)

         # Output binary clustering results
         agreement_strs = ['Pearson', 'Spearman', 'Kendall\'s Tau', 'CCC', 'MSE', 'Cohen\'s Kappa Norm Diff', 'Cohen\'s Kappa Abs Norm Diff', 'SDA', 'ASDA']
         cluster_methods = []
         for method in ['Agglomerative', 'Spectral']:
            for i in range(len(agreement_strs)):
               cluster_methods.append(agreement_strs[i] + ' ' + method)
         cluster_labels_mat = np.vstack((agg_pearson_labels, agg_spearman_labels, agg_kendall_labels, agg_ccc_labels, agg_mse_labels, agg_cohens_kappa_norm_diff_labels, agg_cohens_kappa_abs_norm_diff_labels, agg_sda_labels, agg_abs_sda_labels, spec_pearson_labels, spec_spearman_labels, spec_kendall_labels, spec_ccc_labels, spec_mse_labels, spec_cohens_kappa_norm_diff_labels, spec_cohens_kappa_abs_norm_diff_labels, spec_sda_labels, spec_abs_sda_labels)).T
         cluster_df = pd.DataFrame(data=cluster_labels_mat, index=combined_anno_df.columns, columns=cluster_methods)
         cluster_labels_dtw_mat = np.vstack((agg_pearson_dtw_labels, agg_spearman_dtw_labels, agg_kendall_dtw_labels, agg_ccc_dtw_labels, agg_mse_dtw_labels, agg_cohens_kappa_norm_diff_dtw_labels, agg_cohens_kappa_abs_norm_diff_dtw_labels, agg_sda_dtw_labels, agg_abs_sda_dtw_labels, spec_pearson_dtw_labels, spec_spearman_dtw_labels, spec_kendall_dtw_labels, spec_ccc_dtw_labels, spec_mse_labels, spec_cohens_kappa_norm_diff_dtw_labels, spec_cohens_kappa_abs_norm_diff_dtw_labels, spec_sda_dtw_labels, spec_abs_sda_dtw_labels)).T
         cluster_dtw_df = pd.DataFrame(data=cluster_labels_dtw_mat, index=combined_anno_df.columns, columns=['DTW '+x for x in cluster_methods])
         cluster_df = pd.concat((cluster_df, cluster_dtw_df), axis=1)

         output_file_path = os.path.join(output_path, project_entry_name+'_clusters.csv')
         cluster_df.to_csv(output_file_path, index=True, header=True)

         # Compute meta cluster
         opt_pids = ComputeOptClusters(cluster_df)
         opt_df = combined_anno_df.loc[:,opt_pids]
         opt_output_file_path = os.path.join(output_path, project_entry_name+'_opt_annotations.csv')
         opt_df.to_csv(opt_output_file_path, index=True, header=True)

         if do_show_plots:
            for opt_pid in opt_pids:
               axs_opt.plot(combined_anno_df.index, combined_anno_df[opt_pid])

            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Pairwise correlation and error matrices
            sns.heatmap(pearson_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,0])
            sns.heatmap(spearman_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,1])
            sns.heatmap(kendall_norm_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,2])
            sns.heatmap(mse_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,3])
            sns.heatmap(ccc_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[0,4])
            sns.heatmap(cohens_kappa_norm_diff_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,1])
            sns.heatmap(sda_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,2])
            sns.heatmap(abs_sda_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[1,3])
            sns.heatmap(pearson_norm_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[2,0])
            sns.heatmap(spearman_norm_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[2,1])
            sns.heatmap(kendall_norm_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[2,2])
            sns.heatmap(mse_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[2,3])
            sns.heatmap(ccc_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[2,4])
            sns.heatmap(cohens_kappa_norm_diff_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[3,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_dtw_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[3,1])
            sns.heatmap(sda_dtw_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[3,2])
            sns.heatmap(abs_sda_dtw_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_corr[3,3])

            # Summary agreement bar graph
            sns.barplot(data=global_agreement_df, ax=axs_agree)
            #for tick in axs_agree.xaxis.get_major_ticks()[1::2]:
            #   tick.set_pad(15)
            for tick in axs_agree.xaxis.get_major_ticks():
               tick.label.set_rotation(90)

            # Agg clustered pairwise correlation
            pearson_corr_mat_agg = pearson_norm_corr_mat.copy()
            spearman_corr_mat_agg = spearman_norm_corr_mat.copy()
            kendall_corr_mat_agg = kendall_norm_corr_mat.copy()
            ccc_corr_mat_agg = ccc_norm_corr_mat.copy()
            mse_corr_mat_agg = mse_corr_mat.copy()
            cohens_kappa_norm_diff_corr_mat_agg = cohens_kappa_norm_diff_corr_mat.copy()
            cohens_kappa_abs_norm_diff_corr_mat_agg = cohens_kappa_abs_norm_diff_corr_mat.copy()
            sda_mat_agg = sda_norm_mat.copy()
            abs_sda_mat_agg = abs_sda_norm_mat.copy()
            pearson_dtw_corr_mat_agg = pearson_norm_dtw_corr_mat .copy()
            spearman_dtw_corr_mat_agg = spearman_norm_dtw_corr_mat.copy()
            kendall_dtw_corr_mat_agg = kendall_norm_dtw_corr_mat.copy()
            ccc_dtw_corr_mat_agg = ccc_dtw_norm_corr_mat.copy()
            mse_dtw_corr_mat_agg = mse_dtw_corr_mat.copy()
            cohens_kappa_norm_diff_dtw_corr_mat_agg = cohens_kappa_norm_diff_dtw_corr_mat.copy()
            cohens_kappa_abs_norm_diff_dtw_corr_mat_agg = cohens_kappa_abs_norm_diff_dtw_corr_mat.copy()
            sda_mat_dtw_agg = sda_norm_dtw_mat.copy()
            abs_sda_mat_dtw_agg = abs_sda_norm_dtw_mat.copy()
            agg_corr_mats = [(pearson_corr_mat_agg, agg_pearson_labels), (spearman_corr_mat_agg, agg_spearman_labels), (kendall_corr_mat_agg, agg_kendall_labels), (ccc_corr_mat_agg, agg_ccc_labels), (mse_corr_mat_agg, agg_mse_labels), (cohens_kappa_norm_diff_corr_mat_agg, agg_cohens_kappa_norm_diff_labels), (cohens_kappa_abs_norm_diff_corr_mat_agg, agg_cohens_kappa_abs_norm_diff_labels), (sda_mat_agg, agg_sda_labels), (abs_sda_mat_agg, agg_abs_sda_labels)]
            agg_corr_mats.extend([(pearson_dtw_corr_mat_agg, agg_pearson_dtw_labels), (spearman_dtw_corr_mat_agg, agg_spearman_dtw_labels), (kendall_dtw_corr_mat_agg, agg_kendall_dtw_labels), (ccc_dtw_corr_mat_agg, agg_ccc_dtw_labels), (mse_dtw_corr_mat_agg, agg_mse_dtw_labels), (cohens_kappa_norm_diff_dtw_corr_mat_agg, agg_cohens_kappa_norm_diff_dtw_labels), (cohens_kappa_abs_norm_diff_dtw_corr_mat_agg, agg_cohens_kappa_abs_norm_diff_dtw_labels), (sda_mat_dtw_agg, agg_sda_dtw_labels), (abs_sda_mat_dtw_agg, agg_abs_sda_dtw_labels)])
            for agg_corr_mat, agg_labels in agg_corr_mats: # Recolor the two clusters
               for i in range(len(agg_corr_mat)):
                  if agg_labels[i] == 1:
                     agg_corr_mat.iloc[i,:] = -np.abs(agg_corr_mat.iloc[i,:])
                     agg_corr_mat.iloc[:,i] = -np.abs(agg_corr_mat.iloc[i,:])
            sns.heatmap(pearson_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,0])
            sns.heatmap(spearman_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,1])
            sns.heatmap(kendall_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,2])
            sns.heatmap(mse_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,3])
            sns.heatmap(ccc_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[0,4])
            sns.heatmap(cohens_kappa_norm_diff_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,1])
            sns.heatmap(sda_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,2])
            sns.heatmap(abs_sda_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[1,3])
            sns.heatmap(pearson_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[2,0])
            sns.heatmap(spearman_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[2,1])
            sns.heatmap(kendall_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[2,2])
            sns.heatmap(mse_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[2,3])
            sns.heatmap(ccc_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[2,4])
            sns.heatmap(cohens_kappa_norm_diff_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[3,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_dtw_corr_mat_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[3,1])
            sns.heatmap(sda_mat_dtw_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[3,2])
            sns.heatmap(abs_sda_mat_dtw_agg, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_agg[3,3])

            # Spectral clustered pairwise correlation matrices
            pearson_corr_mat_spec = pearson_norm_corr_mat 
            spearman_corr_mat_spec = spearman_norm_corr_mat
            kendall_corr_mat_spec = kendall_norm_corr_mat
            ccc_corr_mat_spec = ccc_norm_corr_mat
            mse_corr_mat_spec = mse_corr_mat
            cohens_kappa_norm_diff_corr_mat_spec = cohens_kappa_norm_diff_corr_mat
            cohens_kappa_abs_norm_diff_corr_mat_spec = cohens_kappa_abs_norm_diff_corr_mat
            sda_mat_spec = sda_norm_mat
            abs_sda_mat_spec = abs_sda_norm_mat
            pearson_dtw_corr_mat_spec = pearson_norm_dtw_corr_mat 
            spearman_dtw_corr_mat_spec = spearman_norm_dtw_corr_mat
            kendall_dtw_corr_mat_spec = kendall_norm_dtw_corr_mat
            ccc_dtw_corr_mat_spec = ccc_dtw_norm_corr_mat
            mse_dtw_corr_mat_spec = mse_dtw_corr_mat
            cohens_kappa_norm_diff_dtw_corr_mat_spec = cohens_kappa_norm_diff_dtw_corr_mat
            cohens_kappa_abs_norm_diff_dtw_corr_mat_spec = cohens_kappa_abs_norm_diff_dtw_corr_mat
            sda_mat_dtw_spec = sda_norm_dtw_mat
            abs_sda_mat_dtw_spec = abs_sda_norm_dtw_mat
            spec_corr_mats = [(pearson_corr_mat_spec, spec_pearson_labels), (spearman_corr_mat_spec, spec_spearman_labels), (kendall_corr_mat_spec, spec_kendall_labels), (ccc_corr_mat_spec, spec_ccc_labels), (mse_corr_mat_spec, spec_mse_labels), (cohens_kappa_norm_diff_corr_mat_spec, spec_cohens_kappa_norm_diff_labels), (cohens_kappa_abs_norm_diff_corr_mat_spec, spec_cohens_kappa_abs_norm_diff_labels), (sda_mat_spec, spec_sda_labels), (abs_sda_mat_spec, spec_abs_sda_labels)]
            spec_corr_mats.extend([(pearson_dtw_corr_mat_spec, spec_pearson_dtw_labels), (spearman_dtw_corr_mat_spec, spec_spearman_dtw_labels), (kendall_dtw_corr_mat_spec, spec_kendall_dtw_labels), (ccc_dtw_corr_mat_spec, spec_ccc_dtw_labels), (mse_dtw_corr_mat_spec, spec_mse_dtw_labels), (cohens_kappa_norm_diff_dtw_corr_mat_spec, spec_cohens_kappa_norm_diff_dtw_labels), (cohens_kappa_abs_norm_diff_dtw_corr_mat_spec, spec_cohens_kappa_abs_norm_diff_dtw_labels), (sda_mat_dtw_spec, spec_sda_dtw_labels), (abs_sda_mat_dtw_spec, spec_abs_sda_dtw_labels)])
            for spec_corr_mat, spec_labels in spec_corr_mats: # Recolor the two clusters
               for i in range(len(spec_corr_mat)):
                  if spec_labels[i] == 1:
                     spec_corr_mat.iloc[i,:] = -np.abs(spec_corr_mat.iloc[i,:])
                     spec_corr_mat.iloc[:,i] = -np.abs(spec_corr_mat.iloc[i,:])
            sns.heatmap(pearson_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,0])
            sns.heatmap(spearman_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,1])
            sns.heatmap(kendall_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,2])
            sns.heatmap(mse_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,3])
            sns.heatmap(ccc_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[0,4])
            sns.heatmap(cohens_kappa_norm_diff_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[1,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[1,1])
            sns.heatmap(sda_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,2])
            sns.heatmap(abs_sda_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[1,3])
            sns.heatmap(pearson_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[2,0])
            sns.heatmap(spearman_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[2,1])
            sns.heatmap(kendall_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[2,2])
            sns.heatmap(mse_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[2,3])
            sns.heatmap(ccc_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[2,4])
            sns.heatmap(cohens_kappa_norm_diff_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[3,0])
            sns.heatmap(cohens_kappa_abs_norm_diff_dtw_corr_mat_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidth=0.5, cbar_kws={"shrink" : 0.5}, ax=axs_spec[3,1])
            sns.heatmap(sda_mat_dtw_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[3,2])
            sns.heatmap(abs_sda_mat_dtw_spec, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs_spec[3,3])

            axs_anno.title.set_text(project_entry_name+' Raw Annotations')
            axs_anno.legend(external_pids)
            axs_opt.title.set_text(project_entry_name+' Opt Annotations')
            axs_opt.legend(opt_pids)
            axs_agree.title.set_text(project_entry_name+' Global Agreement Metrics')
            axs_corr[0,0].title.set_text('Pearson Corr')
            axs_corr[0,1].title.set_text('Spearman Corr')
            axs_corr[0,2].title.set_text('Kendall Tau')
            axs_corr[0,3].title.set_text('MSE Corr')
            axs_corr[0,4].title.set_text('CCC')
            axs_corr[1,0].title.set_text('Cohens Kappa Normed Diff')
            axs_corr[1,1].title.set_text('Cohens Kappa Abs Normed Diff')
            axs_corr[1,2].title.set_text('SDA')
            axs_corr[1,3].title.set_text('ASDA')
            axs_corr[2,0].title.set_text('DTW Pearson Corr')
            axs_corr[2,1].title.set_text('DTW Spearman Corr')
            axs_corr[2,2].title.set_text('DTW Kendall Tau')
            axs_corr[2,3].title.set_text('DTW MSE Corr')
            axs_corr[2,4].title.set_text('DTW CCC')
            axs_corr[3,0].title.set_text('DTW Cohens Kappa Normed Diff')
            axs_corr[3,1].title.set_text('DTW Cohens Kappa Abs Normed Diff')
            axs_corr[3,2].title.set_text('DTW SDA')
            axs_corr[3,3].title.set_text('DTW ASDA')
            axs_agg[0,0].title.set_text('Pearson Agg')
            axs_agg[0,1].title.set_text('Spearman Agg')
            axs_agg[0,2].title.set_text('Kendall Agg')
            axs_agg[0,3].title.set_text('MSE Agg')
            axs_agg[0,4].title.set_text('CCC Agg')
            axs_agg[1,0].title.set_text('Cohens Kappa Normed Diff Agg')
            axs_agg[1,1].title.set_text('Cohens Kappa Abs Normed Diff Agg')
            axs_agg[1,2].title.set_text('SDA Agg')
            axs_agg[1,3].title.set_text('ASDA Agg')
            axs_agg[2,0].title.set_text('DTW Pearson Agg')
            axs_agg[2,1].title.set_text('DTW Spearman Agg')
            axs_agg[2,2].title.set_text('DTW Kendall Agg')
            axs_agg[2,3].title.set_text('DTW MSE Agg')
            axs_agg[2,4].title.set_text('DTW CCC Agg')
            axs_agg[3,0].title.set_text('DTW Cohens Kappa Normed Diff Agg')
            axs_agg[3,1].title.set_text('DTW Cohens Kappa Abs Normed Diff Agg')
            axs_agg[3,2].title.set_text('DTW SDA Agg')
            axs_agg[3,3].title.set_text('DTW ASDA Agg')
            axs_spec[0,0].title.set_text('Pearson Spec')
            axs_spec[0,1].title.set_text('Spearman Spec')
            axs_spec[0,2].title.set_text('Kendall Spec')
            axs_spec[0,3].title.set_text('MSE Spec')
            axs_spec[0,4].title.set_text('CCC Spec')
            axs_spec[1,0].title.set_text('Cohens Kappa Normalized Diff Spec')
            axs_spec[1,1].title.set_text('Cohens Kappa Abs Normalized Diff Spec')
            axs_spec[1,2].title.set_text('SDA Spec')
            axs_spec[1,3].title.set_text('ASDA Spec')
            axs_spec[2,0].title.set_text('DTW Pearson Spec')
            axs_spec[2,1].title.set_text('DTW Spearman Spec')
            axs_spec[2,2].title.set_text('DTW Kendall Spec')
            axs_spec[2,3].title.set_text('DTW MSE Spec')
            axs_spec[2,4].title.set_text('DTW CCC Spec')
            axs_spec[3,0].title.set_text('DTW Cohens Kappa Normalized Diff Spec')
            axs_spec[3,1].title.set_text('DTW Cohens Kappa Abs Normalized Diff Spec')
            axs_spec[3,2].title.set_text('DTW SDA Spec')
            axs_spec[3,3].title.set_text('DTW ASDA Spec')
            fig_corr.suptitle(project_entry_name+' Agreement Measures')
            fig_agg.suptitle(project_entry_name+' Agglomerative Clustering of Agreement')
            fig_spec.suptitle(project_entry_name+' Spectral Clustering of Agreement')

            tikz_output_file = os.path.join(output_path, project_entry_name+'_raw_annotations.tex')
            tikzplotlib.save(tikz_output_file, figure=fig_anno)

            plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_log', required=True, help='Path to log file or folder containing such files exported from PAGAN')
   parser.add_argument('--output_path', required=True, help='Path to folder for output cluster data')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Enables display of annotations and approximations used for agreement calculation')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ComputePaganAgreement(args.input_log, args.output_path, args.show_plots)
