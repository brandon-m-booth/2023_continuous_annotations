import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
import util
import agreement_metrics as agree

valid_time_percentage = 0.20

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


def ComputePaganAgreement(input_file_path, do_show_plots=False):
   input_files = [input_file_path]
   if os.path.isdir(input_file_path):
      input_files = glob.glob(os.path.join(input_file_path, '*.csv'))
   for input_file in input_files:
      df = pd.read_csv(input_file)
      unique_experiment_names = np.unique(df['OriginalName'])
      for project_entry_name in unique_experiment_names:
         print('Processing '+project_entry_name)
         proj_df = df.loc[df['OriginalName'] == project_entry_name, :]
         pids = np.unique(proj_df['Participant'])
         external_pids = []

         if do_show_plots:
            fig, axs = plt.subplots(4,4, figsize=(11,9))

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
               axs[3,3].plot(anno_time, anno_vals)

            if not max(anno_time) >= valid_time_percentage*end_time:
               print('Rejecting annotation '+external_pid+' because end time is: '+str(max(anno_time))+'/'+str(end_time))
               continue
            # TODO - Add code to NaN-out periods where no data is present, then suddenly data appears at a different value.  I.e., remove the sloped lines in the plot. These should get no vote during this period (lost data)

            if combined_anno_df is None:
               combined_anno_df = anno_df
            else:
               combined_anno_df = pd.merge(combined_anno_df, anno_df, how='outer', left_index=True, right_index=True, sort=False)
         combined_anno_df = combined_anno_df.sort_index()
         combined_anno_df = combined_anno_df.interpolate(method='linear', axis=0)
         combined_anno_df = combined_anno_df.groupby(level=0).mean()

         ###############
         # Compute different pairwise agreement measures
         ###############
         # Pearson
         pearson_corr_mat = agree.PearsonCorr(combined_anno_df)

         # Spearman
         spearman_corr_mat = agree.SpearmanCorr(combined_anno_df)

         # Kendall's Tau
         kendall_corr_mat = agree.KendallTauCorr(combined_anno_df)

         # Cohen's kappa 
         # BB - skipping this one; requires partitioning the continuous space into mutex bins

         # MSE
         mse_mat = agree.MeanSquaredErrorMat(combined_anno_df)

         # CCC
         ccc_mat = agree.ConcordanceCorrelationCoef(combined_anno_df)

         # TSS, T>=N-1
         norm_diff_df = util.NormedDiffRank(combined_anno_df)
         norm_sum_delta_mat = agree.NormedSumDelta(norm_diff_df)

         # TSS corrected, T>=N-1
         # TODO - Should I correct using the same Cronbach's alpha idea?
         norm_sum_delta_corrected_mat = agree.NormedSumDeltaCorrected(norm_diff_df, prob_value=1.0/3)

         ###############
         # Compute summary agreement measures
         ###############
         # Cronbach's alpha
         cronbachs_alpha = agree.CronbachsAlphaCorr(combined_anno_df)

         # ICC(2,1)
         icc_df = agree.ICC(combined_anno_df)
         icc21_df = icc_df.loc[icc_df['type'] == 'ICC2',:]
         icc21 = icc21_df['ICC'].iloc[0]

         # Krippendorff's alpha
         krippendorffs_alpha = agree.KrippendorffsAlpha(combined_anno_df)

         # SAGR (signed agreement)
         # BB - Doesn't make sense for scales where zero isn't the center

         # Accumulated Normed Rank-based Krippendorff's Alpha
         accum_norm_rank_df = util.AccumNormedDiffRank(combined_anno_df)
         krippendorffs_alpha_accum_norm_rank = agree.KrippendorffsAlpha(accum_norm_rank_df)

         # TSS for T < N-1 ?
         # TODO - Compute TSR separately and load then in?
         # TODO - Add a TSR->TSS transform, then add a normed zero-one similarity function

         # Put global agreement measures into a dataframe
         global_agreement_df = pd.DataFrame(data=[[cronbachs_alpha, icc21, krippendorffs_alpha, krippendorffs_alpha_accum_norm_rank]], columns=['Cronbach\'s Alpha', 'ICC(2)', 'Krippendorff\'s Alpha', 'Krippendorff\'s Alpha Normed Ranks'])

         # Max-normalize the MSE and convert to a correlation-like matrix
         mse_corr_mat = 1.0 - mse_mat.values/np.max(mse_mat.values)

         # Force symmetry for corr matrices
         pearson_corr_mat[pd.isna(pearson_corr_mat)] = 0
         np.fill_diagonal(pearson_corr_mat.values, 1)
         spearman_corr_mat[pd.isna(spearman_corr_mat)] = 0
         np.fill_diagonal(spearman_corr_mat.values, 1)
         kendall_corr_mat[pd.isna(kendall_corr_mat)] = 0
         np.fill_diagonal(kendall_corr_mat.values, 1)
         np.fill_diagonal(mse_corr_mat, 1)

         # Clustering
         (agg_pearson_sim, pearson_cluster_order_idx, pearson_agg_link) = ClusterSimilarityMatrix(pearson_corr_mat.values, method='centroid')
         (agg_spearman_sim, spearman_cluster_order_idx, spearman_agg_link) = ClusterSimilarityMatrix(spearman_corr_mat.values, method='centroid')
         (agg_kendall_sim, kendall_cluster_order_idx, kendall_agg_link) = ClusterSimilarityMatrix(kendall_corr_mat.values, method='centroid')
         (agg_mse_sim, mse_cluster_order_idx, mse_agg_link) = ClusterSimilarityMatrix(mse_corr_mat, method='centroid')
         (agg_ccc_sim, ccc_cluster_order_idx, ccc_agg_link) = ClusterSimilarityMatrix(ccc_mat.values, method='centroid')
         (agg_norm_sum_delta_sim, norm_sum_delta_cluster_order_idx, norm_sum_delta_agg_link) = ClusterSimilarityMatrix(norm_sum_delta_mat.values, method='centroid')
         (agg_norm_sum_delta_corrected_sim, norm_sum_delta_corrected_cluster_order_idx, norm_sum_delta_corrected_agg_link) = ClusterSimilarityMatrix(norm_sum_delta_corrected_mat.values, method='centroid')

         if do_show_plots:
            # Pairwise correlation and error matrices
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(pearson_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[0,0])
            sns.heatmap(spearman_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[0,1])
            sns.heatmap(kendall_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[0,2])
            sns.heatmap(mse_corr_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[0,3])
            sns.heatmap(ccc_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[1,0])
            sns.heatmap(norm_sum_delta_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[1,1])
            sns.heatmap(norm_sum_delta_corrected_mat, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[1,2])

            # Summary agreement bar graph
            sns.barplot(data=global_agreement_df, ax=axs[1,3])

            # Clustered pairwise correlation and error matrices
            sns.heatmap(agg_pearson_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[2,0])
            sns.heatmap(agg_spearman_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[2,1])
            sns.heatmap(agg_kendall_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[2,2])
            sns.heatmap(agg_mse_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[2,3])
            sns.heatmap(agg_ccc_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[3,0])
            sns.heatmap(agg_norm_sum_delta_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[3,1])
            sns.heatmap(agg_norm_sum_delta_corrected_sim, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axs[3,2])

            axs[0,0].title.set_text('Pearson Corr')
            axs[0,1].title.set_text('Spearman Corr')
            axs[0,2].title.set_text('Kendall Tau')
            axs[0,3].title.set_text('MSE')
            axs[1,0].title.set_text('CCC')
            axs[1,1].title.set_text('Normed Sum Delta')
            axs[1,2].title.set_text('Normed Sum Delta Corrected')
            axs[1,3].title.set_text('Global Agreement Metrics')
            axs[2,0].title.set_text('Pearson Agg Clustering')
            axs[2,1].title.set_text('Spearman Agg Clustering')
            axs[2,2].title.set_text('Kendall Agg Clustering')
            axs[2,3].title.set_text('MSE Agg Clustering')
            axs[3,0].title.set_text('CCC Agg Clustering')
            axs[3,1].title.set_text('Normed Sum Delta Agg Clustering')
            axs[3,2].title.set_text('Normed Sum Delta Corrected Agg Clustering')
            axs[3,3].title.set_text('Raw Annotations')
            axs[2,2].legend(external_pids)
            fig.suptitle(project_entry_name+' Agreement Measures')
            plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_log', required=True, help='Path to log file or folder containing such files exported from PAGAN')
   parser.add_argument('--show_plots', required=False, action='store_true', help='Enables display of annotations and approximations used for agreement calculation')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ComputePaganAgreement(args.input_log, args.show_plots)
