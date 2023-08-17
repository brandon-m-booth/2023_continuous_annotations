import os
import dtw
import numpy as np
import pandas as pd
import tempfile
import krippendorff
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
import pytz

# Compute the DTW for each column (signal) in the data frame with respect to the reference
def DTWReference(df, reference_df, max_warp_distance=None):
   window_type = None
   window_args = {}
   step_pattern = dtw.asymmetric
   if max_warp_distance is not None:
      window_type = "sakoechiba"
      window_args = {'window_size': max_warp_distance}
   dtw_df = pd.DataFrame(data=np.zeros((df.shape[0], df.shape[1])), columns=df.columns, index=df.index)
   for i in range(df.shape[1]):
      warp = dtw.dtw(reference_df.values, df.iloc[:,i].values, keep_internals=True, step_pattern=step_pattern, window_type=window_type, window_args=window_args)

      # Reassemble the warped signals and resample them to align with the original df index
      dtw_df.iloc[:, i] = df.iloc[:,i].iloc[warp.index2].values
   return dtw_df

# Compute the DTW for each unique ordered pair of signals
def DTWPairwise(df, max_warp_distance=None):
   window_type = None
   window_args = {}
   step_pattern = dtw.symmetric2
   if max_warp_distance is not None:
      window_type = "sakoechiba"
      window_args = {'window_size': max_warp_distance}
   dtw_df = pd.DataFrame(data=np.zeros((df.shape[0], 2*df.shape[1]**2)), columns=pd.MultiIndex.from_product([df.columns,df.columns, ['query', 'template']]), index=df.index)
   for i in range(df.shape[1]):
      for j in range(df.shape[1]):
         col1 = df.columns[i]
         col2 = df.columns[j]
         warp = dtw.dtw(df.iloc[:,i].values, df.iloc[:,j].values, keep_internals=True, step_pattern=step_pattern, window_type=window_type, window_args=window_args)

         # Reassemble the warped signals and resample them to align with the original df index
         warp_sym_index = (warp.index1+warp.index2)/2.0
         warp_sym_df_index_values = np.interp(warp_sym_index, range(df.shape[0]), df.index.values)
         warp_temp_mat = np.vstack((df.iloc[:,i].iloc[warp.index1].values, df.iloc[:,j].iloc[warp.index2].values)).T
         temp_df = pd.DataFrame(data=warp_temp_mat, columns=['query', 'template'], index=warp_sym_df_index_values)
         temp_df.index = [datetime.fromtimestamp(t, pytz.utc) for t in temp_df.index]
         temp_df = temp_df.resample('100ms').mean().interpolate()
         temp_df.index = [t.timestamp() for t in temp_df.index]
         dtw_df[col1, col2, 'query'] = temp_df['query'].values
         dtw_df[col1, col2, 'template'] = temp_df['template'].values
         #dtw_df[col1, col2, 'query'] = df.iloc[:,i].iloc[warp.index1].values
         #dtw_df[col1, col2, 'template'] = df.iloc[:,j].iloc[warp.index2].values
   return dtw_df

def DTWPairwiseAgreement(dtw_df, columns, agreement_func, agreement_func_args={}, is_agreement_symmetric=True):
   corr_df = pd.DataFrame(data=np.zeros((len(columns), len(columns))), columns=columns, index=columns)
   for i in range(len(columns)):
      lower_j = i if is_agreement_symmetric else 0
      for j in range(lower_j,len(columns)):
         col1 = columns[i]
         col2 = columns[j]
         aligned_signals_df = dtw_df[col1, col2]
         if len(agreement_func_args) > 0:
            corr = agreement_func(aligned_signals_df, **agreement_func_args)
         else:
            corr = agreement_func(aligned_signals_df)
         corr_df.loc[col1, col2] = corr.iloc[1,0]
         if is_agreement_symmetric:
            corr_df.loc[col2, col1] = corr.iloc[1,0]
   return corr_df

def PearsonCorr(df):
   return df.corr(method='pearson')

def DTWPearsonCorr(dtw_df):
   cols = dtw_df.columns

def SpearmanCorr(df):
   return df.corr(method='spearman')

def KendallTauCorr(df):
   return df.corr(method='kendall')

#def CronbachsAlphaCorr(df):
#   cov = df.cov()
#   k = df.shape[1]
#   return k/(k-1.0)*(1 - cov.sum().sum()/np.trace(cov))

def CronbachsAlphaCorr(df):
    # cols are items, rows are observations
    itemscores = np.asarray(df)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]

    return (nitems / (nitems-1)) * (1 - (itemvars.sum() / tscores.var(ddof=1)))

# DF columns are annotators, row are time indices
#def ComputeCronbachAlpha(df):
#   vars_vals = df.var(axis=0, ddof=1)
#   sum_vals = df.sum(axis=1)
#   N = df.shape[1]
#   return (N/(N-1))*(1.0 - vars_vals.sum() / sum_vals.var(ddof=1))

def MeanSquaredErrorMat(df):
   mse_mat = df.corr(method=mean_squared_error)
   np.fill_diagonal(mse_mat.values, 0)
   return mse_mat

# ddof == 0: unbiased
# ddof == 1: biased
def _CCCHelper(x, y, ddof=0):
   pearson_corr = pearsonr(x,y)[0]
   if np.isnan(pearson_corr):
      return 1.0 # Happens when x and y are constant
   return 2*pearson_corr*np.std(x, ddof=ddof)*np.std(y, ddof=ddof)/(np.var(x, ddof=ddof) + np.var(y, ddof=ddof) + (np.mean(x) - np.mean(y))**2)

def ConcordanceCorrelationCoef(df, biased=True):
   ddof = 1 if biased else 0
   ccc = lambda x,y: _CCCHelper(x,y,ddof)
   return df.corr(method=ccc)

def _KappaHelper(x, y, p_agree=None, labels=None):
   if p_agree is None:
      return cohen_kappa_score(x, y, labels)
   else:
      obs_p_agree = float(np.sum(x == y))/len(x)
      kappa = (obs_p_agree - p_agree)/(1.0 - p_agree)
      return kappa

def CohensKappaCorr(df, labels=None):
   # Compute empirical estimate of the prior probability of agreement over all annotators
   #total_agree = 0
   #for i in range(df.shape[1]):
   #   for j in range(i+1,df.shape[1]):
   #      num_agree = np.sum(df.iloc[:,i] == df.iloc[:,j])
   #      total_agree += num_agree
   #p_agree = 2.0*total_agree/(df.shape[1]*(df.shape[1]-1)*df.shape[0])
   #kappa = lambda x,y: _KappaHelper(x, y, p_agree=p_agree, labels=labels)
   kappa = lambda x,y: cohen_kappa_score(x, y, labels=labels)
   return df.corr(method=kappa)

def KrippendorffsAlpha(df):
   dir_path = os.path.dirname(os.path.realpath(__file__))
   temp_df = os.path.join(tempfile.gettempdir(), 'temp_df.csv')
   results_path = os.path.join(tempfile.gettempdir(), 'krippendorffs_alpha_results.csv')
   df.to_csv(temp_df, index=False, header=True)
   os.system("Rscript "+os.path.join(dir_path, 'krippendorffs_alpha_helper.R')+ " -i "+temp_df+" -m nominal"+" -o "+results_path)
   alpha_df = pd.read_csv(results_path)
   return alpha_df.values[0,0]
   #return krippendorff.alpha(df.values.T)

def ICC(df):
   dir_path = os.path.dirname(os.path.realpath(__file__))
   temp_df = os.path.join(tempfile.gettempdir(), 'temp_df.csv')
   results_path = os.path.join(tempfile.gettempdir(), 'icc_results.csv')
   df.to_csv(temp_df, index=False, header=True)
   os.system("Rscript "+os.path.join(dir_path, 'icc_helper.R')+ " -i "+temp_df+" -o "+results_path)
   icc_df = pd.read_csv(results_path, index_col=0, header=0)
   return icc_df

def SDA(norm_diff_df):
   norm_delta = lambda x,y: 2*np.sum((x-y) == 0)/float(len(x)) - 1.0
   return norm_diff_df.corr(method=norm_delta)

def SDACorrected(norm_diff_df):
   norm_delta_kronecker = lambda x,y: np.sum((x-y) == 0)/float(len(x))
   po = norm_diff_df.corr(method=norm_delta_kronecker)
   chance_agree = lambda x,y: np.sum([np.sum(x==i)*np.sum(y==i) for i in [-1,0,1]])/float((len(x)-1)**2)
   pe = norm_diff_df.corr(method=chance_agree)
   return 1-(1.0-po)/(1.0-pe)

def SDACorrectedTrend(norm_diff_df):
   sda_trend = lambda x,y: (np.sum(x+y == 2)+np.sum(x+y == -2))/float(np.sum(np.logical_or(x != 0, y != 0)))
   po = norm_diff_df.corr(method=sda_trend)
   trend_chance_agree = lambda x,y: np.sum([np.sum(x==i)*np.sum(y==i) for i in [-1,1]])/float((len(x)-1)**2)
   pe = norm_diff_df.corr(method=trend_chance_agree)
   return 1-(1.0-po)/(1.0-pe)

def KSDA(norm_diff_df):
   return CohensKappaCorr(norm_diff_df, labels=np.unique(norm_diff_df.values))
