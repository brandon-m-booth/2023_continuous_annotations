import os
import numpy as np
import pandas as pd
import tempfile
import krippendorff
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def PearsonCorr(df):
   return df.corr(method='pearson')

def SpearmanCorr(df):
   return df.corr(method='spearman')

def KendallTauCorr(df):
   return df.corr(method='kendall')

def CronbachsAlphaCorr(df):
   cov = df.cov()
   k = df.shape[1]
   return k/(k-1.0)*(1 - cov.sum().sum()/np.trace(cov))

# DF columns are annotators, row are time indices
#def ComputeCronbachAlpha(df):
#   vars_vals = df.var(axis=0, ddof=1)
#   sum_vals = df.sum(axis=1)
#   N = df.shape[1]
#   return (N/(N-1))*(1.0 - vars_vals.sum() / sum_vals.var(ddof=1))

def MeanSquaredErrorMat(df):
   return df.corr(method=mean_squared_error)

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

def NormedSumDelta(df):
   norm_delta = lambda x,y: np.sum((x-y) == 0)/float(len(x))
   return df.corr(method=norm_delta)

# prob_value: The probability of getting any individual value in the df. Each observed value
#             is assumed to be iid.
def NormedSumDeltaCorrected(df, prob_value):
   norm_delta_corrected = lambda x,y: (1.0 - prob_value**len(x))*np.sum((x-y) == 0)/float(len(x))
   return df.corr(method=norm_delta_corrected)
