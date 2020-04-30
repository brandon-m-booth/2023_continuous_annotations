import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import itertools
import seaborn as sns
from datetime import datetime
# BB - This package doesn't compute the FFT properly
#from nfft import nfft, nfft_adjoint

def ProposeMTurkRejects(pagan_logs_path):
   input_files = [pagan_logs_path]
   if os.path.isdir(pagan_logs_path):
      input_files = glob.glob(os.path.join(pagan_logs_path, '*.csv'))


   for input_file in input_files:
      df = pd.read_csv(input_file)
      unique_experiment_names = np.unique(df['OriginalName'])
      for project_entry_name in unique_experiment_names:
         proj_df = df.loc[df['OriginalName'] == project_entry_name, :]

         pids = np.unique(proj_df['Participant'])

         fig, axs = plt.subplots(1,len(pids)+1)
         palette = itertools.cycle(sns.color_palette())

         for pid_idx in range(len(pids)):
            pid = pids[pid_idx]
            pid_df = proj_df.loc[proj_df['Participant']==pid,:]
            pid_df = pid_df.sort_values(by=['VideoTime'], kind='mergesort') # Stable sort alg.
            anno_time = pid_df.loc[:,'VideoTime'].values
            anno_vals = pid_df.loc[:,'Value'].values

            # High frequency
            timestamps = [datetime.fromtimestamp(t) for t in anno_time]
            resample_series = pd.Series(data=anno_vals, index=timestamps)
            upsampled_series = resample_series.resample('1S').mean().interpolate()
            upsampled_time = [t.strftime('%s') for t in upsampled_series.index]
            upsampled_vals = upsampled_series.values
            N = len(upsampled_vals)
            dt = 1.0
            f, t, Sxx = signal.spectrogram(upsampled_vals,1/dt)

            # BB - This code uses np.fft, which works, but we want to threshold multiple occurrences
            # of high frequency over time
            #timestamps = [datetime.fromtimestamp(t) for t in anno_time]
            #resample_series = pd.Series(data=anno_vals, index=timestamps)
            #upsampled_series = resample_series.resample('1S').mean().interpolate()
            #upsampled_time = [t.strftime('%s') for t in upsampled_series.index]
            #upsampled_vals = upsampled_series.values
            #N = len(upsampled_vals)
            #if N >= 2:
            #   if N%2 == 1:
            #      upsampled_time.append(str(2*int(upsampled_time[-1])-int(upsampled_time[-2])))
            #      upsampled_vals = np.append(upsampled_vals, np.nan)
            #      N += 1
            #   psd = np.absolute(np.fft.fft(upsampled_vals))
            #   dt = int(upsampled_time[1]) - int(upsampled_time[0])
            #   freqs = np.fft.fftfreq(N, d=dt)
            #else:
            #   dt = 1
            #   freqs = np.fft.fftfreq(N, d=dt)
            #   psd = len(freqs)*[np.nan]

            # BB - This code block uses the nfft package and it doesn't compute the FFT correctly
            #if len(anno_vals) > 3:
            #   if len(anno_vals)%2 == 1:
            #      anno_time = anno_time[0:-1]
            #      anno_vals = anno_vals[0:-1]
            #   N = len(anno_vals)
            #   fourier_trans = nfft(anno_time, anno_vals)
            #else:
            #   N = 2
            #   fourier_trans = [0,1]
            #psd = np.absolute(fourier_trans)

            # No variability at all

            # Not full duration (must check length)

            c = next(palette)
            axs[0].plot(anno_time, anno_vals, color=c)
            #axs[1].plot(freqs, psd, color=c)
            axs[pid_idx].pcolormesh(t,f,Sxx)
         axs[0].legend(pids)
         axs[0].set(xlabel='Video Time', ylabel='Annotation Value')
         for spec_gram_idx in range(1,len(axs)):
            axs[spec_gram_idx].set(xlabel='Freq Coef', ylabel='Spectral Power')
            axs[spec_gram_idx].title.set_text(pids[spec_gram_idx-1])
         fig.suptitle(project_entry_name)
         plt.show()
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--pagan_logs_path', required=True, help='Path to file or folder containing PAGAN logs produced from MTurk crowd sourcing')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   ProposeMTurkRejects(args.pagan_logs_path)
