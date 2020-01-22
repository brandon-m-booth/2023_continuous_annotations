#!/usr/bin/env python3
#Author: Brandon M. Booth

import io
import os
import sys
import pdb
import math
import decimal
import argparse
import statprof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib2tikz
from multiprocessing import Pool
from collections.abc import Iterable
import cProfile, pstats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'util')))
import util

# For debugging
show_verbose_output = True
show_final_plot = True
show_debug_plots = False
can_parallelize = True
enable_profiler = False

# Recovers the optimum TSR of the signal up to index i (1 <= i <= n) for the given number of segments t (1 <= t <= num_segments)
def RecoverOptimumTSR(i, t, X, I, A, B, signal):
   if i == 1:
      x = [signal.index[0]]
      y = [A[0,0]*signal.index[0] + B[0,0]]
   elif t == 1:
      x = signal.index[0:i].tolist()
      y = [A[i-1,0]*z + B[i-1,0] for z in x]
   else:
      # Find the knot x-axis locations and get the indices of the signal points with x values just before each knot
      knots = [i]
      x = [signal.index[i-1]]
      while t > 0:
         x.append(X[knots[-1]-1,t-1])
         knot = I[knots[-1]-1,t-1]
         #if knot == knots[-1] and len(knots) > 1:
         #   knot -= 1
         knots.append(knot)
         t -= 1
      knots.reverse()
      x.reverse()

      # Compute the y-axis value at each knot location
      y = [A[knots[1]-1,0]*x[0] + B[knots[1]-1,0]]
      t = 0
      for i in range(1,len(knots)-1):
         t += 1
         y.append(A[knots[i]-1,t-1]*x[i] + B[knots[i]-1,t-1])
      if False:#knots[-2] == knots[-1]:
         y.append(A[knots[-2]-1,t]*x[-1] + B[knots[-2]-1,t])
      else:
         y.append(A[knots[-1]-1,t]*x[-1] + B[knots[-1]-1,t])

   return (x,y)

def FitNextSegment(signal, n, i, j, t, A, B, X, started_const):
   """
   Helper function to fit the correct next line segment type for trapezoidal functions
   """
   last_knot_x = X[i-1,t-2]
   x1 = max(last_knot_x, signal.index[max(0,i-2)])
   x2 = signal.index[min(i-1,n-1)]
   #previous_segment_slope = A[i-1,t-2]
   #if previous_segment_slope == 0.0:
   if (started_const and t%2 == 0) or (not started_const and t%2 == 1):
      #mid_x = (x1+x2)/2.0
      #x1 = mid_x
      #x2 = mid_x
      (a, b, x, cost) = util.FitLineSegmentWithIntersection(signal, i-1, j-1, A[i-1,t-2], B[i-1,t-2], x1, x2)
   else:
      a = 0
      (b, x, cost) = util.FitConstantSegmentWithIntersection(signal, i-1, j-1, A[i-1,t-2], B[i-1,t-2], x1, x2)
   return a, b, x, cost

# The num_segments argument can be a list or an integer. A dictionary is returned where the keys
# are the different number of segments requested and the values are tuples containing the optimal
# signal and optimal cost. Runtime performance scales by the the maximum value in the num_segments
# list.
def ComputeOptimalTSRFixedSegments(num_segments, time, signal):
   if not isinstance(num_segments, Iterable):
      num_segments = [num_segments]
   if show_verbose_output:
      print("Computing optimal segmented trapezoidal fit with %s segments..."%(str(num_segments)))
   max_num_segments = max(num_segments)
   opt_tsr_dict = {}
   for start_with_constant_segment in [True, False]:
      n = len(time)
      F = np.nan*np.zeros((n, max_num_segments))
      I = np.zeros((n, max_num_segments)).astype(int)
      A = np.nan*np.zeros((n,n))
      B = np.nan*np.zeros((n,n))
      X = np.nan*np.zeros((n, max_num_segments))
      # Iterate over all sorted points in order
      for j in range(1,n+1):
         if show_verbose_output:
            print("Computing optimal sub-segmentation for points up to index %d of %d"%(j, n))

         # Initialize costs and set knot indices to an invalid index
         for t in range(j,max_num_segments+1):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0
            X[j-1,t-1] = 0

         # Fit a single line segment to all points up to the current one
         if start_with_constant_segment:
            a = 0.0
            (b, cost) = util.FitConstantSegment(signal.iloc[0:j])
         else:
            (a,b,cost) = util.FitLineSegment(signal.iloc[0:j])
            #if abs(a) < 1e-4:
            #   a = 0.0
            #   cost = np.inf # Don't allow fitted lines to be constant
         A[j-1,0] = a
         B[j-1,0] = b
         X[j-1,0] = signal.index[0]
         F[j-1,0] = cost
         I[j-1,0] = 1

         # Consider using all possible number of segments between the first point and the current
         for t in range(2,min(j,max_num_segments+1)):
            F[j-1,t-1] = np.inf
            I[j-1,t-1] = 0

            # For the target number of t segments, find the best break point reusing the optimum
            # fit for t-1 segments over all points up to some point before the current
            last_knots = []
            for i in range(t,j+1): # BB TODO - Can I go with j+1?  I don't have enough points for i=j to work? No constraint on the slope...
               k = I[i-1,t-2]
               if k != 0 and A[k-1,i-1] != A[i-1,j-1]:
                  last_knots.append(i)

            # Fit the next segment
            results = [FitNextSegment(signal, n, i, j, t, A, B, X, start_with_constant_segment) for i in last_knots]

            #if t == 3 and j == 4:
            #   pdb.set_trace()
            #   FitNextSegment(signal, n, 3, j, t, A, B, start_with_constant_segment)
            # Update the cost, linear coefficients, and break points
            avals, bvals, xvals, costs = zip(*results)

            # BB TODO - Update the actual cost with the best estimate.  This does not fix anything
            # DEBUG COST
            costs = [x for x in costs]
            for results_idx in range(len(results)):
               a,b,x,cost = results[results_idx]
               if cost < 0:
                  print("COST IS NEGATIVE")
                  pdb.set_trace()

               new_points_x = np.array(signal.index[last_knots[results_idx]-1:j])
               new_points_y = a*new_points_x + b
               new_sse = np.sum((new_points_y - signal.iloc[last_knots[results_idx]-1:j])**2)
               costs[results_idx] = new_sse
            # END DEBUG COST

            last_knots = np.array(last_knots)
            prev_sum_costs = F[last_knots-1,t-2]
            last_knot_cost_fixup = (A[last_knots-1,t-2]*signal.index[last_knots-1]+B[last_knots-1,t-2]-signal.iloc[last_knots-1])**2
            prev_sum_costs = np.array(prev_sum_costs-last_knot_cost_fixup)
            min_idx = np.argmin(prev_sum_costs+costs)
            if F[j-1,t-1] >= (prev_sum_costs+costs)[min_idx]:
               F[j-1,t-1] = (prev_sum_costs+costs)[min_idx]
               A[j-1,t-1] = avals[min_idx]
               B[j-1,t-1] = bvals[min_idx]
               I[j-1,t-1] = last_knots[min_idx]
               X[j-1,t-1] = xvals[min_idx]

            # BB TODO - Compare the actual cost against the one stored in F.  This is not helped by the above debug code
            # which sets F to the actual cost
            ###### DEBUGGING ######
            dx,dy = RecoverOptimumTSR(j, t, X, I, A, B, signal)
            if not np.any(np.isnan(dx)):
               ddf = pd.DataFrame(data=dy, index=dx, columns=['Data'])
               signal_df = pd.DataFrame(data=signal.iloc[0:j], index=signal.index[0:j], columns=['Data'])
               actual_cost = util.GetTSRSumSquareError(ddf, signal_df)
               dcost = F[j-1,t-1]
               if np.isnan(actual_cost) or (abs(actual_cost - dcost)/actual_cost > 0.01 and not actual_cost < 1e-15):
                  print('Cost mismatch!')
                  print('T='+str(t)+', i='+str(i)+', j='+str(j))
                  pdb.set_trace()
                  dx,dy = RecoverOptimumTSR(j, t, X, I, A, B, signal)
                  actual_cost = util.GetTSRSumSquareError(ddf, signal_df)
            ### END DEBUGGING ####

            if show_debug_plots:
               for results_idx in range(len(results)):
                  a,b,x,cost = results[results_idx]
                  i = last_knots[results_idx]
                  if not np.isinf(cost):
                     # Get best TSR up to signal index i for t-1 segments
                     best_tsr_so_far_x, best_tsr_so_far_y = RecoverOptimumTSR(i, t-1, X, I, A, B, signal)

                     # Store the best line segment computed this iteration
                     new_line_x = np.array(signal.index[i-1:j]).astype(float)
                     new_line_y = a*new_line_x + b

                     # Find the intersection of the new line and the best TSR so far
                     if len(best_tsr_so_far_x) < 2:
                        best_tsr_so_far_x = 2*best_tsr_so_far_x
                        best_tsr_so_far_y = 2*best_tsr_so_far_y
                        # In this case, assume best_tsr_so_far_y is a constant function with duplicates in best_tsr_so_far_x
                        if abs(a) == 0.0:
                           x_int = signal.index[i-1]
                           y_int = b
                        else:
                           x_int = (best_tsr_so_far_y[-1]-b)/a
                           y_int = best_tsr_so_far_y[-1]
                     #if np.isnan(c):
                     #   x_int = best_tsr_so_far_x[-1]
                     #   y_int = best_tsr_so_far_y[-1]
                     else:
                        denom = a*(best_tsr_so_far_x[-1]-best_tsr_so_far_x[-2])-best_tsr_so_far_y[-1]+best_tsr_so_far_y[-2]
                        c = (best_tsr_so_far_y[-2]-b-a*best_tsr_so_far_x[-2])/denom
                        x_int = best_tsr_so_far_x[-2] + c*(best_tsr_so_far_x[-1]-best_tsr_so_far_x[-2])
                        y_int = best_tsr_so_far_y[-2] + c*(best_tsr_so_far_y[-1]-best_tsr_so_far_y[-2])


                     # Fix up the TSR and new line points so they share the intersection
                     best_tsr_so_far_x[-1] = x_int
                     best_tsr_so_far_y[-1] = y_int
                     if len(new_line_x) == 1:
                        new_line_x = [new_line_x[0], new_line_x[0]]
                        new_line_y = [new_line_y[0], new_line_y[0]]
                        new_line_x[0] = x_int
                        new_line_y[0] = y_int
                     else:
                        new_line_x[0] = x_int
                        new_line_y[0] = y_int

                     plt.figure()
                     plt.plot(signal.index[0:i-1], signal.iloc[0:i-1], 'mo')
                     plt.plot(signal.index[i:j], signal.iloc[i:j], 'bo')
                     plt.plot(signal.index[i-1:i], signal.iloc[i-1:i], 'ko')
                     plt.plot(new_line_x, new_line_y, 'g--')
                     plt.plot(best_tsr_so_far_x, best_tsr_so_far_y, 'r-')
                     plt.title("T=%d, i=%d, j=%d, Old cost: %f, New cost: %f"%(t, i, j, F[i-1, t-2], cost))
                     plt.show()

         # Handle under-constrained cases by back-filling results
         #if j > 2:
         #   F[j-2,j-2] = F[j-1,j-2]
         #   A[j-2,j-2] = A[j-1,j-2]
         #   B[j-2,j-2] = B[j-1,j-2]
         #   I[j-2,j-2] = j-2 if I[j-1,j-2] > j-2 else I[j-1,j-2]
         #   X[j-2,j-2] = X[j-1,j-2]

      # Recover optimum TSR
      for t in num_segments:
         x,y = RecoverOptimumTSR(n, t, X, I, A, B, signal)
         cost = F[n-1, t-1]

         start_segment_type = "constant-segment-first" if start_with_constant_segment else "linear-segment-first"
         if show_verbose_output:
            print("Final T=%d %s approximation loss value: %f"%(t, start_segment_type, cost))

         if t not in opt_tsr_dict.keys(): 
            opt_tsr_dict[t] = {'x':x, 'y':y, 'cost':cost}
         elif cost <= opt_tsr_dict[t]['cost']:
            opt_tsr_dict[t] = {'x':x, 'y':y, 'cost':cost}
   return opt_tsr_dict

def ComputeOptimalTSRFixedSegmentsArgs(args):
   return ComputeOptimalTSRFixedSegments(*args)

# Dynamic program to find the optimal trapezoidal segmented regression
def ComputeOptimalTSR(input_csv_path, min_segments, max_segments, output_csv_path, opt_strategy=None, tikz_file=None):
   if not os.path.isdir(os.path.dirname(output_csv_path)):
      os.makedirs(os.path.dirname(output_csv_path))

   # Get the signal data
   signal_df = pd.read_csv(input_csv_path)
   time = signal_df.iloc[:,0]
   signal = signal_df.iloc[:,1]
   signal.index = time

   # Quantize the signal to remove meaningless tiny differences between values
   quantized_signal= [decimal.Decimal(x).quantize(decimal.Decimal('0.00001'), rounding=decimal.ROUND_HALF_UP) for x in signal]
   signal = pd.Series(np.array(quantized_signal).astype(float), index=signal.index)

   results = ComputeOptimalTSRFixedSegments(range(min_segments, max_segments+1), time, signal)
   best_xs = []
   best_ys = []
   best_costs = []
   for t in sorted(results.keys()):
      best_xs.append(results[t]['x'])
      best_ys.append(results[t]['y'])
      best_costs.append(results[t]['cost'])

   # Plot the cost function for each segment
   if show_final_plot or tikz_file is not None:
      plt.ion()
      plt.figure()

   # Find the "best" result according to the optimization strategy
   num_segments_tested = max_segments-min_segments+1
   if opt_strategy == "minimum" or num_segments_tested < 3:
      min_cost_idx = np.argmin(best_costs)
   elif opt_strategy == "elbow":
      # Find the elbow by fitting a 2-segment TSR to the cost curve
      segments_series = pd.Series(list(range(min_segments, max_segments+1)))
      costs_series = pd.Series(best_costs, index=list(range(min_segments, max_segments+1)))
      elbow_results = ComputeOptimalTSRFixedSegments(2, segments_series, costs_series)
      if show_final_plot:
         plt.plot(elbow_results[2]['x'], elbow_results[2]['y'], 'g--')
      min_cost_idx = int(elbow_results[2]['x'][1])-min_segments # TODO - add sanity check for elbow-like cost function shape?
   else:
      if opt_strategy != '' or opt_strategy is not None:
         print("Input optimization strategy '%s' not recognized. Defaulting to no optimization.  Separate output files will be written for every number of segments in the desired range."%(opt_strategy))
      min_cost_idx = None
   if min_cost_idx is not None:
      best_x = best_xs[min_cost_idx]
      best_y = best_ys[min_cost_idx]
      best_cost = best_costs[min_cost_idx]
      best_num_segs = min_segments+min_cost_idx
   else:
      best_x = best_xs[-1]
      best_y = best_ys[-1]
      best_cost = best_costs[-1]
      best_num_segs = max_segments

   # Plot the cost function for each segment
   if show_final_plot:
      plt.plot(range(min_segments, max_segments+1), best_costs, 'b-')
      if min_cost_idx is not None:
         plt.plot(best_num_segs, best_costs[min_cost_idx], 'ro')
      plt.title(os.path.basename(input_csv_path)+": Cost function for input range of segments")
      plt.show()

   # Plot results
   num_plot_rows = int(math.ceil(math.sqrt(num_segments_tested)))
   num_plot_cols = int(math.ceil(num_segments_tested/float(num_plot_rows)))
   if show_final_plot or tikz_file is not None:
      if num_segments_tested == 1:
         plt.figure()
         plt.plot(time, signal, 'bo')
         plt.plot(best_xs[0], best_ys[0], 'r-')
         plt.title(os.path.basename(input_csv_path)+": TSR for %d segments"%(min_segments))
      else:
         fig, ax = plt.subplots(num_plot_rows, num_plot_cols)
         plt.title(os.path.basename(input_csv_path)+": TSR for each number of segments")
         segment_idx = 0
         for row_idx in range(num_plot_rows):
            for col_idx in range(num_plot_cols):
               if segment_idx >= num_segments_tested:
                  continue
               ax[row_idx, col_idx].plot(time, signal, 'bo')
               ax[row_idx, col_idx].plot(best_xs[segment_idx], best_ys[segment_idx], 'r-')
               ax[row_idx, col_idx].title.set_text(str(min_segments+segment_idx)+" segment(s)")
               segment_idx += 1
      if tikz_file is not None:
         matplotlib2tikz.save(tikz_file)
      plt.ioff()
      plt.show()

   # Save the "best" result according to the optimization strategy
   if min_cost_idx is not None:
      out_df = pd.DataFrame(data={'Time': best_x, 'Value': best_y})
      out_df.to_csv(output_csv_path, header=True, index=False)
   else:
      for i in range(max_segments-min_segments+1):
         t = min_segments + i
         x = best_xs[i]
         y = best_ys[i]
         output_file_name, output_file_ext = os.path.basename(output_csv_path).split('.')
         output_file_path = os.path.join(os.path.dirname(output_csv_path), output_file_name+'_T%04d'%(t)+'.'+output_file_ext)
         out_df = pd.DataFrame(data={'Time': x, 'Value': y})
         out_df.to_csv(output_file_path, header=True, index=False)

   return best_x, best_y, best_cost, best_num_segs

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input', dest='input_csv', required=True, help='CSV-formatted input signal file with (first column: time, second: signal value)')
   parser.add_argument('--segments', dest='num_segments', required=True, help='Number of segments to use in the approximation. Can be an integer or a range (e.g. 1-10)')
   parser.add_argument('--opt_strategy', dest='opt_strategy', default=None, required=False, help='Strategy for determining which regression to save among the best fit TSRs for the input range of segments to test.  Allowed values are: "elbow" (uses elbow minimization), "minimum" ( pick the TSR with the smallest loss value). If this flag is not set, separate output will be written for each number of segments in the specified range.')
   parser.add_argument('--output', dest='output_csv', required=True, help='Output csv path')
   parser.add_argument('--tikz', dest='tikz', required=False, help='Output path for TikZ PGF plot code') 
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   input_csv_path = args.input_csv
   num_segments = args.num_segments
   if num_segments:
      if '-' in num_segments:
         min_segments = int(num_segments.split('-')[0])
         max_segments = int(num_segments.split('-')[1])
      else:
         min_segments = int(num_segments)
         max_segments = int(num_segments)
   opt_strategy = args.opt_strategy
   tikz_file = args.tikz
   output_csv_path = args.output_csv

   if enable_profiler:
      profile = cProfile.Profile()
      profile.enable()

   ComputeOptimalTSR(input_csv_path, min_segments, max_segments, output_csv_path, opt_strategy=opt_strategy, tikz_file=tikz_file)

   if enable_profiler:
      profile.disable()
      s = io.StringIO()
      ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
      ps.print_stats()
      print(s.getvalue())
