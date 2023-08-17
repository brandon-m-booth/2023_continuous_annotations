import cvxopt
import numpy as np
import pandas as pd

zero_slope_tol = 1e-7
zero_value_tol = 1e-10

# Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
# See: https://github.com/nschloe/tikzplotlib/issues/557
def TikzplotlibFixNCols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        TikzplotlibFixNCols(child)

def GetUpperTri(corr_mat):
   m = corr_mat.shape[0]
   r,c = np.triu_indices(m,1)
   upper_tri_vals = corr_mat[r,c].flatten()
   return upper_tri_vals

# Columns are individual signals and rows are frames
def NormedDiff(df, tol=0.005):
   diff_df = df.diff(axis=0)
   diff_df.iloc[0,:] = 0 # Assume zero diff for first row
   diff_df[np.abs(diff_df) <= tol] = 0
   norm_diff_df = np.sign(diff_df)
   return norm_diff_df

# Columns are individual signals and rows are frames
def AccumNormedDiff(df, tol=0.005):
   norm_diff_df = NormedDiff(df, tol=tol)
   acc_norm_diff_df = norm_diff_df.cumsum(axis=0, skipna=True)
   return acc_norm_diff_df

def GetTSRSumSquareError(tsr_df, signal_df):
   signal_index = signal_df.index
   tsr_series = tsr_df.iloc[:,0]
   new_tsr_index = np.unique(tsr_series.index.tolist()+signal_index.tolist())
   try:
      tsr_series = tsr_series.reindex(new_tsr_index, method=None, fill_value=np.nan)
   except ValueError: # Can happen if the tsr_series has duplicate values in the index
      tsr_series = tsr_series.groupby(tsr_series.index).first() # Remove duplicate index entries
      tsr_series = tsr_series.reindex(new_tsr_index, method=None, fill_value=np.nan)
   tsr_series = tsr_series.interpolate(method='slinear')
   diff = tsr_series[signal_index] - signal_df.iloc[:,0]
   sse = diff.dot(diff).sum()
   return sse

def GetCVXOPTMatrix(M):
   if M is None:
      return None
   elif type(M) is np.ndarray:
      return cvxopt.matrix(M)
   elif type(M) is cvxopt.matrix or type(M) is cvxopt.spmatrix:
      return M
   coo = M.tocoo()
   return cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)

def FitConstantSegment(signal):
   if len(signal) > 0:
      b = np.mean(signal)
      residuals = signal - b*np.ones(len(signal))
      loss_value = np.dot(residuals,residuals)
   else:
      b = np.nan
      loss_value = np.inf
   return b, loss_value

def FitLineSegment(signal):
   if len(signal) > 1:
      M = np.vstack((signal.index, np.ones(len(signal)))).T
      coefs, residuals, rank, singular_vals = np.linalg.lstsq(M, signal.values, rcond=None)
      a,b = coefs
      residuals = (a*signal.index + b)-signal.values
      if len(residuals) > 0:
         loss_value = np.dot(residuals, residuals)
      else:
         loss_value = 0.0
   elif len(signal) == 1:
      a = 0.0
      b = signal.iloc[0]
      residuals = signal.iloc[1:] - b*np.ones(len(signal)-1)
      loss_value = np.dot(residuals,residuals)
   else:
      a = np.nan
      b = np.nan
      loss_value = np.inf
   return a, b, loss_value 

def FitConstantSegmentWithIntersection(signal, i, j, a, b, x1, x2):
   if np.isnan(a) or np.isnan(b):
      return np.nan, np.nan, np.inf

   x = np.array(signal.iloc[i:j+1].index)
   y = np.array(signal.iloc[i:j+1])
   P = np.array([len(x)]).astype(float)
   q = np.array([-np.sum(y)]).astype(float)
   G = np.array([[1], [-1]]).astype(float)
   h = np.array([a*x2+b, -a*x1-b]).reshape(-1,1).astype(float)
   if a < 0.0:
      G = -G
      h = -h

   P = GetCVXOPTMatrix(P)
   q = GetCVXOPTMatrix(q)
   G = GetCVXOPTMatrix(G)
   h = GetCVXOPTMatrix(h)
   cvxopt.solvers.options['maxiters'] = 300
   cvxopt.solvers.options['abstol'] = 1e-12
   cvxopt.solvers.options['reltol'] = 1e-12
   cvxopt.solvers.options['feastol'] = 1e-12
   cvxopt.solvers.options['show_progress'] = False
   cvx_solution = cvxopt.solvers.coneqp(P=P, q=q, G=G, h=h, kktsolver='ldl')
   loss_value = np.nan
   if 'optimal' in cvx_solution['status']:
      loss_value = cvx_solution['primal objective']
      signal_fit = np.array(cvx_solution['x']).reshape((len(q),))
   else:
      print("Warning: CVXOPT did not find an optimal solution")

   loss_value += 0.5*np.dot(y,y)
   loss_value *= 2 # The QP minimizes 1/2 SSE, so double it here
   if loss_value < 0.0: # Handle numerical precision issues
      loss_value = 0.0
   if a == 0.0:
      x = (x1+x2)/2.0
   else:
      x = (signal_fit[0]-b)/a

   # Ensure boundaries are satisfied.  The solver isn't exact, but close.
   if x > x2 or x < x1:
      # Perturb 'b' to enforce boundary conditions
      if abs(x-x1) < abs(x-x2):
         signal_fit[0] = a*x1 + b
         x = x1
      else:
         signal_fit[0] = a*x2 + b
         x = x2

   return signal_fit[0], x, loss_value

def FitLineSegmentWithIntersection(signal, i, j, a, b, x1, x2):
   if np.isnan(a) or np.isnan(b):
      return np.nan, np.nan, np.nan, np.inf

   # Special case: if signal.iloc[i:j+1] has one point, then there is no unique optimal
   # solution. Pick the line going through a*(x1+x2)/2+b and signal.iloc[i].
   if i == j:
      x_point = (x1+x2)/2.0
      denom = signal.index[i]-x_point
      is_zero_denom = abs(denom) < zero_value_tol
      if not is_zero_denom:
         new_a = (signal.iloc[i]-(a*x_point+b))/denom
         new_b = signal.iloc[i] - new_a*signal.index[i]
      loss_value = 0.0
      if is_zero_denom or (a == 0.0 and abs(new_a) < zero_slope_tol):
         x_point = (x1+x2)/2.0 # If the lines are parallel, use a valid x value
         new_a = 0.0 # This can only happen if the lines are constant, so force it to be so
         new_b = b
      return new_a, new_b, x_point, loss_value

   x = np.array(signal.index[i:j+1])
   y = np.array(signal.iloc[i:j+1])
   x_squared = np.square(x)
   P = np.array([[np.sum(x_squared), np.sum(x)],[np.sum(x), len(x)]]).astype(float)
   q = np.array([-np.dot(x,y), -np.sum(y)]).astype(float)
   G1 = np.array([[-x2, -1], [x1, 1], [-1, 0]]).astype(float)
   h1 = np.array([-a*x2-b, a*x1+b, -a]).reshape(-1,1).astype(float)
   G2 = -G1
   h2 = -h1

   P = GetCVXOPTMatrix(P)
   q = GetCVXOPTMatrix(q)
   G1 = GetCVXOPTMatrix(G1)
   h1 = GetCVXOPTMatrix(h1)
   G2 = GetCVXOPTMatrix(G2)
   h2 = GetCVXOPTMatrix(h2)
   cvxopt.solvers.options['maxiters'] = 300
   cvxopt.solvers.options['abstol'] = 1e-12
   cvxopt.solvers.options['reltol'] = 1e-12
   cvxopt.solvers.options['feastol'] = 1e-12
   cvxopt.solvers.options['show_progress'] = False
   near_flat_line_error = False
   try:
      cvx_solution1 = cvxopt.solvers.coneqp(P=P, q=q, G=G1, h=h1, kktsolver='ldl')
      cvx_solution2 = cvxopt.solvers.coneqp(P=P, q=q, G=G2, h=h2, kktsolver='ldl')
      loss_value1 = cvx_solution1['primal objective'] 
      loss_value2 = cvx_solution2['primal objective']
      if 'optimal' in cvx_solution1['status']:
         if 'optimal' in cvx_solution2['status'] and loss_value2 < loss_value1:
            loss_value = loss_value2
            signal_fit = np.array(cvx_solution2['x']).reshape((len(q),))
         else:
            loss_value = loss_value1
            signal_fit = np.array(cvx_solution1['x']).reshape((len(q),))
      elif 'optimal' in cvx_solution2['status']:
         loss_value = loss_value2
         signal_fit = np.array(cvx_solution2['x']).reshape((len(q),))
      else:
         print("Warning: CVXOPT did not find an optimal solution")
         loss_value = None
   except ZeroDivisionError:
      near_flat_line_error = True
      loss_value = 0.0
      signal_fit = np.zeros(2)

   loss_value += 0.5*np.dot(y,y)
   loss_value *= 2 # The QP minimizes 1/2 SSE, so double it here
   if loss_value < 0.0: # Handle numerical precision issues
      loss_value = 0.0
   if a == 0.0 and (near_flat_line_error or abs(signal_fit[0]) < zero_slope_tol):
      x = (x1+x2)/2.0 # If the lines are parallel, use a valid x value
      signal_fit[0] = 0.0 # This can only happen if the lines are constant, so force it to be so
      signal_fit[1] = b
      loss_value = np.dot(y-b, y-b)
   else:
      x = (signal_fit[1]-b)/(a-signal_fit[0])

   # Ensure boundaries are satisfied.  The solver isn't exact, but close.
   if x > x2 or x < x1:
      # Perturb 'b' to enforce boundary conditions
      if abs(x-x1) < abs(x-x2):
         signal_fit[1] = (a*x1 + b) - signal_fit[0]*x1
         x = x1
      else:
         signal_fit[1] = (a*x2 + b) - signal_fit[0]*x2
         x = x2

   return signal_fit[0], signal_fit[1], x, loss_value
