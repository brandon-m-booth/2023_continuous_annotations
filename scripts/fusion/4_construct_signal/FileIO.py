import csv
import pdb
import numpy as np
import pandas as pd

def IsNumeric(obj):
   if hasattr(obj, '__iter__'):
      ret_val = True
      for item in obj:
         try:
            float(item)
            ret_val = ret_val and True
         except ValueError:
            ret_val = False
   else:
      try:
         float(obj)
         ret_val = True
      except ValueError:
         ret_val = False
   return ret_val

def GetCsvData(file_path, first_line_header=True, delimiter=','):
   with open(file_path, 'rb') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=delimiter)
      # Get row and column counts and data type
      row_count = sum(1 for row in csvreader)
      csvfile.seek(0) # Reset file iterator
      csvreader.next() # Skip header row
      first_data_row = csvreader.next()
      col_count = len(first_data_row)
      is_numeric = IsNumeric(first_data_row)
      num_data_rows = row_count-1 if first_line_header else row_count
      if is_numeric:
         data = np.zeros((num_data_rows, len(first_data_row)))
      else:
         data = np.empty((num_data_rows, len(first_data_row)), dtype='S12')

      csvfile.seek(0) # Reset file iterator
      row_num = 0
      header = None if first_line_header else np.array([])
      for row in csvreader:
         if header is None:
            header = np.array(row)
         else:
            if is_numeric:
               data[row_num, :] = np.array(row).astype(float)
            else:
               data[row_num, :] = np.array(row)

            row_num = row_num + 1
               
   return (header, data)

def SaveCsvData(file_path, header, data, delimiter=','):
   with open(file_path, 'wb') as csvfile:
      csvwriter = csv.writer(csvfile, delimiter=delimiter)
      if header is not None:
         csvwriter.writerow(header)
      for row in data:
         csvwriter.writerow(row)

def GetArffData(file_path):
   with open(file_path, 'rb') as arff_file:
      lines = arff_file.readlines()
      data = {}
      feature_names = []
      pass_data = False
      for line in lines:
         if not line.strip(): # If line contains whitespace
            continue

         if '@data' in line:
            pass_data = True
            for feat_name in feature_names:
               data[feat_name] = []
            continue

         if not pass_data:
            if line.startswith('@attribute'):
               line = line.strip().split(' ')
               feature_names.append(line[1])
         else:
            line = line.strip().split(',')
            for i in range(len(line)):
               data[feature_names[i]].append(line[i])

   return pd.DataFrame(data, columns=feature_names)
