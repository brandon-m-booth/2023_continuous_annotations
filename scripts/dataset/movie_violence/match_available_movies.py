#!/usr/bin/env python3
#Author: Brandon M Booth

import os
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz

def MatchAvailableMovies(available_movies_path, match_csv_path, match_csv_column, output_path):
   available_movies_df = pd.read_csv(available_movies_path, header=None)
   available_movies_path = available_movies_df.iloc[:,0].tolist()
   available_movies = [os.path.basename(x) for x in available_movies_path]
   available_movies = [x.split('.')[0] for x in available_movies]
   available_movies = [x.replace('_', ' ') for x in available_movies]

   match_df = pd.read_csv(match_csv_path)
   if not match_csv_column in match_df.columns:
      print("Could not find column '%s' in %s"%(match_csv_column, match_csv_path))
      return
   match_titles = match_df[match_csv_column].tolist()

   out_match_col_name = match_csv_column+'_matched'
   out_match_path_col_name = out_match_col_name+'_path'
   match_df[out_match_col_name] = match_df.shape[0]*['']
   match_df[out_match_path_col_name] = match_df.shape[0]*['']

   # Fuzzy match 
   for i in tqdm(range(match_df.shape[0])):
      match_title = match_titles[i]
      best_match_ratio = -np.inf
      best_match_idx = np.nan
      for available_idx in range(len(available_movies)):
         available_movie = available_movies[available_idx]
         match_ratio = fuzz.ratio(match_title, available_movie)
         if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_match_idx = available_idx

      if best_match_ratio > 80:
         match_df.loc[i, out_match_col_name] = available_movies[best_match_idx]
         match_df.loc[i, out_match_path_col_name] = available_movies_path[best_match_idx]
         #print("Found match (%d): %s <=> %s"%(i, match_title, available_movies[best_match_idx]))

   match_csv_col_idx = match_df.columns.tolist().index(match_csv_column)
   new_column_order = match_df.columns.tolist()[:-2]
   new_column_order.insert(match_csv_col_idx, out_match_col_name)
   new_column_order.insert(match_csv_col_idx, out_match_path_col_name)
   match_df = match_df[new_column_order]
   match_df.to_csv(output_path, index=False, header=True)
   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--movie_list', required=True, help='Path to file containing a list of available movies.  The list can contain paths to movies; only the basename is used')
   parser.add_argument('--match_csv', required=True, help='Path to CSV file to match against')
   parser.add_argument('--match_csv_column', required=True, help='Column heading of the movie title in the csv to match against')
   parser.add_argument('--output', required=True, help='Output CSV with matched column')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   MatchAvailableMovies(args.movie_list, args.match_csv, args.match_csv_column, args.output)
