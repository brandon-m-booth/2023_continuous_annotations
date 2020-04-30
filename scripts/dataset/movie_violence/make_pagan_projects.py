#! /usr/bin/env python
# This script outputs SQL commands and anonymizes the names of movie files
import os
import sys
import glob
import uuid
import shutil
import argparse

def MakePaganProjectsSQL(movie_clip_folder, output_anon_movie_folder, output_sql_file):
   movie_clip_files = glob.glob(os.path.join(movie_clip_folder, '*.mp4'))
   pagan_user = "brandon"

   if not os.path.isdir(output_anon_movie_folder):
      os.makedirs(output_anon_movie_folder)
   if not os.path.isdir(os.path.dirname(output_sql_file)):
      os.makedirs(os.path.dirname(output_sql_file))

   out_sql = open(output_sql_file, 'wb')

   # Print SQL instructions to create a separate project entry for each movie clip
   for i in range(len(movie_clip_files)):
      movie_clip_file = movie_clip_files[i]
      proj_id = str(uuid.uuid4()).upper()
      proj_name = 'Movie Violence Rating Experiment #'+str(i)
      annotation_dim = 'violence'
      annotation_type = 'bounded'
      video_sequencing = 'random'
      endless_mode = 'off'
      num_entries = 1 # number of movie clip files in the project
      num_participant_runs = 1
      start_message = 'Please annotate the level of violence as you perceive it in real time as you watch the video.  The cursor starts at the bottom of the scale (no violence). The top of the scale represents extreme violence. There may be no violence depicted in this clip.'
      upload_message = ''
      end_message = 'Your annotation has been saved. Please share your thoughts about this annotation experiment in the anonymous exit survey. You will receive a survey code to complete your M-Turk HIT once the survey is complete.'
      survey_link = 'https://docs.google.com/forms/d/e/1FAIpQLScNQERvASo53qdTfVzfDZgkmwy62h23dyARjqNzIA11MwAEsQ/viewform?usp=sf_link'
      sound_enabled = 'on'
      values_list = [pagan_user, proj_id, proj_name, annotation_dim, annotation_type, 'upload', video_sequencing, endless_mode, num_entries, num_participant_runs, end_message, survey_link, sound_enabled, start_message, 'false']
      sql_values = []
      for value in values_list:
         if isinstance(value, str):
            sql_values.append('"'+value+'"')
         else:
            sql_values.append(str(value))

      out_sql.write("INSERT INTO projects (username, project_id, project_name, target, type, source_type, video_loading, endless, n_of_entries, n_of_participant_runs, end_message, survey_link, sound, start_message, archived) VALUES ("+','.join(sql_values)+");\n")

      source_url = os.path.join('user_uploads', proj_id+'.mp4')
      original_name = os.path.basename(movie_clip_file).split('.')[0]
      values_list2 = [proj_id, 1, 'upload', source_url, original_name]
      sql_values2 = []
      for value2 in values_list2:
         if isinstance(value2, str):
            sql_values2.append('"'+value2+'"')
         else:
            sql_values2.append(str(value2))
      out_sql.write("INSERT INTO project_entries (project_id, entry_id, source_type, source_url, original_name) VALUES ("+','.join(sql_values2)+");\n")

      shutil.copyfile(movie_clip_file, os.path.join(output_anon_movie_folder, proj_id+'.mp4'))

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--movie_clip_folder', required=True, help='Path to folder containing a movie clips')
   parser.add_argument('--output_anon_movie_folder', required=True, help='Output path for the movie clips with anonymized names')
   parser.add_argument('--output_sql_file', required=True, help='Path to output file for SQL commands')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   MakePaganProjectsSQL(args.movie_clip_folder, args.output_anon_movie_folder, args.output_sql_file)
