import os
import sys
import glob
import argparse
import pandas as pd

def MakeStreamableMovieClips(input_csv, output_clip_path):
   if not os.path.isdir(output_clip_path):
      os.makedirs(output_clip_path)

   df = pd.read_csv(input_csv)

   for idx, clip_data in df.iterrows():
      mp4_file = clip_data[0]
      cut_name = clip_data[1]
      start_time = clip_data[2]
      stop_time = clip_data[3]
      output_file = os.path.join(output_clip_path, os.path.basename(mp4_file).split('.')[0]+'_'+cut_name+'.mp4')
      print('Processing file: '+os.path.basename(mp4_file))
      #out_mp4_file = os.path.join(output_clip_path, os.path.basename(mp4_file))
      start_time_str = "%02d:%02d:%02d"%(start_time/3600, (start_time%3600)/60, start_time%60)
      duration_time_str = "%02d:%02d:%02d"%((stop_time-start_time)/3600, ((stop_time-start_time)%3600)/60, (stop_time-start_time)%60)
      #os.system('ffmpeg -ss %s -i Video.mp4 -ss %s -t %s -c copy VideoClip.mp4'%(start_time_str, start_time_str, duration_time_str))
      #os.system('ffmpeg -i %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(mp4_file, out_mp4_file))
      #print('ffmpeg -ss %s -i %s -ss %s -t %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(start_time_str, mp4_file, start_time_str, duration_time_str, output_file))
      #os.system('ffmpeg -ss %s -i %s -ss %s -t %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(start_time_str, mp4_file, start_time_str, duration_time_str, output_file))
      os.system('ffmpeg -i %s -ss %s -t %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(mp4_file, start_time_str, duration_time_str, output_file))

   return
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_csv', required=True, help='CSV file containing these columns: movie path, cut name, cut start seconds, stop cut seconds')
   parser.add_argument('--output_clip_path', required=True, help='Output path for streamable movie clips')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   MakeStreamableMovieClips(args.input_csv, args.output_clip_path)
