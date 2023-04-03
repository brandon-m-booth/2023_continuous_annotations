import os
import glob
import sys

mp4_files = glob.glob('/media/brandon/SailData/2019_Continuous_Annotations/datasets/movie_violence/movies/*.mp4')
out_path = '/media/brandon/SailData/2019_Continuous_Annotations/datasets/movie_violence/movies2/'
for mp4_file in mp4_files:
   if 'cut' in os.path.basename(mp4_file):
      print('Processing file: '+os.path.basename(mp4_file))
      out_mp4_file = os.path.join(out_path, os.path.basename(mp4_file))
      os.system('ffmpeg -i %s -c:v libx264 -crf 23 -c:a libvo_aacenc -ac 2 -movflags faststart %s'%(mp4_file, out_mp4_file))
