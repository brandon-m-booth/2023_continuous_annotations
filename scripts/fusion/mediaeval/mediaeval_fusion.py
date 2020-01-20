#!/usr/bin/env python3

import os
import sys
import pdb
import glob
import math
import signal
import argparse
import subprocess
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'fusion', '1_tsr')))
import compute_tsr as tsr 

# User variables
show_verbose_output = False
subject_ids =  ['1761','1585','629','1322','536','1537','882','1428','1952','1823','1662','934','939','1149','1749','244','614','1518','1978','926','1633','1401','1690','302','988','1280','1676','1821','451','1315','232','1100','1989','1963','739','732','959','1682','961','1692','1509','399','1626','527','863','1887','343','613','1669','1894','1822','507','1716','1559','1008','400','110','1471','1685','1340','345','1384','1408','1132','71','781','1615','608','1228','387','949','1326','1474','1388','1836','369','833','1871','848','1157','404','365','403','605','672','1053','1090','1367','448','386','1396','453','790','667','1006','1185','1720','242','107','124','40','1111','276','649','577','846','314','485','1652','688','1663','1842','512','1258','1070','428','1860','51','893','1602','1893','686','1030','1987','274','494','1565','1927','346','1343','1944','179','4','779','1817','101','76','1746','664','808','674','1254','1398','784','265','1334','1050','591','406','1273','1991','1381','1217','88','719','1712','1540','1018','289','1087','1074','1827','1163','1196','250','236','960','1905','1947','944','1215','704','303','1600','1942','1282','1764','131','1036','178','1864','1232','456','301','1628','1778','524','856','109','375','460','336','1161','1094','1846','938','1177','1369','159','191','714','156','234','115','1888','1208','621','1504','1632','1423','1743','1814','459','824','1314','1207','1790','199','1943','708','931','1881','3','884','1569','1974','1678','913','1993','1122','1610','1497','220','1170','326','1182','1355','1782','983','170','311','1697','1223','1037','1741','430','1325','86','99','1153','1005','151','1441','777','402','1067','1969','648','1955','1473','970','1339','383','82','1168','273','997','921','1945','300','1560','1178','67','638','789','1416','1896','1220','1302','503','490','693','1553','1957','1152','358','819','1151','1857','1801','146','1665','1870','1205','1318','684','1728','1044','1647','1065','530','962','1844','1071','663','887','262','134','278','850','1136','1353','1173','711','319','1137','1057','1571','1022','54','1033','127','1826','1468','1863','671','1695','96','371','1420','1391','1016','1126','1346','980','1436','1683','350','640','248','153','722','1069','175','1459','796','1268','401','696','1088','1640','1307','1274','1323','390','1304','1024','1895','1654','1221','1250','1650','1121','631','1637','820','1802','190','1243','757','1484','736','425','911','1493','1366','1586','1994','676','952','1393','1726','1145','801','467','685','39','1386','1968','1251','1212','1765','13','24','990','1101','307','464','329','1623','675','362','452','431','617','1756','341','1284','1520','1359','1635','1289','1990','1031','1142','1536','473','1116','584','253','435','568','1533','192','1505','1478','2','1772','1816','506','225'] # The 431 annotations with the highest agreement
#subject_ids = [str(x) for x in range(2001,2059)] # 58 test annotation for full songs
#skip_subj_ids = ['2055','2056','2057','2058']
skip_subj_ids = []

# Global variables
pool = None

def cmp_file_names(x, y):
   x_number_str = os.path.basename(x).split('.')[0]
   y_number_str = os.path.basename(y).split('.')[0]
   if int(x_number_str) < int(y_number_str):
      return -1
   else:
      return 1

def ProcessMediaevalTask(cur_dimension, anno_files_path, target_subject_ids, features_dict):
   print("Processing %s annotations for subject(s):%s"%(cur_dimension, str(target_subject_ids)))
   anno_files = glob.glob(os.path.join(anno_files_path, '*.csv'))

   results_path = os.path.join('..','..','..','results','mediaeval',cur_dimension)

   # Variables for custom minor steps
   step_1_1_df = None

   # Compute annotator agreement
   fusion_dict = {}
   for anno_file_idx in range(len(anno_files)):
      anno_file = anno_files[anno_file_idx]
      task_name = os.path.basename(anno_file)
      if not task_name in features_dict.keys():
         print("Unable to find features for annotation task: %s"%(task_name))
         continue

      # Only process the requested subject ids
      subject_id = task_name.split('.')[0]
      if not subject_id in target_subject_ids:
         continue

      # Read and format annotation data
      anno_df = pd.read_csv(anno_file)
      if 'WorkerId' in anno_df.columns:
         anno_df = anno_df.drop(columns='WorkerId')
      num_annos = anno_df.shape[0]
      times_str = [x[7:-2] for x in anno_df.columns]
      times_sec = [float(t)/1000.0 for t in times_str]
      anno_df = anno_df.T
      anno_df.index = times_sec
      anno_df.columns = ['A%0d'%(i) for i in range(anno_df.shape[1])]

      # Read and format the features data
      feature_df = pd.read_csv(features_dict[task_name], sep=';')
      feature_df = feature_df.set_index('frameTime')

      # Fusion pipeline variables
      matlab_cmd = 'matlab -nodisplay -nosplash -nodesktop -r '
      anno_subj_regexp = '(\d+).csv'
      fusion_script_path = os.path.join('..','fusion')

      # Step 0: time align the annotations
      time_align_path = os.path.join(fusion_script_path, '0_time_alignment/')
      rel_results_path = os.path.join( '..', results_path)
      out_dir_0 = '0_alignment/'
      results_time_align_path = os.path.join(rel_results_path, out_dir_0)
      if not os.path.isdir(os.path.join(results_path, out_dir_0)):
         os.makedirs(os.path.join(results_path, out_dir_0))
      time_align_cmd = 'estimate_lags_mediaeval(\''+anno_file+'\', \''+features_dict[task_name]+'\', \''+anno_subj_regexp+'\', \''+results_time_align_path+'\')'
      cmd = 'cd '+time_align_path+'; '+matlab_cmd+'"'+time_align_cmd+';quit"'
      if show_verbose_output:
         print("======\nRunning command:\n======")
         print(cmd)
         print("======\nOutput:\n======")
      proc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
      if show_verbose_output:
         print(proc.stdout)

      # Step 0.5: Average the time-aligned annotations (when not using TSS)
      out_dir_0_5 = '0.5_average/'
      out_file_0_5 = subject_id+'_average.csv'
      if not os.path.isdir(os.path.join(results_path, out_dir_0_5)):
         os.makedirs(os.path.join(results_path, out_dir_0_5))
      aligned_anno_files = glob.glob(os.path.join(results_path, out_dir_0, subject_id+'_ann*.csv'))
      combined_df = None
      for aligned_anno_file in aligned_anno_files:
         df = pd.read_csv(aligned_anno_file)
         new_col_name = os.path.basename(aligned_anno_file)
         df = df.rename(columns={'Data': new_col_name})
         if combined_df is None:
            combined_df = df
         else:
            combined_df = pd.concat((combined_df, df[new_col_name]), axis=1)
      combined_df['Data_average'] = combined_df.iloc[:,1:].mean(axis=1)
      average_df = combined_df.reindex(columns=('Time_seconds', 'Data_average'))
      average_df = average_df.rename(columns={'Data_average': 'Data'})
      average_df.to_csv(os.path.join(results_path, out_dir_0_5, out_file_0_5), header=True, index=False)

      # Step 1: Compute the TSR approximation; elbow optimize the number of segments
      out_dir_1 = '1_tsr/'
      out_file_1 = subject_id+'_tsr.csv'
      if not os.path.isdir(os.path.join(results_path, out_dir_1)):
         os.makedirs(os.path.join(results_path, out_dir_1))
      average_signal_csv = os.path.join(results_path, out_dir_0_5, out_file_0_5)
      out_file_path_1 = os.path.join(results_path, out_dir_1, out_file_1)
      min_segments = 1
      max_segments = 40
      #opt_strategy = 'elbow'
      opt_strategy = None
      best_x, best_y, best_cost, best_num_segs = tsr.ComputeOptimalTSR(average_signal_csv, min_segments, max_segments, out_file_path_1, opt_strategy=opt_strategy)

      # Step 1.1: Save the TSR information for each annotation
      if opt_strategy is not None:
         out_dir_1_1 = '1.1_tsr_info/'
         out_file_1_1 = 'tsr_data.csv'
         out_file_1_1_path = os.path.join(results_path, out_dir_1_1, out_file_1_1)
         if not os.path.isdir(os.path.join(results_path, out_dir_1_1)):
            os.makedirs(os.path.join(results_path, out_dir_1_1))
         if step_1_1_df is None:
            if os.path.exists(out_file_1_1_path):
               step_1_1_df = pd.read_csv(out_file_1_1_path)
            else:
               step_1_1_df = pd.DataFrame(None, columns=['id', 'opt_num_segments', 'opt_cost'])
         if int(subject_id) in step_1_1_df['id'].values:
            row_mask = step_1_1_df['id'] == int(subject_id)
            step_1_1_df.loc[row_mask, 'opt_num_segments'] = best_num_segs
            step_1_1_df.loc[row_mask, 'opt_cost'] = best_cost
         else:
            append_idx = step_1_1_df.shape[0]
            step_1_1_df.loc[append_idx] = [0,0,0]
            step_1_1_df.loc[append_idx, 'id'] = int(subject_id)
            step_1_1_df.loc[append_idx, 'opt_num_segments'] = best_num_segs
            step_1_1_df.loc[append_idx, 'opt_cost'] = best_cost
         step_1_1_df = step_1_1_df.sort_values(by=['opt_cost'], ascending=False)
         step_1_1_df.to_csv(os.path.join(results_path, out_dir_1_1, out_file_1_1), header=True, index=False)

      # TODO
      fused_signal = None
      fusion_dict[task_name] = fused_signal

   # TODO - Save the fused signals
   #anno_agree_df.to_csv(os.path.join(cur_file_path, output_file_path), index=False)

def ProcessMediaevalTaskArgs(args):
   ProcessMediaevalTask(*args)

def Terminate(signum, frame):
   global pool
   if pool is not None:
      pool.terminate()

def MediaevalFusion(max_jobs):
   global pool

   cur_file_path = os.path.dirname(os.path.realpath(__file__))
   mediaeval_anno_path = os.path.join(cur_file_path, '..', '..', '..', 'datasets', 'mediaeval', 'annotations', 'annotations per each rater', 'dynamic (per second annotations)')
   mediaeval_features_path = os.path.join(cur_file_path, '..', '..', '..', 'datasets', 'mediaeval', 'features')
   arousal_anno_path = os.path.join(mediaeval_anno_path, 'arousal')
   valence_anno_path = os.path.join(mediaeval_anno_path, 'valence')

   # Create dictionary of features files
   features_files = glob.glob(os.path.join(mediaeval_features_path, '*.csv'))
   features_dict = {}
   for features_file in features_files:
      features_dict[os.path.basename(features_file)] = features_file

   task_args = []
   for anno_dimension in [('arousal', arousal_anno_path), ('valence', valence_anno_path)]:
      for subject_id in subject_ids:
         if subject_id not in skip_subj_ids:
            task_args.append((anno_dimension[0], anno_dimension[1], [subject_id], features_dict))

   # Setup graceful shutdown callbacks
   signal.signal(signal.SIGINT, Terminate)
   signal.signal(signal.SIGINT, Terminate)

   # Process task files in parallel
   pool = Pool(max_jobs)
   results = pool.map(ProcessMediaevalTaskArgs, task_args)
   pool.terminate()

   return

if __name__=='__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--max_jobs', dest='max_jobs', required=False, help='Maximum number of jobs for task processing')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   max_jobs = 8 if not args.max_jobs else int(args.max_jobs)
   MediaevalFusion(max_jobs)
