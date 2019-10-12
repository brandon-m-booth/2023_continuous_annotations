#!/usr/bin/env python

import os
import sys
import pdb
from multiprocessing import Pool

def ComputeForNumSegs(segments):
   if not hasattr(segments, '__iter__'):
      segments = [segments]
   for num_segs in segments:
      print("Computing optimal TSR for %d segments out of %d"%(num_segs, len(segments)))
      os.system('python continuous_opt.py --input /USC/2016_Continuous_Annotations/annotation_tasks/TaskB/AnnotationData/ground_truth_baselines/eval_dep/eval_dep_ground_truth_1hz.csv --segments %d --output /USC/2018_Continuous_Annotations/data/GreenIntensityTasks/tsr/taskB_opt_trapezoid_%d_segments_10hz.csv'%(num_segs, num_segs))
   return

def DoThing():
   pool = Pool(7)
   segments = list(range(3,101))
   results = pool.map(ComputeForNumSegs, segments)

DoThing()
