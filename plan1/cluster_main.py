import unittest
import itertools
import sys
import math
import timeit
import copy
import time
import struct
import os.path
import getopt
import h5py

from cluster_map import ClusterMRJob
from mrjob.protocol import PickleProtocol as protocol

    
def print_usage():
        print """    ---------------------------------------------------------------------
    Speaker Diarization in Python with Asp and the GMM Specializer usage:
    ---------------------------------------------------------------------
    Arguments for the diarizer are parsed from a config file. 
    Default config file is diarizer.cfg, but you can pass your own file with the '-c' option. 
    Required is the config file header: [Diarizer] and the options are as follows:
    
    --- Required: ---
    basename: \t Basename of the file to process
    mfcc_feats: \t MFCC input feature file
    output_cluster: \t Output clustering file
    gmm_output: \t Output GMMs parameters file
    M_mfcc: \t Amount of gaussains per model for mfcc
    initial_clusters: Number of initial clusters

    --- Optional: ---
    spnsp_file: \t spnsp file (all features used by default)
    KL_ntop: \t Nuber of combinations to evaluate BIC on
            \t 0 to deactive KL-divergency (fastmatch-component)
    em_iterations: \t Number of iterations for the standard
                  \t segmentation loop training (3 by default)
    num_seg_iters_init: \t Number of majority vote iterations
                        \t in the initialization phase (2 by default)
    num_seg_iters: \t Number of majority vote iterations
                   \t in the main loop (3 by default)
    seg_length: \t Segment length for majority vote in frames
                \t (250 frames by default)

    For fastest performance, enable KL-divergency (KL_ntop = 3) and set
      \t num_seg_iters_init and num_seg_iters to 1
    """

    
def print_no_config():

    print "Please supply a config file with -c 'config_file_name.cfg' "
    return




if __name__ == '__main__':
    mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']    
    meeting_names = [
        'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d',
        'IS1001a', 'IS1001b', 'IS1001c',
        'IS1003b', 'IS1003d', 
        'IS1006b', 'IS1006d',
        'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d'
    ]
    #meeting_names = ['IS1008a', 'IS1008b']
    task_args = [protocol.write(name, None)+"\n" for name in meeting_names]
#        pickle.dump(self.X, open('pickled_args', 'w'))      
#        os.chmod("pickled_args", S_IRUSR | S_IWUSR | S_IXUSR | \
#                                 S_IRGRP | S_IXGRP |           \
#                                 S_IROTH | S_IXOTH             )
    
    start = time.time()
    job = ClusterMRJob(args=mr_args).sandbox(stdin=task_args)
    runner = job.make_runner()        
    runner.run()
#    kv_pairs = map(job.parse_output_line, runner.stream_output())
#    keys = map(lambda(k, v): k, kv_pairs)
#    print "Returned keys:", keys
#    return map(lambda(k, v): v, kv_pairs)
    print "Tasks done. Total execution time:", time.time()-start, "seconds."


