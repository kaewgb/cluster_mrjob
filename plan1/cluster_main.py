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
    meeting_names = ['HVC006045', 'HVC134406', 'HVC319600', 'HVC486704', 'HVC605240', 'HVC730081', 'HVC006184', 'HVC134634', 'HVC320560', 'HVC498099', 'HVC606315', 'HVC733376', 'HVC011409', 'HVC146032', 'HVC334254', 'HVC499900', 'HVC615626', 'HVC742499', 'HVC022974', 'HVC148793', 'HVC359216', 'HVC506183', 'HVC616948', 'HVC749106', 'HVC026971', 'HVC152448', 'HVC362146', 'HVC507955', 'HVC620320', 'HVC766498', 'HVC027850', 'HVC169936', 'HVC364230', 'HVC513054', 'HVC631691', 'HVC776608', 'HVC029485', 'HVC174999', 'HVC365963', 'HVC515040', 'HVC637907', 'HVC786536', 'HVC031151', 'HVC198184', 'HVC377296', 'HVC516366', 'HVC646345', 'HVC788532', 'HVC032158', 'HVC201655', 'HVC396658', 'HVC520461', 'HVC652043', 'HVC789034', 'HVC036790', 'HVC203988', 'HVC397279', 'HVC528762', 'HVC658517', 'HVC792251', 'HVC040785', 'HVC218344', 'HVC398674', 'HVC528929', 'HVC672564', 'HVC803719', 'HVC042692', 'HVC218765', 'HVC402970', 'HVC529613', 'HVC676565', 'HVC811971', 'HVC049437', 'HVC240490', 'HVC403628', 'HVC532992', 'HVC680956', 'HVC822060', 'HVC064215', 'HVC243157', 'HVC417926', 'HVC532993', 'HVC681821', 'HVC823377', 'HVC067270', 'HVC255452', 'HVC418407', 'HVC539647', 'HVC682794', 'HVC823657', 'HVC068504', 'HVC257987', 'HVC425927', 'HVC541506', 'HVC683835', 'HVC829585', 'HVC087057', 'HVC259599', 'HVC429818', 'HVC542481', 'HVC687278', 'HVC834162', 'HVC090414', 'HVC263724', 'HVC434449', 'HVC549861', 'HVC690639', 'HVC836424', 'HVC091740', 'HVC268521', 'HVC450989', 'HVC553302', 'HVC699748', 'HVC846668', 'HVC095303', 'HVC271903', 'HVC460343', 'HVC561733', 'HVC705168', 'HVC878334', 'HVC098807', 'HVC283780', 'HVC462262', 'HVC562777', 'HVC707236', 'HVC881736', 'HVC103302', 'HVC292292', 'HVC462508', 'HVC570651', 'HVC711253', 'HVC883718', 'HVC103932', 'HVC293678', 'HVC463620', 'HVC573643', 'HVC717615', 'HVC887091', 'HVC105766', 'HVC298568', 'HVC467645', 'HVC579741', 'HVC718274', 'HVC888814', 'HVC108320', 'HVC309504', 'HVC468477', 'HVC591147', 'HVC719228', 'HVC891523', 'HVC109456', 'HVC317804', 'HVC473973', 'HVC597104', 'HVC726373', 'HVC123288', 'HVC319297', 'HVC479361', 'HVC602688', 'HVC730049']    
#    meeting_names = [
#        'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d',
#        'IS1001a', 'IS1001b', 'IS1001c',
#        'IS1003b', 'IS1003d', 
#        'IS1006b', 'IS1006d',
#        'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d'
#    ]
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


