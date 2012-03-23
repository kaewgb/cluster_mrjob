import numpy as np
import scipy.stats.mstats as stats

import os
import cPickle as pickle

from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol

from cluster_trainmrjob import TrainMRJob
from cluster_sgmtmrjob import SegmentMRJob
from cluster_bicmrjob import BICMRJob

class NaivePythonMR:
    def __init__(self, em_iters, X, gmm_list):
        self.em_iters = em_iters
        self.X = X
        self.gmm_list = gmm_list
    
    def train_map(self, init_training):
        g, x = init_training;
        g.train(x, max_em_iters=self.em_iters)

    def train_using_mapreduce(self, init_training, em_iters):
        mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']    
        input = []
        count = 0
        for pair in init_training:
            input.append((count, pair, em_iters))
            count = count+1
        task_args = [protocol.write(pair, None)+"\n" for pair in input]
#        pickle.dump(self.X, open('pickled_args', 'w'))      
#        os.chmod("pickled_args", S_IRUSR | S_IWUSR | S_IXUSR | \
#                                 S_IRGRP | S_IXGRP |           \
#                                 S_IROTH | S_IXOTH             )
        
        job = TrainMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()        
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        keys = map(lambda(k, v): k, kv_pairs)
        print "Returned keys:", keys
        return map(lambda(k, v): v, kv_pairs)  
        
        
    def segment_map(self, iter_item):
        gp, data_list = iter_item
        g = gp[0]
        p = gp[1]
        cluster_data =  data_list[0]

        for d in data_list[1:]:
            cluster_data = np.concatenate((cluster_data, d))

        g.train(cluster_data, max_em_iters=self.em_iters)
        return (g, p, cluster_data)

    def segment_reduce(self, x, y):
        iter_bic_dict, iter_bic_list = x
        g, p, cluster_data = y
        iter_bic_list.append((g, cluster_data))
        iter_bic_dict[p] = cluster_data
        return (iter_bic_dict, iter_bic_list)
    
    def segment_using_mapreduce(self, iter_item, em_iters):
        mr_args = ['-v', '--strict-protocols', '-r', 'local','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
        input = []
        count = 0
        for pair in iter_item:
            input.append((count, pair, em_iters))
        task_args = [protocol.write(pair, None)+"\n" for pair in iter_item]
        job = SegmentMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()
        runner.run()
        
    def score_map(self, g):
        return g.score(self.X)
    
    def score_reduce(self, x, y):
        x = np.column_stack((x, y))
        return x
   
    def vote_map(self, map_input):
        arr, X = map_input
        return (int(stats.mode(arr)[0][0]), X)
    
    def vote_reduce(self, iter_training, item):
        max_gmm, X = item
        iter_training.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append(X)
        return iter_training
    
    def bic_using_mapreduce(self, iteration_bic_list, em_iters):
        mr_args = ['-v', '--strict-protocols', '-r', 'local','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
        input = []
        l = len(iteration_bic_list)
        for gmm1idx in range(l):
            for gmm2idx in range(gmm1idx+1, l):
                g1, d1 = iteration_bic_list[gmm1idx]
                g2, d2 = iteration_bic_list[gmm2idx]
                data = np.concatenate((d1,d2))
                an_item = protocol.write((gmm1idx,gmm2idx),(g1, g2, data, em_iters))
                input.append(an_item+"\n")     
    
        job = BICMRJob(args=mr_args).sandbox(stdin=input)
        runner = job.make_runner()
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        assert len(kv_pairs) == 1
        best_merged_gmm, merged_tuple, merged_tuple_indices, best_score = kv_pairs[0][1]
        
        ind1, ind2 = merged_tuple_indices
        g1, d1 = iteration_bic_list[ind1]
        g2, d2 = iteration_bic_list[ind2]
        data = np.concatenate((d1,d2))
        new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
            
        return new_gmm, (g1, g2), merged_tuple_indices, best_score      
 