import numpy as np
import scipy.stats.mstats as stats

import os
import cPickle as pickle
from stat import *

from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol

from cluster_trainmrjob import TrainMRJob
from cluster_scoremrjob import ScoreMRJob
from cluster_sgmtmrjob import SegmentMRJob
from cluster_bicmrjob import BICMRJob

class MRhelper:
    def __init__(self, em_iters, X, gmm_list):
        #For Naive Python
        self.em_iters = em_iters
        self.X = X
        self.gmm_list = gmm_list
        
        #For mrjob
        pickle.dump(self.X, open('self_X', 'w'))
        os.chmod("self_X", S_IRUSR | S_IWUSR | S_IXUSR | \
                                 S_IRGRP | S_IXGRP |           \
                                 S_IROTH | S_IXOTH             )
        self.Xfilename = 'self_X' 
        
    def train_map(self, init_training):
        g, start, interval = init_training;
        end = start + interval
        g.train(self.X[start:end], max_em_iters=self.em_iters)

    def train_using_mapreduce(self, init_training, em_iters):
        mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']    
        input = []
        count = 0
        for pair in init_training:
            input.append((count, pair, em_iters))
            count = count+1
        task_args = [protocol.write(pair, None)+"\n" for pair in input]
        job = TrainMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()        
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        #keys = map(lambda(k, v): k, kv_pairs)
        #print "Returned keys:", keys
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
    
    def segment_using_mapreduce(self, gmm_list, map_input, em_iter):
        pickle.dump(gmm_list, open('self_gmmlist', 'w'))
        os.chmod("self_gmmlist", S_IRUSR | S_IWUSR | S_IXUSR | \
                                 S_IRGRP | S_IXGRP |           \
                                 S_IROTH | S_IXOTH             )
        pickle.dump(em_iter, open('self_em_iter', 'w'))
        os.chmod("self_em_iter", S_IRUSR | S_IWUSR | S_IXUSR | \
                                 S_IRGRP | S_IXGRP |           \
                                 S_IROTH | S_IXOTH             )
        
        mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']        
        task_args = [protocol.write(i, None)+"\n" for i in map_input]
        job = SegmentMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        iter_bic_list = map(lambda(k, v): v, kv_pairs)
        iter_bic_dict = {}
        for pair in kv_pairs:
            p, (g, data) = pair
            p = int(p)
            iter_bic_dict[p] = data
            gmm_list[p] = g             #Update trained GMMs
        return iter_bic_dict, iter_bic_list

#        data_dict = {}
#        for k, v in kv_pairs:
#            data_dict[int(k)] = v
#        return data_dict
            
            
        
    def score_using_mapreduce(self, gmm_list):
        mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
        input = []
        count = 0
        for g in gmm_list:
            input.append((count, g)) #, self.X))
            count = count + 1
        task_args = [protocol.write(g, None)+"\n" for g in input]
        job = ScoreMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        #keys = map(lambda(k, v): k, kv_pairs)
        #print "Returned keys:", keys
        return map(lambda(k, v): v, kv_pairs)  
        
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
 