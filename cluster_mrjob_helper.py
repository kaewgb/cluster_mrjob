import numpy as np
import scipy.stats.mstats as stats

import os
import sys
import time
import cPickle as pickle
from stat import *

from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol

from cluster_trainmrjob import TrainMRJob
from cluster_scoremrjob import ScoreMRJob
from cluster_sgmtmrjob import SegmentMRJob
from cluster_bicmrjob import BICMRJob

import cluster_tools as tools

class MRhelper:
    def __init__(self, em_iters, X, gmm_list):
        #For Naive Python
        self.em_iters = em_iters
        self.X = X
        self.gmm_list = gmm_list
        
        #For mrjob
        tools.binary_dump(self.X, 'self_X')
        os.chmod("self_X", S_IRUSR | S_IWUSR | S_IXUSR | \
                                 S_IRGRP | S_IXGRP |           \
                                 S_IROTH | S_IXOTH             )
        from subprocess import call
        call(["/n/shokuji/da/penpornk/local/hadoop/bin/hadoop", "dfs", "-rm", "/user/penpornk/tmp/self_X"])
        call(["/n/shokuji/da/penpornk/local/hadoop/bin/hadoop", "dfs", "-put", "self_X", "/user/penpornk/tmp/self_X"])
        
    def train_map(self, init_training):
        g, start, interval = init_training;
        end = start + interval
        g.train(self.X[start:end], max_em_iters=self.em_iters)

    def train_using_mapreduce(self, init_training, em_iters):
        mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']    
        input = []
        count = 0
        t = time.time()
        for pair in init_training:
            g, start, interval = pair
            #g.initialize_asp_mod()
            input.append((count, pair, em_iters))
            count = count+1
        task_args = [protocol.write(pair, None)+"\n" for pair in input]
        print "[train] preparation time:", time.time()-t
        t = time.time()
        job = TrainMRJob(args=mr_args).sandbox(stdin=task_args)
        runner = job.make_runner()
        print "[train] init mrjob:", time.time()-t        
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        #keys = map(lambda(k, v): k, kv_pairs)
        #print "Returned keys:", keys
        return map(lambda(k, v): v, kv_pairs)  
        
        
    def segment_map(self, iter_item):
        gp, data_indices = iter_item
        g = gp[0]
        p = gp[1]
        cluster_data =  tools.get_data_from_indices(self.X, data_indices)

        g.train(cluster_data, max_em_iters=self.em_iters)
        return (g, p, data_indices)

    def segment_reduce(self, x, y):
        iter_bic_dict, iter_bic_list = x
        g, p, data_indices = y
        iter_bic_list.append((p, data_indices))
        iter_bic_dict[p] = data_indices
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
        iter_bic_list = map(lambda(k, v): k, kv_pairs)
        iter_bic_dict = {}
        for pair in kv_pairs:
            (p, data_indices), g = pair
            iter_bic_dict[p] = data_indices
            gmm_list[p] = g             #Update trained GMMs
        return iter_bic_dict, iter_bic_list
            
        
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
        arr, idX = map_input
        return (int(stats.mode(arr)[0][0]), idX)
    
    def vote_reduce(self, iter_training, item):
        max_gmm, idX = item
        iter_training.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append(idX)
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
 