from cluster_mrtemplate import ClusterMRJob
import cluster_tools as tools

import numpy as np
import os
import cPickle as pickle
from stat import *
from gmm_specializer.gmm import compute_distance_BIC
from mrjob.protocol import PickleProtocol as protocol

import time
import sys
import mrjob.util as util

class AllPairsBicScoreMRJob(ClusterMRJob):
    
    def job_runner_kwargs(self):
        config = super(AllPairsBicScoreMRJob, self).job_runner_kwargs()
        config['jobconf']['mapred.line.input.format.linespermap'] = 16
        #config['upload_files'] += ["iter_gmm_list"]
        config['upload_files'] += ['gmm.tgz']
        return config

    def mapper(self, key, value):
        """
        Each mapper computes the BIC score for a GMM pair
        """
        
        overall = t = time.time()
        
        index1, index2 = key        
        didx1, didx2, em_iters = value

        t = time.time()
#        X = tools.binary_read('self_X')
#        d1 = tools.get_data_from_indices(X, didx1)
#        d2 = tools.get_data_from_indices(X, didx2)
#        sys.stderr.write("get_data_from_indices: {0}\n".format(time.time()-t))
        d1 = tools.get_data_from_file_from_indices('self_X', didx1)
        d2 = tools.get_data_from_file_from_indices('self_X', didx2)
        sys.stderr.write("get_data_from_file_from_indices: {0}\n".format(time.time()-t))
        data = np.concatenate((d1, d2))
        
        t = time.time()
        util.unarchive('gmm.tgz', 'gmm')
        g1 = pickle.load(open('gmm/'+str(index1), 'r'))
        g2 = pickle.load(open('gmm/'+str(index2), 'r'))
        sys.stderr.write("read iter_gmm_list: {0}\n".format(time.time()-t))
        new_gmm = g1
        score = 0
        t = time.time()
        try:
            new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
        except:
            raise
        #data_to_yield = (score, new_gmm, g1, g2, index1, index2)
        data_to_yield = (score, index1, index2)
        sys.stderr.write("compute_distance_BIC: {0}\n".format(time.time()-t))
        sys.stderr.write("total BIC time: {0}\n".format(time.time()-overall))
        yield 1, data_to_yield
    
    
    def reducer(self, key, values):
        """
        Finds the GMM pair with the highest BIC score
        """
        best_score = 0.0
        merged_tuple_indices = (0, 0)  
        
        for score, index1, index2 in values:
            if score > best_score:
                best_score = score
                merged_tuple_indices = (index1, index2)
        result = (merged_tuple_indices, best_score)
        yield 1, result
        
    def steps(self):
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]

class AllPairsBicScore(object):
    
    def __init__(self):
        self.pure_python = True
    

    def all_pairs_BIC_using_mapreduce(self, iteration_bic_list, em_iters, X, gmm_list):
        """
        Computes the BIC score for all pairs by using MapReduce and returns
        the pair with the best score
        """
        
        print "Map-Reduce execution"
#        iter_gmm_list = map(lambda(gidx, didx): gmm_list[gidx], iteration_bic_list)
#        pickle.dump(iter_gmm_list, open('iter_gmm_list', 'w'))
#        os.chmod("iter_gmm_list", S_IRUSR | S_IWUSR | S_IXUSR | \
#                                 S_IRGRP | S_IXGRP |           \
#                                 S_IROTH | S_IXOTH             )
        
        from subprocess import call
        call(["mkdir", "-p", "gmm"])
        for i in range (0, len(iteration_bic_list)):
            gidx, didx = iteration_bic_list[i]
            pickle.dump(gmm_list[gidx], open('gmm/'+str(i), 'w'))
            os.chmod("iter_gmm_list", S_IRUSR | S_IWUSR | S_IXUSR | \
                                      S_IRGRP | S_IXGRP |           \
                                      S_IROTH | S_IXOTH             )
        import mrjob.util as util
        util.tar_and_gzip('gmm', 'gmm.tgz') 
        
        input = []
        l = len(iteration_bic_list)
        for gmm1idx in range(l):
            for gmm2idx in range(gmm1idx+1, l):
                gidx1, didx1 = iteration_bic_list[gmm1idx]
                gidx2, didx2 = iteration_bic_list[gmm2idx] 
                an_item = protocol().write((gmm1idx,gmm2idx),(didx1, didx2, em_iters))
                input.append(an_item+"\n")     
        
        mr_args = ['-v', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
        job = AllPairsBicScoreMRJob(args=mr_args).sandbox(stdin=input)
        runner = job.make_runner()
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        assert len(kv_pairs) == 1
        merged_tuple_indices, best_score = kv_pairs[0][1]
    
        # Re-merge the GMM pair with the highest score *here*, otherwise the next
        # segment_majority_vote will crash (issue with data ownership). If we don't
        # find a different workaround, we can simplify more the mapper and the reducer.
        # Essentially, we can avoid moving from mappers to the reducer the GMM pairs and
        # merged GMMs. Instead, we can move just indices and scores.
        # However, this re-merging is serialized...
        ind1, ind2 = merged_tuple_indices
        gidx1, idx1 = iteration_bic_list[ind1]
        gidx2, idx2 = iteration_bic_list[ind2]
        d1 = tools.get_data_from_indices(X, idx1)
        d2 = tools.get_data_from_indices(X, idx2)
        data = np.concatenate((d1,d2))
        g1 = gmm_list[gidx1]
        g2 = gmm_list[gidx2]
        new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
            
        return new_gmm, (g1, g2), merged_tuple_indices, best_score
    
    
    def all_pairs_BIC_serial(self, iter_bic_list, em_iters, X, gmm_list):
        """
        Computes the BIC score for all pairs in a "serial" way and returns
        the pair with the best score
        """
        #print "Serial execution"
            
        l = len(iter_bic_list)
        best_merged_gmm = None
        best_BIC_score = 0.0
        merged_tuple = None
        merged_tuple_indices = None
        
        for gmm1idx in range(l):
            for gmm2idx in range(gmm1idx+1, l):
                score = 0.0
                gidx1, idx1 = iter_bic_list[gmm1idx]
                gidx2, idx2 = iter_bic_list[gmm2idx] 
                d1 = tools.get_data_from_indices(X, idx1)
                d2 = tools.get_data_from_indices(X, idx2)
                data = np.concatenate((d1, d2))
                g1 = gmm_list[gidx1]
                g2 = gmm_list[gidx2]
                new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
                
                if score > best_BIC_score: 
                    best_merged_gmm = new_gmm
                    merged_tuple = (g1, g2)
                    merged_tuple_indices = (gmm1idx, gmm2idx)
                    best_BIC_score = score
        
        return best_merged_gmm, merged_tuple, merged_tuple_indices, best_BIC_score

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    AllPairsBicScoreMRJob().run()
