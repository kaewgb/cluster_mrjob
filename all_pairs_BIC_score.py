from asp.jit import mapreduce_support as mr
from mrjob import protocol as pr
import numpy as np
#from em import *
from gmm_specializer.gmm import compute_distance_BIC
from mrjob.protocol import PickleProtocol as protocol

class AllPairsBicScoreMRJob(mr.AspMRJob):

    #INPUT_PROTOCOL = protocol
    #OUTPUT_PROTOCOL = protocol
    
    DEFAULT_INPUT_PROTOCOL = 'pickle'
    DEFAULT_PROTOCOL = 'pickle'

    def job_runner_kwargs(self):
        config = super(AllPairsBicScoreMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        #config['cmdenv']['PYTHONPATH'] = ":".join([
        #    "/home/kaewgb/gmm/examples"
        #])
        config['cmdenv']['PATH'] = ":".join([
            "/n/shokuji/da/penpornk/env/gmm/bin",
            "/n/shokuji/da/penpornk/local/bin",
            "/usr/local/bin", "/usr/bin", "/bin",
            "/usr/X11/bin",
            "/usr/local64/lang/cuda-3.2/bin/",
            "/n/shokuji/da/penpornk/local/hadoop/bin"
        ])
        config['cmdenv']['LD_LIBRARY_PATH'] = ":".join([
            "/usr/local64/lang/cuda-3.2/lib64",
            "/usr/local64/lang/cuda-3.2/lib",
            "/n/shokuji/da/penpornk/local/lib"                                            
        ])
        config['cmdenv']['C_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
        config['cmdenv']['CPLUS_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
        config['python_bin'] = "/n/shokuji/da/penpornk/env/gmm/bin/python"
        config['bootstrap_mrjob'] = False
        return config

    def mapper(self, key, value):
        """
        Each mapper computes the BIC score for a GMM pair
        """
        #import sys
        index1, index2 = key        
        g1, g2, data, em_iters = value
        new_gmm = g1
        score = 0
#        f = open('inmapper', 'w')
#        f.write('inmapper')
#        f.close()
        #print >>sys.stderr, "K,V", key, value
        try:
            new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
        except:
            #print sys.stderr, "SKIPPING", g1, g2
            raise
        data_to_yield = (score, new_gmm, g1, g2, index1, index2)
        #print "MAP YIELDS", data_to_yield
        yield 1, data_to_yield
    
    
    def reducer(self, key, values):
        """
        Finds the GMM pair with the highest BIC score
        """
        best_score = 0.0
        best_merged_gmm = None
        merged_tuple = None
        merged_tuple_indices = (0, 0)
        for score, merged_gmm, g1, g2, index1, index2 in values:
            if score > best_score:
                best_score = score
                merged_tuple = (g1, g2)
                merged_tuple_indices = (index1, index2)
                best_merged_gmm = merged_gmm
                
        result = (best_merged_gmm, merged_tuple, merged_tuple_indices, best_score)
#        f = open('/home/kaewgb/gmm/examples/results', 'w')
#        f.write(result)
#        f.close()
        yield 1, result
        
    def steps(self):
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]

class AllPairsBicScore(object):
    
    def __init__(self):
        self.pure_python = True
    

    def all_pairs_BIC_using_mapreduce(self, iteration_bic_list, em_iters):
        """
        Computes the BIC score for all pairs by using MapReduce and returns
        the pair with the best score
        """
        
        print "Map-Reduce execution"
        
        input = []
        l = len(iteration_bic_list)
        for gmm1idx in range(l):
            for gmm2idx in range(gmm1idx+1, l):
                g1, d1 = iteration_bic_list[gmm1idx]
                g2, d2 = iteration_bic_list[gmm2idx]
                data = np.concatenate((d1,d2))
                an_item = pr.PickleProtocol().write((gmm1idx,gmm2idx),(g1, g2, data, em_iters))
                input.append(an_item+"\n")     
        
        mr_args = ['-v', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
        job = AllPairsBicScoreMRJob(args=mr_args).sandbox(stdin=input)
        runner = job.make_runner()
        runner.run()
        kv_pairs = map(job.parse_output_line, runner.stream_output())
        #print 'kv_pairs: ', kv_pairs
        assert len(kv_pairs) == 1
        #best_score, merged_tuple, best_merged_gmm, ind1, ind2 = kv_pairs[0][1]
        best_merged_gmm, merged_tuple, merged_tuple_indices, best_score = kv_pairs[0][1]
    
        # Re-merge the GMM pair with the highest score *here*, otherwise the next
        # segment_majority_vote will crash (issue with data ownership). If we don't
        # find a different workaround, we can simplify more the mapper and the reducer.
        # Essentially, we can avoid moving from mappers to the reducer the GMM pairs and
        # merged GMMs. Instead, we can move just indices and scores.
        # However, this re-merging is serialized...
        ind1, ind2 = merged_tuple_indices
        g1, d1 = iteration_bic_list[ind1]
        g2, d2 = iteration_bic_list[ind2]
        data = np.concatenate((d1,d2))
        new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
            
        return new_gmm, (g1, g2), merged_tuple_indices, best_score
        #return best_score, (g1,g2), new_gmm, ind1, ind2
    
    
    def all_pairs_BIC_serial(self, iter_bic_list, em_iters):
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
                g1, d1 = iter_bic_list[gmm1idx]
                g2, d2 = iter_bic_list[gmm2idx] 

                data = np.concatenate((d1,d2))
                new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)

                #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                if score > best_BIC_score: 
                    best_merged_gmm = new_gmm
                    merged_tuple = (g1, g2)
                    merged_tuple_indices = (gmm1idx, gmm2idx)
                    best_BIC_score = score
        
        return best_merged_gmm, merged_tuple, merged_tuple_indices, best_BIC_score
                    
#        l = len(iteration_bic_list)
#        best_merged_gmm = None
#        best_score = 0.0
#        ind1 = 0
#        ind2 = 0
#        merged_tuple = None
#        for gmm1idx in range(l):
#            for gmm2idx in range(gmm1idx+1, l):
#                score = 0.0
#                g1, d1 = iteration_bic_list[gmm1idx]
#                g2, d2 = iteration_bic_list[gmm2idx]
#                data = np.concatenate((d1,d2))
#                new_gmm, score = compute_distance_BIC(g1, g2, data)
#                if score > best_score: 
#                    best_merged_gmm = new_gmm
#                    merged_tuple = (g1, g2)
#                    best_score = score
#                    ind1 = gmm1idx
#                    ind2 = gmm2idx
#                        
#        return best_score, merged_tuple, best_merged_gmm, ind1, ind2

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    AllPairsBicScoreMRJob().run()
