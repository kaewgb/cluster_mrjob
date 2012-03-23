from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol
from gmm_specializer.gmm import compute_distance_BIC

class BICMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(BICMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        config['cmdenv']['PYTHONPATH'] = ":".join([
            "/home/kaewgb/gmm/examples"
        ])
        config['bootstrap_mrjob'] = False
        return config
    
    def mapper(self, key, value):
        index1, index2 = key        
        g1, g2, data, em_iters = value
        new_gmm = g1
        score = 0
        try:
            new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
        except:
            #print sys.stderr, "SKIPPING", g1, g2
            raise
        data_to_yield = (score, new_gmm, g1, g2, index1, index2)
        #print "MAP YIELDS", data_to_yield
        yield 1, data_to_yield
    
    def reducer(self, key, values):
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
        yield 1, result
    
if __name__ == '__main__':
    BICMRJob.run()    