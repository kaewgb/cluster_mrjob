import numpy as np
import scipy.stats.mstats as stats

from cluster_mrtemplate import *
import cluster_tools as tools


class SegmentMRJob(ClusterMRJob):
    
    def job_runner_kwargs(self):
        config = super(SegmentMRJob, self).job_runner_kwargs()
        config['jobconf']['mapred.line.input.format.linespermap'] = 16
        config['upload_files'] += ["self_gmmlist"]
        config['upload_files'] += ["self_em_iter"]
        return config
    
    def mapper(self, pair, _):
        arr, indices = pair
        max_gmm = int(stats.mode(arr)[0][0])
        yield '{0:05d}'.format(max_gmm), indices
    
       
    def reducer(self, gmm_id, indices):
        gmm_id = int(gmm_id)
        gmm_list = pickle.load(open('self_gmmlist', 'r'))
        em_iter = pickle.load(open('self_em_iter', 'r'))
        X = tools.binary_read('self_X')
        
        data_indices = []
        for i in indices:
            data_indices.append(i)
        cluster_data = tools.get_data_from_indices(X, data_indices)
        gmm_list[gmm_id].train(cluster_data, max_em_iters=em_iter)
        yield (gmm_id, data_indices), gmm_list[gmm_id]
        
    
if __name__ == '__main__':
    SegmentMRJob.run()  