import numpy as np
import scipy.stats.mstats as stats

from cluster_mrtemplate import *


class SegmentMRJob(ClusterMRJob):
    
    def job_runner_kwargs(self):
        config = super(SegmentMRJob, self).job_runner_kwargs()
        config['upload_files'] += ["self_X"]
        config['upload_files'] += ["self_gmmlist"]
        config['upload_files'] += ["self_em_iter"]
        return config
    
    def mapper(self, pair, _):
        arr, X = pair
        max_gmm = int(stats.mode(arr)[0][0])
        yield '{0:05d}'.format(max_gmm), X
    
    def reducer(self, gmm_id, data_list):
        X = data_list.next()
        for d in data_list:
            X = np.concatenate((X, d))
        yield gmm_id, X
#        gmm_id = int(gmm_id)
#        gmm_list = pickle.load(open('self_gmmlist', 'r'))
#        em_iter = pickle.load(open('self_em_iter', 'r'))
#        x = data_list.next()
#        for d in data_list:
#            x = np.concatenate((x, d))
#        
#        gmm_list[gmm_id].train(x, em_iter)
#        yield gmm_id, (gmm_list[gmm_id], x)
    
if __name__ == '__main__':
    SegmentMRJob.run()  