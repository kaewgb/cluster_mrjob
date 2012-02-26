import numpy as np
import scipy.stats.mstats as stats

#from mrjob.job import  MRJob
#from mrjob.protocol import PickleProtocol as protocol

class NaivePythonMR:
    def __init__(self, em_iters, X, gmm_list):
        self.em_iters = em_iters
        self.X = X
        self.gmm_list = gmm_list
    
    def train_map(self, init_training):
        g, x = init_training;
        g.train(x, max_em_iters=self.em_iters)
                
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