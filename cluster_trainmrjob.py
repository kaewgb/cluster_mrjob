from cluster_mrtemplate import *

class TrainMRJob(ClusterMRJob):
    
    def mapper(self, pair, _):
        X = pickle.load(open('self_X', 'r'))
        id, (g, start, interval), em_iters = pair
        end = start+interval
        g.train(X[start:end], max_em_iters=em_iters)
        yield "{0:05d}".format(id), g
    
    def reducer(self, key, value):
        for v in value:
            gmm = v
        yield key, gmm
    
if __name__ == '__main__':
    TrainMRJob.run()    