from cluster_mrtemplate import *
import cluster_tools as tools
import sys
import time
import os
import tempfile

class TrainMRJob(ClusterMRJob):
    
    def mapper(self, pair, _):
        #sys.stderr.write("$TMPDIR:" + os.environ['TMPDIR']+"\n")
        #sys.stderr.write("tempfile.gettempdir():" + tempfile.gettempdir()+"\n")
        overall = t = time.time()
        X = tools.binary_read('self_X')
        sys.stderr.write("read self_X: {0}\n".format(time.time()-t))
        
        id, (g, start, interval), em_iters = pair
        end = start+interval
        
        t = time.time()
        g.train(X[start:end], max_em_iters=em_iters)
        sys.stderr.write("train: {0}\n".format(time.time()-t))
        sys.stderr.write("overall train time: {0}\n".format(time.time()-overall))
        yield "{0:05d}".format(id), g
    
    def reducer(self, key, value):
        for v in value:
            gmm = v
        yield key, gmm
    
if __name__ == '__main__':
    TrainMRJob.run()    