from cluster_mrtemplate import *
import cluster_tools as tools

class ScoreMRJob(ClusterMRJob):
    
    def job_runner_kwargs(self):
        config = super(ScoreMRJob, self).job_runner_kwargs()
        config['jobconf']['mapred.line.input.format.linespermap'] = 4
        return config
            
    def mapper(self, pair, _):
        X = tools.binary_read('self_X')
        key, g = pair
        likelihood = g.score(X)
        yield '{0:05d}'.format(key), likelihood
    
    def reducer(self, dummy, cols):
        for col in cols:
            l = col 
        yield dummy, l
        
#        likelihoods = cols.next()
#        for col in cols:
#            likelihoods = np.column_stack((likelihoods, col))            
#        yield dummy, likelihoods.argmax(axis=1)
    
if __name__ == '__main__':
    ScoreMRJob.run()  