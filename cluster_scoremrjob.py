from cluster_mrtemplate import *

class ScoreMRJob(ClusterMRJob):
        
    def job_runner_kwargs(self):
        config = super(ScoreMRJob, self).job_runner_kwargs()
        config['upload_files'] += ["self_X"]
        return config
    
    def mapper(self, pair, _):
        X = pickle.load(open('self_X', 'r'))
        key, g = pair
        likelihood = g.score(X)
        yield '{0:05d}'.format(key), likelihood
    
    def reducer(self, dummy, cols):
        
#        likelihoods = cols.next()
#        for col in cols:
#            likelihoods = np.column_stack((likelihoods, col))            
        for col in cols:
            l = col 
        yield dummy, l
#        likelihoods = l[0]
#        for col in l[1:]:
#            likelihoods = np.column_stack((likelihoods, col))
#        yield dummy, likelihoods.argmax(axis=1)
    
if __name__ == '__main__':
    ScoreMRJob.run()  