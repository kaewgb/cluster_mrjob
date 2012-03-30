from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol
import cPickle as pickle
import numpy as np

class ScoreMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(ScoreMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        config['upload_files'] += ["self_X"]
        
        config['cmdenv']['PYTHONPATH'] = ":".join([
            "/home/kaewgb/gmm/examples"
        ])
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
        
    def hadoop_job_runner_kwargs(self):
        config = super(ScoreMRJob, self).hadoop_job_runner_kwargs()
        config['hadoop_extra_args'] += [
            "-verbose",
        #    "-mapdebug", "/n/shokuji/da/penpornk/diarizer/debug.sh"
        ]
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