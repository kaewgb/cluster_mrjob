from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol
import cPickle as pickle

class TrainMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(TrainMRJob, self).job_runner_kwargs()
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
        
        #setup_cmds doens't work
        #config['setup_cmds'] += ['export PATH=/n/shokuji/da/penpornk/env/gmm/bin:/n/shokuji/da/penpornk/local/bin:/usr/local/bin:/usr/bin:/bin:/usr/X11/bin:/usr/local64/lang/cuda-3.2/bin/:/n/shokuji/da/penpornk/local/hadoop/bin:$PATH']
        #config['setup_cmds'] += ['export LD_LIBRARY_PATH=/usr/local64/lang/cuda-3.2/lib64:/usr/local64/lang/cuda-3.2/lib:/n/shokuji/da/penpornk/local/lib:$LD_LIBRARY_PATH']
        config['bootstrap_mrjob'] = False
        return config
        
    def hadoop_job_runner_kwargs(self):
        config = super(TrainMRJob, self).hadoop_job_runner_kwargs()
        config['hadoop_extra_args'] += [
            "-verbose",
        #    "-mapdebug", "/n/shokuji/da/penpornk/diarizer/debug.sh"
        ]
        return config
    
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