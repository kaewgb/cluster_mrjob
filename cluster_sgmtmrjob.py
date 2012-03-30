from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol
import cPickle as pickle

class SegmentMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(SegmentMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        config['upload_files'] += ["self_X"]
        config['upload_files'] += ["self_gmmlist"]
        
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
        config = super(SegmentMRJob, self).hadoop_job_runner_kwargs()
        config['hadoop_extra_args'] += [
            "-verbose",
        #    "-mapdebug", "/n/shokuji/da/penpornk/diarizer/debug.sh"
        ]
        return config
    
    def mapper(self, pair, _):
        most_likely, start, interval = pair
        max_gmm = int(stats.mode(most_likely)[0][0])
        yield max_gmm, (start, interval)
    
    def reducer(self, gmm_id, indices):
        X = pickle.load(open('self_X', 'r'))
        gmm_list = pickle.load(open('self_gmmlist', 'r'))
        data = []
        for index in indices:
            start, interval = index
            end = start + interval
            data.append(X[start:end])
            
        gmm_list[gmm_id].train(data)
        
        yield gmm_id, data
    
if __name__ == '__main__':
    SegmentMRJob.run()  