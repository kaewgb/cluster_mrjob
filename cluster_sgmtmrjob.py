from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol

class SegmentMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(SegmentMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        config['cmdenv']['PYTHONPATH'] = ":".join([
            "/home/kaewgb/gmm/examples"
        ])
        config['bootstrap_mrjob'] = False
        return config
    
    def mapper(self, pair, _):
        key, (gp, data_list), em_iters = pair
        g = gp[0]
        p = gp[1]
        cluster_data =  data_list[0]

        for d in data_list[1:]:
            cluster_data = np.concatenate((cluster_data, d))

        g.train(cluster_data, max_em_iters=em_iters)
        yield key, (g, p, cluster_data)
    
    def reducer(self, key, value):
        g, p, cluster_data = value
        yield list((g, cluster_data))
    
if __name__ == '__main__':
    SegmentMRJob.run()  