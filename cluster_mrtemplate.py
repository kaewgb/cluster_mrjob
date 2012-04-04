from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol
import cPickle as pickle

import logging
logging.basicConfig()

class ClusterMRJob(MRJob):
    
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(ClusterMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        
        squid = True
        if squid:
            config['cmdenv']['PYTHONPATH'] = ":".join([
                "/n/shokuji/da/penpornk/diarizer"
            ])
            config['cmdenv']['PATH'] = ":".join([
                "/n/shokuji/da/penpornk/env/gmm/bin",
                "/n/shokuji/da/penpornk/local/bin",
                "/usr/local/bin", "/usr/bin", "/bin",
                "/usr/X11/bin",
                "/usr/local64/lang/cuda-3.2/bin/",
                "/n/shokuji/da/penpornk/local/hadoop/bin"
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
                "/usr/lib64/atlas",
                "/usr/local64/lang/cuda-3.2/lib64",
                "/usr/local64/lang/cuda-3.2/lib",
                "/n/shokuji/da/penpornk/local/lib"                                            
            ])
            config['cmdenv']['TMPDIR'] = "/scratch/tmp/penpornk" #for ASP compilation
            config['cmdenv']['C_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
            config['cmdenv']['CPLUS_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
            config['cmdenv']['BLAS'] = "/usr/lib64/atlas/libptcblas.so"
            config['cmdenv']['LAPACK'] = "/usr/lib64/atlas/liblapack.so"
            config['cmdenv']['ATLAS'] = "/usr/lib64/atlas/libatlas.so"
            config['python_bin'] = "/n/shokuji/da/penpornk/env/gmm/bin/python"    
        else:
            # Trying to config $HOME for the expanduser("~") bug. But both didn't work.
            #config['cmdenv']['HOME'] = "/home/kaewgb"
            #config['setup_cmds'] += "export HOME=/home/kaewgb"
            
            config['cmdenv']['PYTHONPATH'] = ":".join([
                "/home/kaewgb/gmm",
                "/home/kaewgb/gmm/examples",
                "/home/kaewgb/cluster_mrjob"
            ])
            config['cmdenv']['PATH'] = ":".join([
                "/home/kaewgb/local/bin",
                "/usr/local/cuda/bin",
                "/usr/local/cuda/cudaprof/bin",
                "/usr/bin",
                "/usr/local/bin"
            ])
            config['cmdenv']['LD_LIBRARY_PATH'] = ":".join([
                "/usr/local/lib",
                "/usr/local/cuda/lib64",
                "/home/kaewgb/local/lib"                                  
            ])
            config['cmdenv']['C_INCLUDE_PATH'] = "/home/kaewgb/local/include"
            config['cmdenv']['CPLUS_INCLUDE_PATH'] = "/home/kaewgb/local/include"
            config['python_bin'] = "/home/kaewgb/local/bin/python" 
        
        config['bootstrap_mrjob'] = False
        return config
    
    def hadoop_job_runner_kwargs(self):
        config = super(ClusterMRJob, self).hadoop_job_runner_kwargs()
        config['hadoop_extra_args'] += [
            "-files", "hdfs:///user/penpornk/tmp/self_X#self_X", #Generic option, has to come before verbose (used to be -cacheFile in Hadoop 0.15)
            "-verbose"
        #    "-mapdebug", "/n/shokuji/da/penpornk/diarizer/debug.sh"
        ]
        return config
    
if __name__ == '__main__':
    ClusterMRJob.run() 