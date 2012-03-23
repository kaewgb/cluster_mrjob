#from asp.config import MapReduceDetector
from scala_module import PseudoModule # steal this for now


class MapReduceModule:
    """
    A module to support specialization via MapReduce. Does nothing for now,
    but exists to be consistent with other backends.
    """
    pass


class MapReduceToolchain:
    """
    Tools to execute mapreduce jobs.
    """
    def __init__(self, cluster='emr'):
        MapReduceDetector.detect_or_exit(cluster)
        self.cluster = cluster # (local|hadoop|emr)


# TODO this should probably subclass ASPBackend but I can't the imports right.
# For now just override the same methods.
class MapReduceBackend(object):
    """
    Class to encapsulate a mapreduce backend for Asp.
    """
    def __init__(self, module=None, toolchain=None):
        self.module = module or MapReduceModule()
        self.toolchain = toolchain or MapReduceToolchain()

    def compile(self):
        """
        Trigger a compile of this backend.
        """
        raise NotImplementedError

    def get_compiled_function(self, name):
        """
        Return a callable for a raw compiled function (that is, this must be a
        variant name rather than a function name). Note that for
        MapReduceBackends functions are not compiled, just stored.
        """
        try:
            func = getattr(self.compiled_module, name)
        except:
            raise AttributeError("Function %s not found in compiled module." %
                                 (name,))
        return func

    def specialize(self, AspMRJobCls):
        """
        Return a callable that runs the given map and reduce functions.
        """
        from asp.jit.mapreduce_support import AspMRJob
        from sys import stderr

        def mr_callable(args):
            mr_args = ['--strict-protocols', '-r', self.toolchain.cluster]
            job = AspMRJobCls(args=mr_args).sandbox(stdin=args)
            runner = job.make_runner()
            runner.run()
            kv_pairs = map(job.parse_output_line, runner.stream_output())
            return kv_pairs

        return mr_callable

class MapReduceDetector(object):
    """
    Detect if a MapReduce platform is available.
    """
    def __init__(self):
        """ Fail on instantiation. """
        raise RuntimeError("MapReduceDetector should not be instantiated.")

    @classmethod
    def detect(cls, platform):
        """ Detect if a particular platform is available. """
        import os
        try:
            import mrjob
            if (platform == 'emr'):
                aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
                aws_secret_acces_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            elif (platform == 'hadoop'):
                hadoop_home = os.environ["HADOOP_HOME"]
            elif (platform == 'local'):
                pass
            else:
                return False
        except ImportError:
            return False
        except KeyError:
            return False
        return platform

    @classmethod
    def get_platforms(cls):
        """ Returns a list of available MapReduce Platforms. """
        return filter(cls.detect, ['local', 'hadoop', 'emr'])

    @classmethod
    def detect_or_exit(cls, platform):
        if not cls.detect(platform):
            raise EnvironmentError("Unable to detect '%s' MapReduce platform. See configuration instructions for all platforms at http://packages.python.org/mrjob/writing-and-running.html#running-on-emr" % platform)

