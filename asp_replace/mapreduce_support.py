from sys import stderr
from mrjob.job import MRJob

"""
NOTE: this file cannot import anything that won't be in the remote env.
"""

# Logging is enabled by mrjob, and configuring it is required if we're printing
# debug info to stderr.
import logging
logging.basicConfig()

class AspMRJob(MRJob):
    """
    Encapsulates an Asp-specific MapReduce job.
    """
    pass


# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    AspMRJob().run()
