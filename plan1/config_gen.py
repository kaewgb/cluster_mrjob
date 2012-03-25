import sys

meeting_names = [
    'IS1000a', 'IS1000b', 'IS1000c', 'IS1000d',
    'IS1001a', 'IS1001b', 'IS1001c',
    'IS1003b', 'IS1003d', 
    'IS1006b', 'IS1006d',
    'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d'
]

tmp = sys.stdout
for name in meeting_names:
    f = open(name+'.cfg', 'w')
    sys.stdout = f
    print "[Diarizer]"
    print "basename =", name
    print "mfcc_feats = /n/shokuji/da/penpornk/full_experiment_sets/AMI/features_ff/{0}_seg.feat.gauss.htk".format(name)
    print "output_cluster = output/{0}.rttm".format(name)
    print "gmm_output = output/{0}.gmm".format(name)
    print
    print "em_iterations = 3"
    print "initial_clusters = 16"
    print "M_mfcc = 5"
    print
    print "KL_ntop = 0"
    print "num_seg_iters_init = 2"
    print "num_seg_iters = 3"
    print "seg_length = 250"
    f.close()
sys.stdout = tmp