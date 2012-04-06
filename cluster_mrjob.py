import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy
import time
import struct
import scipy.stats.mstats as stats
import ConfigParser
import os.path
import getopt
import h5py

import cluster_tools as tools
from gmm_specializer.gmm import *
from cluster_mrjob_helper import *
from all_pairs_BIC_score import AllPairsBicScore
#from mrjob.protocol import PickleProtocol as protocol

MINVALUEFORMINUSLOG = -1000.0

class Diarizer(object):

    def __init__(self, f_file_name, sp_file_name):
        #self.variant_param_spaces = variant_param_spaces
        #self.device_id = device_id
        #self.names_of_backends = names_of_backends
        self.sgmnt_count = 0
        f = open(f_file_name, "rb")
        self.ftime = open('HadoopTime'+str(time.time()), 'w')

        print "...Reading in HTK feature file..."
        
        #=== Read Feature File ==
        try:
            nSamples = struct.unpack('>i', f.read(4))[0]
            sampPeriod = struct.unpack('>i', f.read(4))[0]
            sampSize = struct.unpack('>h', f.read(2))[0]
            sampKind = struct.unpack('>h', f.read(2))[0]

            print "INFO: total number of frames read: ", nSamples
            self.total_num_frames = nSamples
                
            D = sampSize/4 #dimension of feature vector
            l = []
            count = 0
            while count < (nSamples * D):
                bFloat = f.read(4)
                fl = struct.unpack('>f', bFloat)[0]
                l.append(fl)
                count = count + 1
        finally:
            f.close()

        #=== Prune to Speech Only ==
        print "...Reading in speech/nonspeech file..."
        pruned_list = []
        num_speech_frames = nSamples            

        if sp_file_name:
            sp = open(sp_file_name, "r")
                        
            l_start = []
            l_end = []
            num_speech_frames = 0
            for line in sp:
                s = line.split(' ')
                st = math.floor(100 * float(s[2]) + 0.5)
                en = math.floor(100 * float(s[3].replace('\n','')) + 0.5)
                st1 = int(st)
                en1 = int(en)
                l_start.append(st1*19)
                l_end.append(en1*19)
                num_speech_frames = num_speech_frames + (en1 - st1 + 1)

            print "INFO: total number of speech frames: ", num_speech_frames

            total = 0
            for start in l_start:
                end = l_end[l_start.index(start)]
                total += (end/19 - start/19 + 1)
                x = 0
                index = start
                while x < (end-start+19):
                    pruned_list.append(l[index])
                    index += 1
                    x += 1
        else: #no speech file, take in all features
            pruned_list = l

        floatArray = np.array(pruned_list, dtype = np.float32)
        print 'floatArray.shape', floatArray.shape
        self.X = floatArray.reshape(num_speech_frames, D)
        print 'num_speech_frames, D', num_speech_frames, ',', D
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]


    def write_to_RTTM(self, rttm_file_name, sp_file_name, meeting_name, most_likely, num_gmms, seg_length):

        print "...Writing out RTTM file..."

        #do majority voting in chunks of 250
        duration = seg_length
        chunk = 0
        end_chunk = duration

        max_gmm_list = []

        smoothed_most_likely = np.array([], dtype=np.float32)

            
        while end_chunk < len(most_likely):
            chunk_arr = most_likely[range(chunk, end_chunk)]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(250))
            chunk += duration
            end_chunk += duration

        end_chunk -= duration
        if end_chunk < len(most_likely):
            chunk_arr = most_likely[range(end_chunk, len(most_likely))]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(len(most_likely)-end_chunk))

        most_likely = smoothed_most_likely
        
        out_file = open(rttm_file_name, 'w')

        with_non_speech = -1*np.ones(self.total_num_frames)

        if sp_file_name:
            speech_seg = np.loadtxt(sp_file_name, delimiter=' ',usecols=(2,3))
            speech_seg_i = np.round(speech_seg*100).astype('int32')
            sizes = np.diff(speech_seg_i)
        
            sizes = sizes.reshape(sizes.size)
            offsets = np.cumsum(sizes)
            offsets = np.hstack((0, offsets[0:-1]))

            offsets += np.array(range(len(offsets)))
        
        #populate the array with speech clusters
            speech_index = 0
            counter = 0
            for pair in speech_seg_i:
                st = pair[0]
                en = pair[1]
                speech_index = offsets[counter]
                
                counter+=1
                idx = 0
                for x in range(st+1, en+1):
                    with_non_speech[x] = most_likely[speech_index+idx]
                    idx += 1
        else:
            with_non_speech = most_likely
            
        cnum = with_non_speech[0]
        cst  = 0
        cen  = 0
        for i in range(1,self.total_num_frames): 
            if with_non_speech[i] != cnum: 
                if (cnum >= 0):
                    start_secs = ((cst)*0.01)
                    dur_secs = (cen - cst + 2)*0.01
                    out_file.write("SPEAKER " + meeting_name + " 1 " + str(start_secs) + " "+ str(dur_secs) + " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")

                    
                cst = i
                cen = i
                cnum = with_non_speech[i]
            else:
                cen+=1
                  
        if cst < cen:
            cnum = with_non_speech[self.total_num_frames-1]
            if(cnum >= 0):
                start_secs = ((cst+1)*0.01)
                dur_secs = (cen - cst + 1)*0.01
                out_file.write("SPEAKER " + meeting_name + " 1 " + str(start_secs) + " "+ str(dur_secs) + " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")


        print "DONE writing RTTM file"

    def write_to_GMM(self, gmmfile):

        gmm_f = open(gmmfile, 'w')

        gmm_f.write("Number of clusters: " + str(len(self.gmm_list)) + "\n")
             
        #print parameters
        cluster_count = 0
        for gmm in self.gmm_list:

            gmm_f.write("Cluster " + str(cluster_count) + "\n")
            means = gmm.components.means
            covars = gmm.components.covars
            weights = gmm.components.weights

            gmm_f.write("Number of Gaussians: "+ str(gmm.M) + "\n")

            gmm_count = 0
            for g in range(0, gmm.M):
                g_means = means[gmm_count]
                g_covar_full = covars[gmm_count]
                g_covar = np.diag(g_covar_full)
                g_weight = weights[gmm_count]

                gmm_f.write("Gaussian: " + str(gmm_count) + "\n")
                gmm_f.write("Weight: " + str(g_weight) + "\n")
                
                for f in range(0, gmm.D):
                    gmm_f.write("Feature " + str(f) + " Mean " + str(g_means[f]) + " Var " + str(g_covar[f]) + "\n")

                gmm_count+=1
                
            cluster_count+=1

        print "DONE writing GMM file"
        
    def new_gmm(self, M, cvtype):
        self.M = M
        self.gmm = GMM(self.M, self.D, cvtype=cvtype)

    def new_gmm_list(self, M, K, cvtype):
        self.M = M
        self.init_num_clusters = K
        self.gmm_list = [GMM(self.M, self.D, cvtype=cvtype) for i in range(K)]
    
    def compare_array(self, a, b):
        if len(a) != len(b):
            print 'len(a) = {0} != {1} = len(b)'.format(len(a), len(b))
            return False
        cnt_a = [0 for i in range(0,16)]
        cnt_b = [0 for i in range(0,16)]
        for i in a:
            if i >= 16:
                print 'i=', i 
                sys.exit()
            cnt_a[i] = cnt_a[i] + 1
            
        for i in b:
            cnt_b[i] = cnt_b[i] + 1
            
        for i in range(0, len(cnt_a)):
            if cnt_a[i] != cnt_b[i]:
                print 'cnt_a[{0}] = {1} != {2} = cnt_b[{0}]'.format(i, cnt_a[i], cnt_b[i])
                return False
        return True
        
    def compare_dict(self, a, b):
        if len(a.keys()) != len(b.keys()):
            print "len a:", len(a.keys()), "len b:", len(b.keys())
            return False
        for k in a.keys():
            if len(a[k]) != len(b[k]):
                print "len a[{0}]={1}, len b[{0}]={2}".format(k, len(a[k]), len(b[k]))
                return False
            print "len a[{0}]={1} = len b[{0}]".format(k, len(a[k]))
            for i in range(0, len(a[k])):
                if len(a[k][i]) != len(b[k][i]):
                    print "len a[{0}][{1}]= {2} != {3} = len b[{0}][{1}]".format(k, i, len(a[k][i]), len(b[k][i]))
                    return False
                print "len a[{0}][{1}] = {2} = len b[{0}][{1}]".format(k, i, len(a[k][i]))
                for j in range(0, len(a[k][i])):
                    if a[k][i][j] != b[k][i][j]:
                        print "a[{0}][{1}][{4}] = {2} != {3} = b[{0}][{1}][{4}]".format(k,i,a[k][i][j], b[k][i][j], j)
                        return False
        print "dict a and b are equal"
        return True
    
#    def compare_data_list(self, a, b):
#        k = 0
#        if len(a) != len(b):
#            print "len a={1}, len b={2}".format(k, len(a), len(b))
#            return False
#        print "len a = {1} = len b".format(k, len(a))
#        for i in range(0, len(a)):
#            if len(a[i]) != len(b[i]):
#                print "len a[{1}] = {2} != {3} = len b[{1}]".format(k, i, len(a[i]), len(b[i]))
#                return False
#            print "len a[{1}] = {2} = len b[{1}]".format(k, i, len(a[i]))
#            for j in range(0, len(a[i])):
#                if a[i][j] != b[i][j]:
#                    print "a[{1}][{4}] = {2} != {3} = b[{1}][{4}]".format(k,i,a[i][j], b[i][j], j)
#                    return False

    def compare_features(self, a, b, i, ii):
        k=0
        if len(a) != len(b):
            print "len a[{0}]={1}, len b[{3}]={2}".format(i, len(a), len(b),ii)
            return False
        for j in range(0, len(a)):
            if a[j] != b[j]:
                #print "a[{1}][{4}] = {2} != {3} = b[{5}][{4}]".format(k,i,a[j], b[j], j, ii)
                return False
        return True
                
    def compare_data_list(self, a, b):
        k = 0
        if len(a) != len(b):
            print "len a={1}, len b={2}".format(k, len(a), len(b))
            return False
        print "len a = {1} = len b".format(k, len(a))
        unmatched = range(0, len(a))
        #for i in range(0, len(a)):
        for i in range(0, 20):
            matched = False
            for ii in unmatched:
                if self.compare_features(a[i], b[ii], i, ii):
                    matched = True
                    unmatched.remove(ii)
            if matched == False:
                print "a[{0}] unmatched".format(i)
                return False
        print "a and b are equal"
        return True
    
    def segment_majority_vote(self, interval_size, em_iters):
        
        cloud_flag = False
        hadoop = True
        self.em_iters = em_iters
        #print "In segment majority vote"
        # Resegment data based on likelihood scoring
        score_time = time.time()
        num_clusters = len(self.gmm_list)
        if cloud_flag == False:
            likelihoods = self.gmm_list[0].score(self.X)
            for g in self.gmm_list[1:]:
                likelihoods = np.column_stack((likelihoods, g.score(self.X)))
    
            if num_clusters == 1:
                most_likely = np.zeros(len(self.X))
            else:
                most_likely = likelihoods.argmax(axis=1)            
        else:
            if num_clusters == 1:
                most_likely = np.zeros(len(self.X))
            elif hadoop == False:
                map_res = map(self.MRhelper.score_map, self.gmm_list)
                most_likely = reduce(self.MRhelper.score_reduce, map_res).argmax(axis=1) #likelihoods.argmax
            else:
                lst = self.MRhelper.score_using_mapreduce(self.gmm_list)
                likelihoods = lst[0]
                for l in lst[1:]:
                    likelihoods = np.column_stack((likelihoods, l))
                most_likely = likelihoods.argmax(axis=1)
                
        self.ftime.write("Score: {0}\n".format(time.time() - score_time))
        cloud_flag = True
        segment_time = time.time()
        if cloud_flag == False:
            # Across 2.5 secs of observations, vote on which cluster they should be associated with
            iter_training = {}
            iter_indices = {}
            for i in range(interval_size, self.N, interval_size):
    
                arr = np.array(most_likely[(range(i-interval_size, i))])
                max_gmm = int(stats.mode(arr)[0][0])
                iter_training.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append(self.X[i-interval_size:i,:])
                iter_indices.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append((i-interval_size, i))
    
            arr = np.array(most_likely[(range((self.N/interval_size)*interval_size, self.N))])
            max_gmm = int(stats.mode(arr)[0][0])
            iter_training.setdefault((self.gmm_list[max_gmm], max_gmm),[]).append(self.X[(self.N/interval_size)*interval_size:self.N,:])            
            iter_indices.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append((((self.N)/interval_size)*interval_size, self.N))
            

            # for each gmm, append all the segments and retrain
            iter_bic_dict = {}
            iter_bic_list = [] 
#            for gp, data_list in iter_training.iteritems():
#                g = gp[0]
#                p = gp[1]
#                cluster_data =  data_list[0]
#    
#                for d in data_list[1:]:
#                    cluster_data = np.concatenate((cluster_data, d))
##                
##                if self.compare_data_list(cluster_data, iter_bic_dict2[p]) == False:
##                    sys.exit()
##                cluster_data = iter_bic_dict2[p]
#                g.train(cluster_data, max_em_iters=em_iters)
#    
#                iter_bic_list.append((g,cluster_data))
#                iter_bic_dict[p] = cluster_data
                
            for gp, data_indices in iter_indices.iteritems():
                g, p = gp
                cluster_data = tools.get_data_from_indices(self.X, data_indices)
                    
                g.train(cluster_data, max_em_iters=em_iters)
                iter_bic_list.append((p, data_indices))
                iter_bic_dict[p] = data_indices
                
                
        elif hadoop == False:
            # Across 2.5 secs of observations, vote on which cluster they should be associated with
            iter_training = {}
            map_input = zip(np.hsplit(np.array(most_likely), range(interval_size, len(most_likely), interval_size)),
                            map(lambda(x):(x, x+interval_size), range(0, len(most_likely), interval_size)))
            map_res = map(self.MRhelper.vote_map, map_input)
            map_res.insert(0, iter_training)
            iter_training = reduce(self.MRhelper.vote_reduce, map_res)
            
            # for each gmm, append all the segments and retrain
            iter_bic_dict = {}
            iter_bic_list = []   
            map_res = map(self.MRhelper.segment_map, iter_training.iteritems())
            map_res.insert(0, (iter_bic_dict, iter_bic_list))
            iter_bic_dict, iter_bic_list = reduce(self.MRhelper.segment_reduce, map_res)
#            
        else:
            map_input = zip(np.hsplit(np.array(most_likely), range(interval_size, len(most_likely), interval_size)),
                            map(lambda(x): (x, x+interval_size), range(0, len(most_likely), interval_size)))
            iter_bic_dict, iter_bic_list = self.MRhelper.segment_using_mapreduce(self.gmm_list, map_input, em_iters)

        
        self.ftime.write("Segment: {0}\n".format(time.time() - segment_time))

        return iter_bic_dict, iter_bic_list, most_likely
    
    
    def compute_All_BICs(self, iteration_bic_list, cloud_flag, em_iters):
        """
        Finds the GMM pair with the best score by comparing ALL gmm pairs   
        """
        if cloud_flag:
            result = AllPairsBicScore().all_pairs_BIC_using_mapreduce(iteration_bic_list, em_iters, self.X, self.gmm_list)
            #result = self.MRhelper.bic_using_mapreduce(iteration_bic_list, em_iters)
        else:
            result = AllPairsBicScore().all_pairs_BIC_serial(iteration_bic_list, em_iters, self.X, self.gmm_list)
        return result
            
    def cluster(self, em_iters, KL_ntop, NUM_SEG_LOOPS_INIT, NUM_SEG_LOOPS, seg_length):
        cloud_flag = False
        self.MRhelper = MRhelper(em_iters, self.X, self.gmm_list)
        
        print " ====================== CLUSTERING ====================== "
        main_start = time.time()

        # ----------- Uniform Initialization -----------
        # Get the events, divide them into an initial k clusters and train each GMM on a cluster
        per_cluster = self.N/self.init_num_clusters
        init_training = zip(self.gmm_list,np.vsplit(self.X, range(per_cluster, self.N, per_cluster)))
        
        train_start = time.time()
        if cloud_flag == False:
            for g, x in init_training:
                g.train(x, max_em_iters=em_iters)
        else:
            init_training = zip(self.gmm_list, range(0, self.N, per_cluster), [per_cluster for i in range(1, self.N)])
            #map(self.MRhelper.train_map, init_training)
            self.gmm_list = self.MRhelper.train_using_mapreduce(init_training, em_iters)
        self.ftime.write('Train: {0}\n'.format(time.time() - train_start))
#        self.ftime.close()
#        sys.exit()
        
#        import cPickle as pickle
#        pickle.dump(self.gmm_list, open("list1", "w"))
#        pickle.dump(self.gmm_list, open("list2", "wb"))
#        self.ftime.close()
#        sys.exit()
        
        #self.write_to_GMM('init.gmm')

        # ----------- First majority vote segmentation loop ---------
        for segment_iter in range(0,NUM_SEG_LOOPS_INIT):
            iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote(seg_length, em_iters)
        #print "after segmenting"

        # ----------- Main Clustering Loop using BIC ------------

        # Perform hierarchical agglomeration based on BIC scores
        best_BIC_score = 1.0
        total_events = 0
        total_loops = 0
        
        while (best_BIC_score > 0 and len(self.gmm_list) > 1):
            total_loops+=1
            for segment_iter in range(0,NUM_SEG_LOOPS):
                iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote(seg_length, em_iters)
                            
            # Score all pairs of GMMs using BIC
            best_merged_gmm = None
            best_BIC_score = 0.0
            merged_tuple = None
            merged_tuple_indices = None

            # ------- KL distance to compute best pairs to merge -------
            if KL_ntop > 0:

                top_K_gmm_pairs = self.gmm_list[0].find_top_KL_pairs(KL_ntop, self.gmm_list)
                for pair in top_K_gmm_pairs:
                    score = 0.0
                    gmm1idx = pair[0]
                    gmm2idx = pair[1]
                    g1 = self.gmm_list[gmm1idx]
                    g2 = self.gmm_list[gmm2idx]

                    if gmm1idx in iter_bic_dict and gmm2idx in iter_bic_dict:
                        d1 = iter_bic_dict[gmm1idx]
                        d2 = iter_bic_dict[gmm2idx]
                        data = np.concatenate((d1,d2))
                    elif gmm1idx in iter_bic_dict:
                        data = iter_bic_dict[gmm1idx]
                    elif gmm2idx in iter_bic_dict:
                        data = iter_bic_dict[gmm2idx]
                    else:
                        continue

                    new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
                    
                    #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                    if score > best_BIC_score: 
                        best_merged_gmm = new_gmm
                        merged_tuple = (g1, g2)
                        merged_tuple_indices = (gmm1idx, gmm2idx)
                        best_BIC_score = score

            # ------- All-to-all comparison of gmms to merge -------
            else: 
                cloud_flag = 0
                if len(iter_bic_list) >= 2:
                    bic_start = time.time()
                    best_merged_gmm, merged_tuple, merged_tuple_indices, best_BIC_score = self.compute_All_BICs(iter_bic_list, cloud_flag, em_iters)
                    self.ftime.write("BIC: {0}\n".format(time.time() - bic_start))               

            # Merge the winning candidate pair if its deriable to do so
            if best_BIC_score > 0.0:
                gmms_with_events = []
                for gp in iter_bic_list:
                    gmms_with_events.append(self.gmm_list[gp[0]])

                #cleanup the gmm_list - remove empty gm  best_merged_gmm, merged_tuple, merged_tuple_indices, best_BIC_scorems
                for g in self.gmm_list:
                    if g not in gmms_with_events and g != merged_tuple[0] and g!= merged_tuple[1]:
                        #remove
                        self.gmm_list.remove(g)

                self.gmm_list.remove(merged_tuple[0])
                self.gmm_list.remove(merged_tuple[1])
                self.gmm_list.append(best_merged_gmm)
               

            
            print " size of each cluster:", [ g.M for g in self.gmm_list]
#            self.ftime.close()
#            sys.exit()
            
        print "=== Total clustering time: ", time.time()-main_start
        print "=== Final size of each cluster:", [ g.M for g in self.gmm_list]

        return most_likely

    
def print_usage():
        print """    ---------------------------------------------------------------------
    Speaker Diarization in Python with Asp and the GMM Specializer usage:
    ---------------------------------------------------------------------
    Arguments for the diarizer are parsed from a config file. 
    Default config file is diarizer.cfg, but you can pass your own file with the '-c' option. 
    Required is the config file header: [Diarizer] and the options are as follows:
    
    --- Required: ---
    basename: \t Basename of the file to process
    mfcc_feats: \t MFCC input feature file
    output_cluster: \t Output clustering file
    gmm_output: \t Output GMMs parameters file
    M_mfcc: \t Amount of gaussains per model for mfcc
    initial_clusters: Number of initial clusters

    --- Optional: ---
    spnsp_file: \t spnsp file (all features used by default)
    KL_ntop: \t Nuber of combinations to evaluate BIC on
            \t 0 to deactive KL-divergency (fastmatch-component)
    em_iterations: \t Number of iterations for the standard
                  \t segmentation loop training (3 by default)
    num_seg_iters_init: \t Number of majority vote iterations
                        \t in the initialization phase (2 by default)
    num_seg_iters: \t Number of majority vote iterations
                   \t in the main loop (3 by default)
    seg_length: \t Segment length for majority vote in frames
                \t (250 frames by default)

    For fastest performance, enable KL-divergency (KL_ntop = 3) and set
      \t num_seg_iters_init and num_seg_iters to 1
    """

    
def print_no_config():

    print "Please supply a config file with -c 'config_file_name.cfg' "
    return

def get_config_params(config):
        #read in filenames
    try:
        meeting_name = config.get('Diarizer', 'basename')
    except:
        print "basename not specified in config file! exiting..."
        sys.exit(2)
    try:
        f = config.get('Diarizer', 'mfcc_feats')
    except:
        print "Feature file mfcc_feats not specified in config file! exiting..."
        sys.exit(2)

    try:
        sp = config.get('Diarizer', 'spnsp_file')
    except:
        print "Speech file spnsp_file not specified, continuing without it..."
        sp = False

    try:
        outfile = config.get('Diarizer', 'output_cluster')
    except:
        print "output_cluster file not specified in config file! exiting..."
        sys.exit(2)

    try:
        gmmfile = config.get('Diarizer', 'gmm_output')
    except:
        print "gmm_output file not specified in config file! exiting..."
        sys.exit(2)
        
    #read GMM paramters
    try:
        num_gmms = int(config.get('Diarizer', 'initial_clusters'))
    except:
        print "initial_clusters not specified in config file! exiting..."
        sys.exit(2)

    try:
        num_comps = int(config.get('Diarizer', 'M_mfcc'))
    except:
        print "M_mfcc not specified in config file! exiting..."
        sys.exit(2)
        
    #read algorithm configuration
    try:
        kl_ntop = int(config.get('Diarizer', 'KL_ntop'))
    except:
        kl_ntop = 0
    try:
        num_seg_iters_init = int(config.get('Diarizer', 'num_seg_iters_init'))
    except:
        num_seg_iters_init = 2
        
    try:
        num_seg_iters = int(config.get('Diarizer', 'num_seg_iters'))
    except:
        num_seg_iters = 3

    try:
        num_em_iters = int(config.get('Diarizer', 'em_iterations'))
    except:
        num_em_iters = 3

    try:
        seg_length = int(config.get('Diarizer', 'seg_length'))
    except:
        seg_length = 250

        
    return meeting_name, f, sp, outfile, gmmfile, num_gmms, num_comps, num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length



if __name__ == '__main__':
    device_id = 0
    
    
    # Process commandline arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ["help"])
    except getopt.GetoptError, err:
        print_no_config()
        sys.exit(2)

    config_file = 'diarizer.cfg'
    config_specified = False
    for o, a in opts:
        if o == '-c':
            config_file = a
            config_specified = True
        if o == '--help':
            print_usage()
            sys.exit(2)
    

    if not config_specified:
        print "No config file specified, using defaul 'diarizer.cfg' file"
    else:
        print "Using the config file specified: '", config_file, "'"

    try:
        open(config_file)
    except IOError, err:
        print "Error! Config file: '", config_file, "' does not exist"
        sys.exit(2)
        
    # Parse diarizer config file
    config = ConfigParser.ConfigParser()

    config.read(config_file)

    meeting_name, f, sp, outfile, gmmfile, num_gmms, num_comps, num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length = get_config_params(config)

    # Create tester object
    diarizer = Diarizer(f, sp)

    # Create the GMM list
    diarizer.new_gmm_list(num_comps, num_gmms, 'diag')

    # Cluster
    most_likely = diarizer.cluster(num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length)

    # Write out RTTM and GMM parameter files
    diarizer.write_to_RTTM(outfile, sp, meeting_name, most_likely, num_gmms, seg_length)
    diarizer.write_to_GMM(gmmfile)


