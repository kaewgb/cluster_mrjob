import math
import sys
import struct
import numpy as np
import cPickle as pickle


def binary_dump(array, filename):
    N, D = array.shape
    f = open(filename, "wb")
    f.write(struct.pack('>i', N))
    f.write(struct.pack('>i', D))
    for i in range(0, N):
        for j in range(0, D):
            f.write(struct.pack('>f', array[i][j]))
    f.close()
    
def binary_read(filename):
    f = open(filename, "rb")
    N = struct.unpack('>i', f.read(4))[0]
    D = struct.unpack('>i', f.read(4))[0]
    array = struct.unpack('>'+str(N*D)+'f', f.read(N*D*4))
    floatArray = np.array(array, dtype = np.float32)
    return floatArray.reshape(N, D)

def compare_array(a, b):
    if len(a) != len(b):
        print 'len(a) = {0} != {1} = len(b)'.format(len(a), len(b))
        return False
    for i in range(0, len(a)):
        if len(a[i]) != len(b[i]):
            print 'len(a[{0}] = {1} != {2} = len(b[{0}])'.format(i, len(a[i]), len(b[i]))
            return False
        for j in range(0, len(a[i])):
            if a[i][j] != b[i][j]:
                print 'a[{0}][{1}] = {2} != {3} = b[{0}][{1}]'.format(i, j, a[i][j], b[i][j]) 
                return False
    print "Arrays are equal"
    return True

if __name__ == '__main__':
    f = open(sys.argv[1], "rb")
    
    print "...Reading in HTK feature file..."
    
    #=== Read Feature File ==
    try:
        nSamples = struct.unpack('>i', f.read(4))[0]
        sampPeriod = struct.unpack('>i', f.read(4))[0]
        sampSize = struct.unpack('>h', f.read(2))[0]
        sampKind = struct.unpack('>h', f.read(2))[0]
    
        print "INFO: total number of frames read: ", nSamples
            
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
    
    if sys.argv[2]:
        sp = open(sys.argv[2], "r")
                    
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
    X = floatArray.reshape(num_speech_frames, D)
    print 'num_speech_frames, D', num_speech_frames, ',', D
    
    binary_dump(X, "dumpX")
    Y = binary_read("dumpX")
    print X.shape
    print Y.shape
    compare_array(X, Y)

