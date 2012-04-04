import struct
import numpy as np

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
        raise ValueError('len(a) = {0} != {1} = len(b)'.format(len(a), len(b)))
    for i in range(0, len(a)):
        if len(a[i]) != len(b[i]):
            raise ValueError('len(a[{0}] = {1} != {2} = len(b[{0}])'.format(i, len(a[i]), len(b[i])))
        for j in range(0, len(a[i])):
            if a[i][j] != b[i][j]:
                raise ValueError('a[{0}][{1}] = {2} != {3} = b[{0}][{1}]'.format(i, j, a[i][j], b[i][j])) 
                return False
    #print "Arrays are equal"    #<- Must comment if used in mrjob's mapper/reducer, or you'll get cPickle.UnpicklingError: invalid load key, 'A'. from protocol.py:110
    return True

def get_data_from_indices(X, indices):
    start, end = indices[0]
    cluster_data = X[start:end]
    for idx in indices[1:]:
        start, end = idx
        cluster_data = np.concatenate((cluster_data, X[start:end]))
    return cluster_data

def binary_read_from_indices(f, start, end, D):
    N = struct.unpack('>i', f.read(4))[0]
    end = min(end, N)
    N = end - start
    f.seek(8 + start*D*4)
    data = []
    for i in range(0, N):
        features = struct.unpack('>'+str(D)+'f', f.read(D*4))
#        print features
#        sys.exit()
        data.extend(features)
    floatArray = np.array(data, dtype = np.float32)
    return floatArray.reshape(N, D)