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