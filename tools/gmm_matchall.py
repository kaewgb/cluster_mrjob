import sys
import math

def equal(x, y):
    if abs(x-y) < 0.001:
        return True
    if math.isnan(x) and math.isnan(y):
        return True 
    return False

def tequal(x, y):
    if equal(x[0], y[0]) and equal(x[1], y[1]):
        return True
    return False

def fequal(x, y):
    no_features = len(x)
    for k in range(0, no_features):
        if not tequal(x[k], y[k]):
            msg = 'gaussian1[features]['+str(k)+']='+str(x[k])+'!='+ \
                str(y[k])+'= gaussian2[features]['+str(k)+']'
            return (False, msg)
    return (True, "")
        
def gequal(x, y, c):
    no_gaussians = len(x)
    gaussian_list = range(0, no_gaussians)
    for j in range(0, no_gaussians):
        gaussian1 = x[j]
        found = False
        for k in gaussian_list:
            if equal(y[k]['weight'], gaussian1['weight']):
                found, msg = fequal(y[k]['features'], gaussian1['features'])
                if found:
                    gaussian2 = y[k]
                    gaussian_list.remove(k)
                    break
        if not found:
            print msg
            print 'Cannot find any weights in cluster#', c ,'that is matched with gaussian#', j
            return False
        else:
            print 'Matched gaussian#', j, 'with gaussian#', k, 'of cluster#', c
    return True

def check():
    if len(sys.argv) > 3:
        no_features=int(sys.argv[3])
        gmm1 = parse(sys.argv[1], no_features=no_features)
        gmm2 = parse(sys.argv[2], no_features=no_features)
    else:
        gmm1 = parse(sys.argv[1])
        gmm2 = parse(sys.argv[2])

    
    if len(gmm1) != len(gmm2) :
        print 'len(gmm1) = ', len(gmm1), '!= ', len(gmm2), ' = len(gmm2)'
        return False
    no_clusters = len(gmm1)
    cluster_list = range(0, no_clusters)
    for i in range(0, no_clusters):
        print 'Cluster#', i
        
        matched = False
        for j in cluster_list:
            if len(gmm1[i]) != len(gmm2[j]):
                continue
            if gequal(gmm1[i], gmm2[j], j):
                cluster_list.remove(j)
                matched = True
                break
        if not matched:
            return False
                
    print 'The outputs are equal'
    return True
        
def parse(input, no_features=60): #Change to 19 if use with AMI data
    f = open(input, 'r')
    lines = f.readlines()
    no_clusters = int(lines[0][20:])
    i = 1
    clusters = []
    while i < len(lines):
        if lines[i][0:8] != 'Cluster ':
            raise Exception('Wrong input format. No Cluster#.')
        current_cluster = int(lines[i][8:])
        if lines[i+1][0:21] != 'Number of Gaussians: ':
            raise Exception('Wrong input format. No number of Gaussians specified for Cluster #', current_cluster)
        no_gaussians = int(lines[i+1][21:])
        i = i+2
        gaussians = [] 
        for g in range(0, no_gaussians):
            gauss = {}
            if lines[i][:10] != 'Gaussian: ' or int(lines[i][10:]) != g:
                raise Exception('Wrong input format. No Gaussian number specified\nline=', lines[i])
            if lines[i+1][:8] != 'Weight: ':
                raise Exception('Wrong input format. No weights specified for gaussian #', g, 'of Cluster#', current_cluster)
            gauss['weight'] = float(lines[i+1][8:])
            i = i+2
            features = []
            for j in range(0, no_features):
                comps = lines[i].split(' ')
                features.append((float(comps[3]), float(comps[5])))
                i = i+1
            gauss['features'] = features
            gaussians.append(gauss)
        clusters.append(gaussians)
    return clusters
    
if __name__ == '__main__':
    check()  