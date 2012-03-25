import sys
import math

def fequal(x, y):
    if abs(x-y) < 0.001:
        return True
    if math.isnan(x) and math.isnan(y):
        return True 
    return False

def tequal(x, y):
    if fequal(x[0], y[0]) and fequal(x[1], y[1]):
        return True
    return False

def gequal(x, y, c):
    no_gaussians = len(x)
    for j in range(0, no_gaussians):
        gaussian1 = x[j]
        found = False
        gaussian_list = range(0, no_gaussians)
        for k in gaussian_list:
            if(fequal(y[k]['weight'], gaussian1['weight']) and
               tequal(y[k]['features'][0], gaussian1['features'][0])):
                gaussian2 = y[k]
                found = True
                gaussian_list.remove(k)
                break
        if not found:
            print 'Cannot find any weights in cluster#', c ,'that is matched with gaussian#', j
            return False
        else:
            print 'Matched gaussian#', j, 'with gaussian#', k, 'of cluster#', c
        no_features = len(gaussian1['features'])
        if len(gaussian2['features']) != no_features:
            print 'Number of features are not equal'
            return False
        for k in range(0, no_features):
            if not tequal(gaussian1['features'][k], gaussian2['features'][k]):
                print 'gaussian1[features][', k,']=', gaussian1['features'][k], '!=', \
                    gaussian2['features'][k], '= gaussian2[features][',k,']'
                return False
    return True

def check():
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
        
def parse(input):
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
                raise Exception('Wrong input format. No Gaussian number specified')
            if lines[i+1][:8] != 'Weight: ':
                raise Exception('Wrong input format. No weights specified for gaussian #', g, 'of Cluster#', current_cluster)
            gauss['weight'] = float(lines[i+1][8:])
            i = i+2
            features = []
            for j in range(0, 19):
                comps = lines[i].split(' ')
                features.append((float(comps[3]), float(comps[5])))
                i = i+1
            gauss['features'] = features
            gaussians.append(gauss)
        clusters.append(gaussians)
    return clusters
    
if __name__ == '__main__':
    check()  