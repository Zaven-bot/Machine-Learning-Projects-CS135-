'''
K Nearest Neighbors

>>> data_NF = np.asarray([
...     [1, 0],
...     [0, 1],
...     [-1, 0],
...     [0, -1]])
>>> query_QF = np.asarray([
...     [0.9, 0],
...     [0, -0.9]])

Example Test K=1
----------------
# Find the single nearest neighbor for each query vector
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
>>> neighb_QKF.shape
(2, 1, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[1., 0.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.]])

Example Test K=3
----------------
# Now find 3 nearest neighbors for the same queries
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
>>> neighb_QKF.shape
(2, 3, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.],
       [ 1.,  0.],
       [-1.,  0.]])
'''
import numpy as np

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    '''
        
    # create the array which will hold the KNN
    neighb_QKF = np.zeros(shape=(query_QF.shape[0], K, query_QF.shape[1]))
            
    q_ctr = 0
    # for every query point
    for query in query_QF:
        # create a N x 2 2D-array to hold the index in data_NF [0]
        # and the distance from the query pt. to data pt. [1]
        distances = np.zeros((data_NF.shape[0], 2))
        counter = 0
        data_features = data_NF.shape[1]
        for row in data_NF:
            distance = 0
            for col in range(0, data_features):
                # add dist. from a data pt. to query pt. to running total
                distance = distance + (query[col] - row[col]) ** 2
        
            # square root the running total
            distance = distance ** 0.5
            # save ordered pair [counter, abs. proximity measurement]
            distances[counter] = [counter, float(distance)]
            # compare data_NF[row + 1] vs. same query
            counter += 1
        
        # merge Sort to sort all the distances based off
        # of their distances (query pt. to data pt.)
        mergesort(distances, 0, distances.shape[0] - 1)

        # insert only K point coordinates into return array
        # in increasing order
        for i in range(0, K):
            data_key = int(distances[i, 0])
            neighb_QKF[q_ctr, i] = data_NF[data_key]
        
        # look at next query point
        q_ctr += 1
        
    return neighb_QKF

def mergesort(arr, l, r):
    if (l < r):
        # recursively call 
        m = (l + r) // 2
        mergesort(arr, l, m)
        mergesort(arr, m + 1, r)
        merge(arr, l, m, r)

# l = left index
# m = middle index
# r = right index
# arr = array of distances
def merge(arr, l, m, r):
    # create temp left / right arrays
    L = np.zeros((m - l + 1, 2))
    R = np.zeros((r - m, 2))
    
    # copy distances to L
    lidx = 0
    for i in range(l, m + 1):
        L[lidx] = arr[i]
        lidx += 1
        
    # copy distances to R
    ridx = 0
    for k in range(m + 1, r + 1):
        R[ridx] = arr[k]
        ridx += 1

    # merge L and R back into arr
    rsize = np.shape(R)[0]
    rctr = 0
    lsize = np.shape(L)[0]
    lctr = 0
    actr = l
    
    # expend the distances (in increasing order) until one array
    # is completely done
    while lctr < lsize and rctr < rsize:
        # left array has smaller value
        if L[lctr][1] <= R[rctr][1]:
            arr[actr] = L[lctr]
            lctr += 1
        # right array has smaller value
        else:
            arr[actr] = R[rctr]
            rctr += 1
        actr += 1 
        
    # left didn't expend all distances
    while lctr < lsize:
        arr[actr] = L[lctr]
        lctr += 1
        actr += 1
    
    # right didn't expend all distances
    while rctr < rsize:
        arr[actr] = R[rctr]
        rctr += 1
        actr += 1