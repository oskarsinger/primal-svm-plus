import numpy as np

def get_one_hots(A):

    N = A.shape[0]
    unique = np.unique(A)
    num_unique = unique.shape[0]
    one_hots = np.zeros((N, num_unique))
    
    for (i, u) in enumerate(unique):
        one_hots[A == u, i] = 1

    return one_hots

