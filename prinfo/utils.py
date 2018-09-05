import numpy as np

def get_multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def get_quadratic(X, A):

    return get_multi_dot([X.T, A, X])

def get_thresholded(x, upper=None, lower=None):

    new_x = np.copy(x)

    if upper is not None:
        upper = np.ones_like(x) * upper
        upper_idx = new_x > upper
        new_x[upper_idx] = upper[upper_idx]

    if lower is not None:
        lower = np.ones_like(x) * lower
        lower_idx = new_x < lower
        new_x[lower_idx] = lower[lower_idx]

    return new_x

def get_kernel_matrix(kernel, X):

    N = X.shape[0] 
    K = np.zeros((N, N))

    for n in range(N):

        X_n = X[n,:]

        for m in range(n, N):

            K_nm = kernel(X_n, X[m,:])
            K[n,m] = K_nm
            K[m,n] = K_nm

    return K
