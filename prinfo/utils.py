import numpy as np

def unzip(l):

    ls = [[item[i] for item in l]
          for i in get_range_len(l[0])]

    return tuple(ls)

def get_moving_avg(old, new, beta):

    weighted_old = beta * old
    weighted_new = (1 - beta) * new

    return weighted_old + weighted_new

def get_checklist(keys):

    return {k : False for k in keys}

def get_multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def get_quadratic(X, A):

    return get_multi_dot([X.T, A, X])

def get_largest_entries(s, energy=None, k=None):

    n = s.shape[0]

    if k is not None and energy is not None:
        raise ValueError(
            'At most one of the k and energy parameters should be set.')

    if k is not None:
        if k > n:
            raise ValueError(
                'The value of k must not exceed length of the input vector.')

    if energy is not None and (energy <= 0 or energy >= 1):
        raise ValueError(
            'The value of energy must be in the open interval (0,1).')

    s = np.copy(s)

    if not (s == np.zeros_like(s)).all():
        if k is not None:
            s[k+1:] = 0
        elif energy is not None:
            total = sum(s)
            current = 0
            count = 0
            
            for i in range(n):
                if current / total < energy:
                    current = current + s[i]
                    count = count + 1

            s[count+1:] = 0

    return s

def get_rank_k(m, n, k):

    if k > min([m, n]):
        raise ValueError(
            'The value of k must not exceed the minimum matrix dimension.')

    A = np.zeros((m,n))

    U = np.random.randn(m, k)
    V = np.random.randn(k, n)

    return np.dot(U, V)

def get_thresholded(x, upper=None, lower=None):

    new_x = np.copy(x)

    if upper is not None:
        new_x = np.minimum(upper, new_x)

    if lower is not None:
        new_x = np.maximum(lower, new_x)

    return new_x

def get_safe_power(s, power):

    new = np.zeros_like(s)
    masked_s = np.ma.masked_invalid(s).filled(0)

    if power < 0:
        non_zero = masked_s != 0
        new[non_zero] = np.power(
            masked_s[non_zero], power)
    else:
        new = np.power(masked_s, power)

    return new

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

def get_multiplied_svd(U, s, Vh):

    (n, p) = (U.shape[1], Vh.shape[0])
    sigma = np.zeros((n,p))

    for i in range(s.shape[0]):
        sigma[i,i] = s[i]

    return get_multi_dot([U, sigma, Vh])

def get_transformed_svd(A, get_trans, energy=0.95, k=None):

    (U, s, Vh) = np.linalg.svd(A)
    s = get_largest_entries(s, energy=energy, k=k)
    transformed_s = get_trans(s)

    return get_multiplied_svd(U, transformed_s, Vh)

def get_svd_power(A, power, energy=0.95, k=None):

    get_trans = lambda s: get_safe_power(s, power)

    return get_transformed_svd(A, get_trans, energy=energy, k=k)

def get_rotation(angle, P, P_inv=None):

    if P_inv is None:
        P_inv = get_svd_power(P, -1)

    A = np.eye(P.shape[0])

    A[0,0] = np.cos(angle)
    A[1,1] = np.cos(angle)
    A[0,1] = -np.sin(angle)
    A[1,0] = np.sin(angle)

    return get_multi_dot(
        [P_inv, A, P])
