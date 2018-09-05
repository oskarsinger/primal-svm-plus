import numpy as np

from whitehorses.loaders.simple import BernoulliLoader as BL
from whitehorses.loaders.simple import GaussianLoader as GL
from whitehorses.loaderts.multiview import StaticCCAProbabilisticModelLoader as SCCAPML

def get_easy_BGCCALUPIL(
    d_o, 
    d_p, 
    k=5, 
    n=1000, 
    p=0.5, 
    fraction_flipped=0.2,
    lazy=True):

    # Generate observation covariance matrices
    Psi_init_o = np.random.randn(d_o * 2, d_o)
    Psi_init_p = np.random.randn(d_p * 2, d_p)
    Psi_o = np.dot(Psi_init_o.T, Psi_init_o)
    Psi_p = np.dot(Psi_init_p.T, Psi_init_p)

    # Generate lifts from z to observations
    W_o = np.random.randn(d_o, k)
    W_p = np.random.randn(d_p, k)

    # Generate means for observations
    mu_o = np.random.randn(d_o, 1)
    mu_p = np.random.randn(d_p, 1)

    # Generate means for Z
    mu_zpos = np.random.randn(k, 1)
    mu_zneg = np.random.randn(k, 1)
    
    return BernoulliGaussianCCALUPILoader(
        n=n,
        p=p,
        Psi_o=Psi_o, Psi_p=Psi_p,
        W_o=W_o, W_p=W_p,
        mu_o=mu_o, mu_p=mu_p,
        mu_zpos=mu_zpos, mu_zneg=mu_zneg,
        fraction_flipped=fraction_flipped)

class BernoulliGaussianCCALUPILoader:

    def __init__(self,
        n=1000,
        p=0.5,
        Psi_o=None, Psi_p=None,
        W_o=None, W_p=None,
        mu_o=None, mu_p=None, 
        mu_zpos=None, mu_zneg=None,
        fraction_flipped=0.2):

        self.n = n
        self.p = p
        (self.Psi_o, self.Psi_p) = (Psi_o, Psi_p)
        (self.W_o, self.W_p) = (W_o, W_p)
        (self.mu_o, self.mu_p) = (mu_o, mu_p)
        (self.mu_zpos, self.mu_zneg) = (mu_zpos, mu_zneg)
        self.ff = fraction_flipped
        self.bias = bias

        self.data = None
        self.k = self.mu_zpos.shape[0]
        self.d_o = self.W_o.shape[0]
        self.d_p = self.W_p.shape[0]

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        # Generate labels y
        y = BL(self.n, 1, p=self.p).get_data()

        # Generate noise for hidden variables
        Z = np.random.randn(self.k, self.n)

        # Add mean to hidden variables
        Z[:,(y == 1)[:,0]] += self.mu_zpos
        Z[:,(y == 0)[:,0]] += self.mu_zneg

        # Generate Xo and Xp
        X_o = SCCAPML(
            self.W_o,
            self.Psi_o,
            self.mu_o,
            Z).get_data()
        X_p = SCCAPML(
            self.W_p,
            self.Psi_p,
            self.mu_p,
            Z).get_data()

        if self.ff is not None:
            size = int(self.ff * y.shape[0])
            flipped = np.random.choice(
                y.shape[0],
                size,
                replace=False)
            y[flipped,:] = 1 - y[flipped,:]

        self.data = (X_o, X_p, y)
