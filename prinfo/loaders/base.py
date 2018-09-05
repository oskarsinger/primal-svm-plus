import numpy as np

from ..utils import get_svd_power
from ..utils import get_rank_k
from ..utils import get_checklist

class BernoulliLoader:

    def __init__(self, n, m, p=0.5, lazy=True):

        self.n = n
        self.m = m
        self.p = p
        self.lazy = lazy

        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        init = np.random.binomial(1, self.p, size=self.n*self.m)

        self.data = init.reshape((self.n, self.m))

    def rows(self):

        return self.n

    def cols(self):

        return self.m

class GaussianLoader:

    def __init__(self, 
        n, p, 
        lazy=True,
        batch_size=None, 
        k=None, 
        mean=None,
        unit_norm=False):

        # Check validity of k parameter
        if k is None:
            self.low_rank = False
        else:
            if k > min([n, p]):
                raise ValueError(
                    'Parameter k must not exceed the minimum matrix dimension.')
            else:
                self.low_rank = True

        # Check validity of batch_size parameter
        if batch_size is not None:
            if batch_size > n:
                raise ValueError(
                    'Parameter batch_size must not exceed parameter n.')

        self.batch_size = batch_size

        self.n = n
        self.p = p
        self.k = k
        self.lazy = lazy
        self.unit_norm = unit_norm

        # Set mean of each column by input constants
        if mean is not None:
            if not mean.shape[1] == self.p:
                raise ValueError(
                    'Length of means parameter must be equal to p.')

        self.mean = mean

        # Generate data
        self.X = None
        
        if not self.lazy:
            self.X = _get_batch(
                self.n, 
                self.p, 
                self.k, 
                self.mean, 
                unit_norm=self.unit_norm)
            
        # Checklist for which rows sampled in current epoch
        self.sampled = get_checklist(range(self.n))

        # Number of requests made for data
        self.num_rounds = 0

        # Number of times through the full data set
        self.num_epochs = 0

    def get_data(self):

        if self.X is None:
            self.X = _get_batch(
                self.n, 
                self.p, 
                self.k, 
                self.mean,
                unit_norm=self.unit_norm)

        self.num_rounds += 1

        data = None

        if self.batch_size is None:
            data = np.copy(self.X)
        else:
            # Check for the rows that have not been sampled this epoch
            unsampled = [i 
                         for (i, s) in self.sampled.items() 
                         if not s]

            # Refresh if unsampled will not fill a batch
            if len(unsampled) < self.batch_size:
                self.sampled = get_checklist(range(self.n))
                self.num_epochs += 1

                unsampled = len(self.sampled.keys())

            # Sample indexes corresponding to rows in data matrix
            sample_indexes = np.random.choice(
                np.array(unsampled), 
                self.batch_size, 
                replace=False)
            
            # Update checklist with sampled rows
            for i in sample_indexes.tolist():
                self.sampled[i] = True

            data = np.copy(self.X[sample_indexes,:])

        return data

    def get_status(self):

        return {
            'n': self.n,
            'p': self.p,
            'num_rounds': self.num_rounds,
            'k': self.k,
            'batch_size': self.batch_size,
            'sampled': self.sampled,
            'lazy': self.lazy,
            'low_rank': self.low_rank,
            'mean': self.mean}

    def name(self):

        return 'GaussianData'

    def finished(self):

        finished = False

        if self.batch_size is None:
            finished = self.num_rounds > 1
        else:
            num_unsampled = sum(
                [0 if s else 1 for s in self.sampled.values()])
            finished = num_unsampled < self.batch_size
            
        return finished

    def refresh(self):

        self.X = None
        self.sampled = get_checklist(range(self.n))
        self.num_rounds = 0

    def cols(self):
        
        return self.p

    def rows(self):
        
        return self.n

def _get_batch(bs, p, k=None, mean=None, unit_norm=True):

    batch = None

    if k is not None:
        batch = get_rank_k(bs, p, k)
    else:
        batch = np.random.randn(bs, p)

    if mean is not None:
        batch += mean

    return batch

class LinearDynamicsSequenceLoader:

    def __init__(self,
        A,
        num_data,
        seed=None):

        self.A = A
        self.num_data = num_data
        
        if seed is None:
            seed = np.ones((self.A.shape[1], 1))

        self.seed = seed

        self._set_noiseless_Y()
        self._set_Y()

    def _set_noiseless_Y(self):

        y_list = [self.seed]

        for t in range(self.num_data):
            old_y = y_list[-1]
            new_y = np.dot(self.A, old_y)

            y_list.append(new_y)

        self.noiseless_Y = np.array(y_list[1:])[:,:,0]

    def _set_Y(self):

        noise = np.random.randn(
            self.num_data, 
            self.seed.shape[0])

        self.Y = self.noiseless_Y + noise

    def get_data(self):

        return np.copy(self.Y)

    def rows(self):

        return self.Y.shape[0]

    def cols(self):

        return self.Y.shape[1]

    def refresh(self):

        self._set_Y()

