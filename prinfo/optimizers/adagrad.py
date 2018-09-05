import numpy as np

from .utils import get_mirror_update as get_mu
from .utils import get_shrunk_and_thresholded as get_st
from ..utils import get_svd_power

class FullAdaGradOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=10000,
        delta=10**(-5),
        epsilon=10**(-5),
        eta0=0.1,
        lower=None,
        verbose=False):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.d = d
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.eta0 = eta0
        self.verbose = verbose

        if theta_init is None:
            theta_init = np.random.randn(self.d, 1) / self.d

        self.theta_init = self.get_projected(theta_init)
        self.adagrad = FullAdaGradServer(
            delta=delta,
            lower=lower)
        self.theta_hat = None
        self.objectives = []

    def get_parameters(self):

        if self.theta_hat is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.theta_hat)

    def run(self):

        estimate = np.copy(self.theta_init)
        updatet1 = estimate
        updatet = None

        self.objectives.append(
            self.get_objective(estimate))
        
        search_dir_norm = float('inf')
        t = 0

        while search_dir_norm > self.epsilon and t < self.max_rounds:

            # Compute unprojected update
            grad = self.get_gradient(estimate)
            updatet = np.copy(updatet1)
            updatet1 = self.adagrad.get_update(
                estimate,
                grad,
                self.eta0)

            # Compute convergence criterion
            search_dir = - (updatet1 - updatet) / self.eta0
            search_dir_norm = np.linalg.norm(search_dir)

            # Project onto feasible region for new estimate
            estimate = self.get_projected(updatet1)

            if t % 1000 == 0:

                self.objectives.append(
                    self.get_objective(estimate))

                if self.verbose:
                    print('Round:', t)
                    print('Objective:', self.objectives[-1])
                    print('Gradient norm:', np.linalg.norm(grad))
                    print('Search direction norm:', search_dir_norm)

            t += 1

        self.theta_hat = estimate

class FullAdaGradServer:

    def __init__(self,
        delta=10**(-5),
        lower=None):

        self.delta = delta
        self.lower= lower

        self.num_rounds = 0
        self.G = None

    def get_update(self, parameters, gradient, eta):
        
        self.num_rounds += 1

        if self.G is None:
            self.G = np.dot(gradient, gradient.T)
            self.d = gradient.shape[0]
        else:
            self.G += np.dot(gradient, gradient.T)

        self.S = get_svd_power(self.G, 0.5)
        self.H = self.S + np.eye(self.d) * self.delta

        return get_mu(
            parameters,
            eta,
            gradient, 
            self._get_dual,
            self._get_primal)

    def _get_dual(self, parameters):

        return np.dot(self.H, parameters)

    def _get_primal(self, dual_update):

        if self.lower is not None:
            dus = dual_update.shape

            if len(dus) == 2 and not 1 in set(dus):
                (U, s, V) = np.linalg.svd(dual_update)
                sparse_s = get_st(s, lower=self.lower)
                dual_update = get_multiplied_svd(U, s, V)
            else:
                dual_update = get_st(
                    dual_update, lower=self.lower) 

        H_inv = get_svd_power(self.H, -1)

        return np.dot(H_inv, dual_update)
