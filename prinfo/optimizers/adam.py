import numpy as np

from .utils import get_mirror_update as get_mu
from .utils import get_shrunk_and_thresholded as get_st
from ..utils import get_multiplied_svd, get_svd_power
from ..utils import get_moving_avg as get_ma

class DiagonalAdamOptimizer:

    def __init__(self,
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=100,
        epsilon=10**(-5),
        eta0=0.1,
        delta=10**(-5),
        beta1=0.9,
        beta2=0.9,
        lower=None, 
        verbose=False):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.d = d
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.eta0 = eta0

        if theta_init is None:
            theta_init = self.get_projected(
                np.random.randn(self.d, 1) / self.d)

        self.theta_init = theta_init
        self.adam = DiagonalAdamServer(
            delta=delta,
            beta1=beta1,
            beta2=beta2,
            lower=lower,
            verbose=verbose)
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

        self.objectives.append(
            self.get_objective(estimate))
        
        search_dir_size = float('inf')
        t = 0

        while search_dir_size > self.epsilon and t < self.max_rounds:

            # Compute unprojected update
            grad = self.get_gradient(estimate)
            update = self.adam.get_update(
                estimate,
                grad,
                self.eta0)

            # Compute convergence criterion
            search_dir = - (update - estimate) / self.eta0
            search_dir_size = np.linalg.norm(search_dir)**2

            # Project onto feasible region for new estimate
            estimate = self.get_projected(update)

            self.objectives.append(
                self.get_objective(estimate))

            if t % 100 == 0:
                print('Round:', t)
                print('Objective:', self.objectives[-1])
                print('Search direction size:', search_dir_size)

            t += 1

        self.theta_hat = estimate

class StochasticCoordinateDiagonalAdamServer:

    def __init__(self,
        p,
        delta=10**(-5),
        beta1=0.9,
        beta2=0.9,
        lower=None, 
        verbose=False):

        self.p = p
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.lower = lower
        self.verbose = verbose

        self.first_moment = np.zeros((self.p, 1))
        self.second_moment = np.zeros((self.p, 1))
        self.num_rounds = np.zeros((self.p, 1))
        self.denom1_subtractor = np.ones((self.p, 1))
        self.denom2_subtractor = np.ones((self.p, 1))

    def get_update(self, parameters, gradient, eta, batch):

        self.num_rounds[batch,:] += 1
        self.second_moment[batch,:] = get_ma(
            self.second_moment[batch,:],
            np.power(gradient, 2), 
            self.beta2)
        self.first_moment[batch,:] = get_ma(
            self.first_moment[batch,:],
            gradient, 
            self.beta1)
        self.denom1_subtractor[batch,:] *= self.beta1
        self.denom2_subtractor[batch,:] *= self.beta2

        # Update the link function
        denom = 1 - self.denom2_subtractor[batch,:]
        sm_hat = self.second_moment[batch,:] / denom
        
        self.H = np.power(sm_hat, 0.5) + self.delta

        denom = 1 - self.denom1_subtractor[batch,:]
        fm_hat = self.first_moment[batch,:] / denom
        mirror_update = get_mu(
            parameters,
            eta,
            fm_hat,
            self._get_dual, 
            self._get_primal)

        return mirror_update

    def _get_dual(self, parameters):

        return self.H * parameters

    def _get_primal(self, dual_update):

        if self.lower is not None:
            dual_update = get_st(
                dual_update, lower=self.lower) 

        return dual_update / self.H

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'second_moment': self.second_moment,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'grad': self.first_moment,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}

class DiagonalAdamServer:

    def __init__(self, 
        delta=10**(-5),
        beta1=0.9,
        beta2=0.9,
        lower=None, 
        verbose=False):

        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.lower = lower
        self.verbose = verbose

        self.first_moment = None
        self.second_moment = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        if self.first_moment is None:
            self.first_moment = np.copy(gradient)
            self.second_moment = np.power(gradient, 2)

        self.second_moment = get_ma(
            self.second_moment,
            np.power(gradient, 2), 
            self.beta2)
        self.first_moment = get_ma(
            self.first_moment, 
            gradient, 
            self.beta1)

        denom = 1 - self.beta2**(self.num_rounds)
        sm_hat = self.second_moment / denom

        self.H = np.power(sm_hat, 0.5) + self.delta

        denom = 1 - self.beta1**(self.num_rounds)
        fm_hat = self.first_moment / denom
        mirror_update = get_mu(
            parameters, 
            eta, 
            fm_hat,
            self._get_dual, 
            self._get_primal)

        return mirror_update

    def _get_dual(self, parameters):

        return self.H * parameters

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

        return dual_update / self.H

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'second_moment': self.second_moment,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'grad': self.first_moment,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
