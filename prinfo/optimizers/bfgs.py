import numpy as np

from scipy.optimize import line_search
from ..utils import get_multi_dot as get_md

class BFGSSolver:

    def __init__(self, 
        d,
        get_objective,
        get_gradient,
        get_projected,
        theta_init=None,
        max_rounds=100,
        epsilon=10**(-5)):

        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.d = d
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if theta_init is None:
            theta_init = np.random.randn(self.d, 1)

        self.theta_init = theta_init

        self.theta_hat = None
        self.objectives = []

    def get_parameters(self):

        if self.theta_hat is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.theta_hat)

    def run(self):

        estimatet = np.copy(self.theta_init)
        estimatet1 = None

        self.objectives.append(
            self.get_objective(estimatet))
        
        gradt = self.get_gradient(estimatet)
        gradt1 = None
        grad_norm = np.linalg.norm(gradt)
        H = np.eye(self.d)
        t = 0

        while grad_norm > self.epsilon and t < self.max_rounds:

            # Compute new estimate
            s = self._get_s(H, gradt, estimatet, t)
            estimatet1 = estimatet + s
            # TODO: is this okay?
            estimatet1 = self.get_projected(estimatet1)

            self.objectives.append(
                self.get_objective(estimatet1))

            # Compute new gradient and y
            gradt1 = self.get_gradient(estimatet1)
            grad_norm = np.linalg.norm(gradt1)
            y = gradt1 - gradt

            # Update gradient and esimtate state
            estimatet = np.copy(estimatet1)
            gradt = np.copy(gradt1)

            # Update H (B's inverse)
            H = self._get_H(H, s, y, t)

            t += 1

        self.theta_hat = estimatet1

    def _get_s(self, H, grad, estimate, t):

        p = np.dot(-H, grad)
        oofv = None if t == 0 else self.objectives[-2]
        results = line_search(
            self.get_objective,
            lambda x: self.get_gradient(x)[:,0],
            estimate,
            p,
            gfk=grad[:,0],
            old_fval=self.objectives[-1],
            old_old_fval=oofv)
        eta = results[0]

        return eta * p

    def _get_H(self, H, s, y, t):

        if t == 0:
            H *= np.dot(s.T, y) / np.dot(y.T, y)

        rho = 1 / np.dot(s.T, y)
        quad_terms = rho * np.dot(s, y.T)
        left = np.eye(self.d) - quad_terms
        right = np.eye(self.d) - quad_terms.T
        H_quad = get_md([left, H, right])
        s_cov = rho * np.dot(s, s.T)
        H = H_quad + s_cov

        return H
