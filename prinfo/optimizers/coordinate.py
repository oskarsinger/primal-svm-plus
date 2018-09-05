import numpy as np

from .adam import StochasticCoordinateDiagonalAdamServer as SCDAS
from .stepsize import InversePowerScheduler as IPS
from .stepsize import FixedScheduler as FS

# TODO: quasi-Newton stuff gets weird here; figure it out
class StochasticCoordinateDescentOptimizer:

    def __init__(self, 
        p, 
        get_objective,
        get_gradient,
        get_projected,
        epsilon=10**(-5),
        batch_size=1, 
        max_rounds=10,
        eta0=10**(-1),
        eta_scheduler=None,
        qn_server=None,
        theta_init=None):
        
        self.p = p
        self.get_objective = get_objective
        self.get_gradient = get_gradient
        self.get_projected = get_projected
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.eta0 = eta0

        if eta_scheduler is None:
            eta_scheduler = FS(self.eta0)

        self.eta_scheduler = eta_scheduler

        if qn_server is None:
            qn_server = SCDAS(self.p, beta1=0.99, beta2=0.999)

        self.qn = qn_server

        if theta_init is None:
            theta_init = self.get_projected(
                np.random.randn(self.d, 1) / self.d)

        self.theta_init = self.get_projected(theta_init)
        self.theta = None
        self.cushion = 0 if self.batch_size == 1 \
            else self.p % self.batch_size
        self.num_batches = int(self.p / self.batch_size)

        if self.cushion > 0:
            self.num_batches += 1

        self.objectives = []

    def get_parameters(self):

        return self.theta
        
    def run(self):

        i = 0
        converged = False
        order = np.arange(self.p)
        theta_t = np.copy(self.theta_init)
        theta_t1 = np.zeros_like(theta_t)

        self.objectives.append(
            self.get_objective(theta_t))

        while i < self.max_rounds and not converged:
            np.random.shuffle(order)

            batches = None
            eta = self.eta_scheduler.get_stepsize()

            if self.batch_size > 1:
                batches = np.hstack([
                    order,
                    order[:self.cushion]])
                batches = np.sort(
                    batches.reshape((
                        self.num_batches,
                        self.batch_size)))
            else:
                batches = order
            
            for batch in batches:
                grad = self.get_gradient(
                    theta_t1,
                    batch)

                theta_t1[batch,:] = self.qn.get_update(
                    theta_t[batch,:],
                    grad,
                    eta,
                    batch)
                # TODO: is there a more elegant way to do the projection?
                theta_t1 = self.get_projected(theta_t1)

            self.objectives.append(
                self.get_objective(theta_t1))
            
            diff = self.objectives[-1] - self.objectives[-2]
            converged = self.epsilon > np.abs(diff)
            theta_t = np.copy(theta_t1) 

            i += 1

        self.theta = theta_t1
