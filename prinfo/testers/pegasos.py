import numpy as np

from prinfo.models import PegasosHingeLossSVMPlusWithSlacksModel as PHLSVMPWSM
from fitterhappier.qn import DiagonalAdamOptimizer
from fitterhappier.qn import FullAdaGradOptimizer
from fitterhappier.qn import EmpiricalNaturalGradientOptimizer
from fitterhappier.zeroorder import HyperBandOptimizer
from whitehorses.utils import get_random_k_folds as get_rkf
from drrobert.stats import log_uniform
from .utils import get_primal_evaluation as get_evaluation

class PegasosSVMPlusFullAdaGradTesterWithHPSelection:

    def __init__(self,
        data_server, 
        fold_k=5,
        max_rounds=50000,
        epsilon=10**(-5),
        get_zero_order=None):

        self.ds = data_server
        self.fold_k = fold_k
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        zero_order = None

        if get_zero_order is None:
            max_iter = int(self.max_rounds / 1000)
            zero_order = HyperBandOptimizer(
                self.get_hyperparameter_sample,
                self.get_validation_loss,
                max_iter=max_iter)
        else:
            zero_order = get_zero_order(
                self.get_hyperparameter_sample,
                self.get_validation_loss)

        self.zero_order = zero_order

        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.w = None
        self.objectives = None

    def get_parameters(self):

        return self.w

    def run(self):

        self.zero_order.run()

        (X_o, X_p, y) = self.data
        hps = self.zero_order.get_parameters()
        (c, gamma, delta, eta0, theta, bs) = hps
        model = PHLSVMPWSM(
            X_o.shape[1],
            X_p.shape[1],
            c=c, 
            gamma=gamma, 
            theta=theta)
        optimizer = self._get_fold_optimizer(
            model,
            self.data,
            self.max_rounds / 1000,
            delta,
            eta0,
            bs)

        optimizer.run()

        self.w = optimizer.get_parameters()
        self.objectives = optimizer.objectives

    def get_validation_loss(self, hyperparameters, num_iters):

        print('new hps', hyperparameters)
        (c, gamma, delta, eta0, theta, bs) = hyperparameters
        (X_o, X_p, y) = self.data
        evaluations = []

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = PHLSVMPWSM(
                fold_data[0].shape[1],
                fold_data[1].shape[1],
                c=c, 
                gamma=gamma, 
                theta=theta)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                eta0,
                bs)

            fold_optimizer.run()

            evaluation = get_evaluation(
                fold_optimizer,
                X_o[holdout,:],
                y[holdout,:])

            evaluations.append(evaluation['auroc'])

        return sum(evaluations) / self.fold_k

    def _get_fold_optimizer(self, 
        model, 
        data, 
        max_rounds, 
        delta, 
        eta0, 
        bs):
        
        get_obj = lambda w: model.get_objective(
            data, w)
        get_proj = lambda w: model.get_projected(
            data, w)

        def get_grad(w):

            indexes = np.random.choice(
                data[0].shape[0], 
                size=bs, 
                replace=False)
            mb = (
                data[0][indexes,:],
                data[1][indexes,:],
                data[2][indexes,:])

            return model.get_gradient(mb, w)

        data_mean = np.mean(
            np.hstack([data[0], data[1]]),
            axis=0)[:,np.newaxis]
        scale = np.linalg.norm(data_mean) + model.lam_o + model.lam_p
        w_init = data_mean / scale

        return FullAdaGradOptimizer(
            data[0].shape[1] + data[1].shape[1],
            get_obj,
            get_grad,
            get_proj,
            eta0=eta0,
            delta=delta,
            epsilon=self.epsilon,
            max_rounds=max_rounds * 1000,
            theta_init=w_init)

    def get_hyperparameter_sample(self):

        c = log_uniform(low=0, high=1)
        gamma = log_uniform(low=-4, high=-1)
        delta = log_uniform(low=-5, high=-1)
        eta0 = log_uniform(low=-3, high=-1)
        theta = log_uniform(low=-2, high=1)
        bs = int(log_uniform(low=0, high=2))
        
        return (c, gamma, delta, eta0, theta, bs)

class PegasosSVMPlusEmpiricalNaturalGradientTester:

    def __init__(self,
        data_server,
        c=10,
        gamma=0.5,
        eta0=1,
        theta=1,
        fold_k=5,
        batch_size=10,
        max_rounds=float('inf'),
        epsilon=10**(-3)):

        self.ds = data_server
        self.c = c
        self.gamma = gamma
        self.eta0 = eta0
        self.theta = theta
        self.fold_k = fold_k
        self.bs = batch_size
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.w = None
        self.objectives = []
        self.evaluations = []

    def run(self):

        (X_o, X_p, y) = self.data

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = PHLSVMPWSM(
                X_o.shape[1],
                X_p.shape[1],
                c=self.c, 
                gamma=self.gamma, 
                theta=self.theta)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data)

            fold_optimizer.run()

            self.objectives.append(
                fold_optimizer.objectives[-1])

            evaluation = get_evaluation(
                fold_optimizer,
                X_o[holdout,:],
                y[holdout,:])

            self.evaluations.append(evaluation)

        print('Mean AUROC:', 
            sum([e['auroc'] for e in self.evaluations]) / self.fold_k)
        print('Final objective values:', self.objectives)

    def _get_fold_optimizer(self, model, data):
        
        get_obj = lambda w: model.get_objective(
            data, w)
        get_proj = lambda w: model.get_projected(
            data, w)

        def get_grad(w):

            indexes = np.random.choice(
                data[0].shape[0], 
                size=self.bs, 
                replace=False)
            mb = (
                data[0][indexes,:],
                data[1][indexes,:],
                data[2][indexes,:])

            return model.get_gradient(mb, w)

        data_mean = np.mean(
            np.hstack([data[0], data[1]]),
            axis=0)[:,np.newaxis]
        scale = np.linalg.norm(data_mean) + model.lam_o + model.lam_p
        w_init = data_mean / scale

        return EmpiricalNaturalGradientOptimizer(
            data[0].shape[1] + data[1].shape[1],
            get_obj,
            get_grad,
            get_proj,
            eta0=self.eta0,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            theta_init=w_init)

class PegasosSVMPlusFullAdaGradTester:

    def __init__(self,
        data_server,
        c=10,
        gamma=0.5,
        delta=10**(-3),
        eta0=1,
        theta=1,
        fold_k=5,
        batch_size=10,
        max_rounds=float('inf'),
        epsilon=10**(-3)):

        self.ds = data_server
        self.c = c
        self.gamma = gamma
        self.delta = delta
        self.eta0 = eta0
        self.theta = theta
        self.fold_k = fold_k
        self.bs = batch_size
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.w = None
        self.objectives = []
        self.evaluations = []

    def run(self):

        (X_o, X_p, y) = self.data

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = PHLSVMPWSM(
                X_o.shape[1],
                X_p.shape[1],
                c=self.c, 
                gamma=self.gamma, 
                theta=self.theta)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data)

            fold_optimizer.run()

            self.objectives.append(
                fold_optimizer.objectives[-1])

            evaluation = get_evaluation(
                fold_optimizer,
                X_o[holdout,:],
                y[holdout,:])

            self.evaluations.append(evaluation)

        print('Mean AUROC:', 
            sum([e['auroc'] for e in self.evaluations]) / self.fold_k)
        print('Final objective values:', self.objectives)

    def _get_fold_optimizer(self, model, data):
        
        get_obj = lambda w: model.get_objective(
            data, w)
        get_proj = lambda w: model.get_projected(
            data, w)

        def get_grad(w):

            indexes = np.random.choice(
                data[0].shape[0], 
                size=self.bs, 
                replace=False)
            mb = (
                data[0][indexes,:],
                data[1][indexes,:],
                data[2][indexes,:])

            return model.get_gradient(mb, w)

        data_mean = np.mean(
            np.hstack([data[0], data[1]]),
            axis=0)[:,np.newaxis]
        scale = np.linalg.norm(data_mean) + model.lam_o + model.lam_p
        w_init = data_mean / scale

        return FullAdaGradOptimizer(
            data[0].shape[1] + data[1].shape[1],
            get_obj,
            get_grad,
            get_proj,
            eta0=self.eta0,
            delta=self.delta,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            theta_init=w_init)
