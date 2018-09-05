import numpy as np

from models.svm import Li2016SVMPlus as L2016SVMP
from fitterhappier.coordinate import StochasticCoordinateDescentOptimizer as SCDO
from fitterhappier.qn import DiagonalAdamOptimizer
from fitterhappier.qn import FullAdaGradOptimizer
from fitterhappier.qn import StochasticCoordinateDiagonalAdamServer as SCDAS
from fitterhappier.qn import BFGSSolver as BFGSS
from fitterhappier.zeroorder import HyperBandOptimizer
from whitehorses.utils import get_random_k_folds as get_rkf
from drrobert.stats import log_uniform
from .utils import get_dual_evaluation as get_evaluation

class Li2016FullAdaGradTester:

    def __init__(self,
        data_server, 
        c=10,
        gamma=0.5,
        delta=10**(-3),
        eta0=0.01,
        theta=None,
        fold_k=5,
        max_rounds=float('inf'),
        epsilon=10**(-3),
        o_kernel=None, 
        p_kernel=None):

        self.ds = data_server
        self.c = c
        self.gamma = gamma
        self.delta = delta
        self.eta0 = eta0
        self.theta = theta
        self.fold_k = fold_k
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if o_kernel is None:
            o_kernel = lambda x,y: np.dot(x.T, y)
    
        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x,y: np.dot(x.T, y)

        self.p_kernel = p_kernel
        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.w = None
        self.objectives = []
        self.evaluations = []

    def run(self):

        (X_o, X_p, y) = self.data

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = L2016SVMP(
                self.c, 
                self.gamma, 
                theta=self.theta,
                o_kernel=self.o_kernel, 
                p_kernel=self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data)

            fold_optimizer.run()

            self.objectives.append(
                fold_optimizer.objectives[-1])

            evaluation = get_evaluation(
                fold_optimizer,
                fold_data[0],
                fold_data[2],
                X_o[holdout,:],
                y[holdout,:])

            self.evaluations.append(evaluation)

            print('Mean AUROC:', 
                sum([e['auroc'] for e in self.evaluations]) / self.fold_k)
            print('Final objective values:', self.objectives)

    def _get_fold_optimizer(self, model, data):

        get_obj = lambda alpha: model.get_objective(
            data, alpha)
        get_grad = lambda alpha: model.get_gradient(
            data, alpha)
        get_proj = lambda alpha: model.get_projected(
            data, alpha)
        init_scale = self.c / (self.gamma * data[0].shape[0])
        alphabeta_init = init_scale * np.ones((data[0].shape[0]*2, 1))

        return FullAdaGradOptimizer(
            data[0].shape[0] * 2,
            get_obj,
            get_grad,
            get_proj,
            eta0=self.eta0,
            delta=self.delta,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            theta_init=alphabeta_init)

class Li2016AdamTester:

    def __init__(self,
        data_server, 
        c=10,
        gamma=0.5,
        delta=10**(-3),
        beta1=0.999,
        beta2=0.9999,
        eta0=0.01,
        theta=None,
        fold_k=5,
        max_rounds=float('inf'),
        epsilon=10**(-3),
        o_kernel=None, 
        p_kernel=None):

        self.ds = data_server
        self.c = c
        self.gamma = gamma
        self.delta = delta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta0 = eta0
        self.theta = theta
        self.fold_k = fold_k
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if o_kernel is None:
            o_kernel = lambda x,y: np.dot(x.T, y)
    
        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x,y: np.dot(x.T, y)

        self.p_kernel = p_kernel
        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.w = None
        self.objectives = []
        self.evaluations = []

    def run(self):

        (X_o, X_p, y) = self.data

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = L2016SVMP(
                self.c, 
                self.gamma, 
                theta=self.theta,
                o_kernel=self.o_kernel, 
                p_kernel=self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data)

            fold_optimizer.run()

            self.objectives.append(
                fold_optimizer.objectives[-1])

            evaluation = get_evaluation(
                fold_optimizer,
                fold_data[0],
                fold_data[2],
                X_o[holdout,:],
                y[holdout,:])

            self.evaluations.append(evaluation)

            print('Mean AUROC:', 
                sum([e['auroc'] for e in self.evaluations]) / self.fold_k)
            print('Final objective values:', self.objectives)

    def _get_fold_optimizer(self, 
        model, 
        data):

        get_obj = lambda alpha: model.get_objective(
            data, alpha)
        get_grad = lambda alpha: model.get_gradient(
            data, alpha)
        get_proj = lambda alpha: model.get_projected(
            data, alpha)
        init_scale = self.c / (self.gamma * data[0].shape[0])
        alphabeta_init = init_scale * np.ones((data[0].shape[0]*2, 1))

        return DiagonalAdamOptimizer(
            data[0].shape[0] * 2,
            get_obj,
            get_grad,
            get_proj,
            eta0=self.eta0,
            beta1=self.beta1,
            beta2=self.beta2,
            delta=self.delta,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            theta_init=alphabeta_init)

class Li2016BFGSTester:

    def __init__(self,
        data_server, 
        c,
        gamma,
        o_kernel=None, 
        p_kernel=None,
        max_rounds=10,
        epsilon=10**(-5),
        w_init=None):

        self.ds = data_server
        self.data = self.ds.get_data()
        self.c = c
        self.gamma = gamma
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if o_kernel is None:
            o_kernel = lambda x,y: np.dot(x.T, y)
    
        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x,y: np.dot(x.T, y)

        self.p_kernel = p_kernel

        if w_init is None:
            w_init = np.zeros((
                self.ds.rows() * 2, 1))
        else:
            w_init = np.copy(w_init)

        self.w_init = w_init
        self.model = L2016SVMP(
            self.c, 
            self.gamma,
            self.o_kernel,
            self.p_kernel)
        self.w = None
        self.objectives = None

    def get_parameters(self):

        return self.w

    def run(self):

        get_obj = lambda alpha: self.model.get_objective(
            self.data, alpha)
        get_grad = lambda alpha: self.model.get_gradient(
            self.data, alpha)
        get_projected = lambda alpha: self.model.get_projected(
            self.data, alpha)
        bfgs = BFGSS(
            self.ds.rows() * 2,
            get_obj,
            get_grad,
            get_projected,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            w_init=self.w_init)

        bfgs.run()

        self.w = bfgs.get_parameters()
        self.objectives = bfgs.objectives

class Li2016AdamTesterWithHPSelection:

    def __init__(self,
        data_server, 
        fold_k=5,
        max_rounds=10000,
        epsilon=10**(-5),
        o_kernel=None, 
        p_kernel=None,
        get_zero_order=None,
        mixture=False):

        self.ds = data_server
        self.fold_k = fold_k
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.mixture = mixture

        if o_kernel is None:
            o_kernel = lambda x,y: np.dot(x.T, y)
    
        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x,y: np.dot(x.T, y)

        self.p_kernel = p_kernel
        zero_order = None

        if get_zero_order is None:
            max_iter = int(self.max_rounds / 100)
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

        hps = self.zero_order.get_parameters()
        (c, gamma, delta, beta1, beta2, eta0, theta) = hps
        model = L2016SVMP(
            c, 
            gamma, 
            theta=theta,
            o_kernel=self.o_kernel, 
            p_kernel=self.p_kernel)
        optimizer = self._get_fold_optimizer(
            model,
            self.data,
            self.max_rounds / 100,
            delta,
            beta1,
            beta2,
            eta0)

        optimizer.run()

        self.w = optimizer.get_parameters()
        self.objectives = optimizer.objectives

    def get_validation_loss(self, hyperparameters, num_iters):

        print('new hps', hyperparameters)
        (c, gamma, delta, beta1, beta2, eta0, theta) = hyperparameters
        (X_o, X_p, y) = self.data
        evaluations = []

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = L2016SVMP(
                c, 
                gamma, 
                theta=theta,
                o_kernel=self.o_kernel, 
                p_kernel=self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                beta1,
                beta2,
                eta0)

            fold_optimizer.run()

            evaluation = get_evaluation(
                fold_optimizer,
                fold_data[0],
                fold_data[2],
                X_o[holdout,:],
                y[holdout,:])

            evaluations.append(evaluation['auroc'])

        return sum(evaluations) / self.fold_k

    def _get_fold_optimizer(self, 
        model, 
        data, 
        num_iters, 
        delta,
        beta1,
        beta2,
        eta0):

        get_obj = lambda alpha: model.get_objective(
            data, alpha)
        get_grad = lambda alpha: model.get_gradient(
            data, alpha)
        get_proj = lambda alpha: model.get_projected(
            data, alpha)
        num_coords = data[0].shape[0] * 2
        dao = DiagonalAdamOptimizer(
            num_coords,
            get_obj,
            get_grad,
            get_proj,
            eta0=eta0,
            beta1=beta1,
            beta2=beta2,
            delta=delta,
            epsilon=self.epsilon,
            max_rounds=num_iters*100)

        return dao

    def get_hyperparameter_sample(self):

        c = np.random.uniform(low=1, high=100)
        gamma = log_uniform(low=-4, high=-1)
        delta = log_uniform(low=-5, high=-1)
        beta1 = np.random.uniform(low=0.99, high=0.99999)
        beta2 = np.random.uniform(low=0.99, high=0.99999)
        eta0 = log_uniform(low=-6, high=-2)
        theta = log_uniform(low=-1, high=2) if self.mixture else None
        
        return (c, gamma, delta, beta1, beta2, eta0, theta)

class Li2016SDCATesterWithHPSelection:

    def __init__(self,
        data_server, 
        fold_k=5,
        num_batches=1,
        max_rounds=10000,
        epsilon=10**(-5),
        o_kernel=None, 
        p_kernel=None,
        get_zero_order=None,
        mixture=False):

        self.ds = data_server
        self.fold_k = fold_k
        self.num_batches = num_batches
        self.max_rounds = max_rounds
        self.epsilon = epsilon
        self.mixture = mixture

        if o_kernel is None:
            o_kernel = lambda x,y: np.dot(x.T, y)
    
        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x,y: np.dot(x.T, y)

        self.p_kernel = p_kernel
        zero_order = None

        if get_zero_order is None:
            max_iter = int(self.max_rounds / 100)
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

        hps = self.zero_order.get_parameters()
        (c, gamma, delta, beta1, beta2, eta0, theta) = hps
        model = L2016SVMP(
            c, 
            gamma, 
            theta=theta,
            o_kernel=self.o_kernel, 
            p_kernel=self.p_kernel)
        optimizer = self._get_fold_optimizer(
            model,
            self.data,
            self.max_rounds / 100,
            delta,
            beta1,
            beta2,
            eta0)

        optimizer.run()

        self.w = optimizer.get_parameters()
        self.objectives = optimizer.objectives

    def get_validation_loss(self, hyperparameters, num_iters):

        print('new hps')
        (c, gamma, delta, beta1, beta2, eta0, theta) = hyperparameters
        (X_o, X_p, y) = self.data
        evaluations = []

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = L2016SVMP(
                c, 
                gamma, 
                theta=theta,
                o_kernel=self.o_kernel, 
                p_kernel=self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                beta1,
                beta2,
                eta0)

            fold_optimizer.run()

            evaluation = get_evaluation(
                fold_optimizer,
                fold_data[0],
                fold_data[2],
                X_o[holdout,:],
                y[holdout,:])

            evaluations.append(evaluation['auroc'])

        return sum(evaluations) / self.fold_k

    def _get_fold_optimizer(self, 
        model, 
        data, 
        num_iters, 
        delta,
        beta1,
        beta2,
        eta0):

        get_obj = lambda alpha: model.get_objective(
            data, alpha)
        get_grad = lambda alpha, batch: model.get_gradient(
            data, alpha, batch=batch)
        get_proj = lambda alpha: model.get_projected(
            data, alpha)
        num_coords = data[0].shape[0] * 2
        batch_size = int(num_coords / self.num_batches)
        qn_server = SCDAS(
            num_coords,
            beta1=beta1,
            beta2=beta2,
            delta=delta)
        scdo = SCDO(
            num_coords,
            get_obj,
            get_grad,
            get_proj,
            eta0=eta0,
            qn_server=qn_server,
            epsilon=self.epsilon,
            batch_size=batch_size,
            max_rounds=num_iters*100)

        return scdo

    def get_hyperparameter_sample(self):

        c = np.random.uniform(low=1, high=100)
        gamma = log_uniform(low=-4, high=-1)
        delta = log_uniform(low=-5, high=-1)
        beta1 = np.random.uniform(low=0.99, high=0.99999)
        beta2 = np.random.uniform(low=0.99, high=0.99999)
        eta0 = log_uniform(low=-6, high=-2)
        theta = log_uniform(low=-1, high=2) if self.mixture else None
        
        return (c, gamma, delta, beta1, beta2, eta0, theta)
