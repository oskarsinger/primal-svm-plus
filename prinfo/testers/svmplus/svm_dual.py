import numpy as np

from models.svm import SupportVectorMachineDualModel as SVMD
from fitterhappier.coordinate import StochasticCoordinateDescentOptimizer as SCDO
from fitterhappier.qn import StochasticCoordinateDiagonalAdamServer as SCDAS
from fitterhappier.zeroorder import HyperBandOptimizer
from whitehorses.utils import get_random_k_folds as get_rkf
from drrobert.stats import get_binary_classification_eval as get_bce

class SVMSDCATester:

    def __init__(self,
        data_server,
        fold_k=5,
        num_batches=1,
        max_rounds=10000,
        epsilon=10**(-5),
        kernel=None, 
        get_zero_order=None):

        self.ds = data_server
        self.fold_k = fold_k
        self.num_batches = num_batches
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        if kernel is None:
            kernel = lambda x,y: np.dot(x.T, y)
    
        self.kernel = kernel
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
        self.theta = None
        self.objectives = None

    def get_parameters(self):

        return self.theta

    def run(self):

        self.zero_order.run()

        (c, delta, beta1, beta2, eta0) = self.zero_order.get_parameters()
        model = SVMD(c, self.kernel)
        optimizer = self._get_fold_optimizer(
            model,
            self.data,
            self.max_rounds / 100,
            delta,
            beta1,
            beta2,
            eta0)

        optimizer.run()

        self.theta = optimizer.get_parameters()
        self.objectives = optimizer.objectives

    def get_validation_loss(self, hyperparameters, num_iters):

        print('new hps')
        (c, delta, beta1, beta2, eta0) = hyperparameters
        (X, y) = self.data
        evaluations = []

        for (fold, holdout) in self.folds:
            print('new fold')
            fold_data = (X[fold,:], y[fold,:])
            model = SVMD(c, self.kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                beta1,
                beta2,
                eta0)

            fold_optimizer.run()

            holdout_X = X[holdout,:]
            holdout_y = y[holdout,:]
            alphas = fold_optimizer.get_parameters()[:fold_data[0].shape[0],:]
            print('\tnum nonzeros:', np.count_nonzero(alphas))

            # TODO: make this work for non-linear kernel
            params = np.dot((alphas * fold_data[1]).T, fold_data[0]).T
            y_hat = np.sign(np.dot(holdout_X, params))
            evaluation = get_bce(holdout_y, y_hat)

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

        get_obj = lambda theta: model.get_objective(
            data, theta)
        get_grad = lambda theta, batch: model.get_gradient(
            data, theta, batch=batch)
        get_proj = lambda theta: model.get_projected(
            data, theta)
        num_coords = data[0].shape[0]
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

        c = np.random.uniform(low=1, high=10)
        delta = np.random.uniform(low=10**(-5), high=10**(-2))
        beta1 = np.random.uniform(low=0.99, high=0.99999)
        beta2 = np.random.uniform(low=0.99, high=0.99999)
        eta0 = np.random.uniform(low=10**(-3), high=10**(-1))

        return (c, delta, beta1, beta2, eta0)
