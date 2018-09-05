import numpy as np

from prinfo.models import Li2016SVMPlusWithMissingPrivilegedInformation as L2016SVMPWMPI
from fitterhappier.qn import DiagonalAdamOptimizer
from fitterhappier.coordinate import StochasticCoordinateDescentOptimizer as SCDO
from fitterhappier.qn import StochasticCoordinateDiagonalAdamServer as SCDAS
from fitterhappier.zeroorder import HyperBandOptimizer
from whitehorses.utils import get_random_k_folds as get_rkf
from drrobert.stats import get_binary_classification_eval as get_bce

class Li2016WMPIAdamTester:

    def __init__(self,
        data_server, 
        fold_k=5,
        max_rounds=10000,
        epsilon=10**(-5),
        o_kernel=None, 
        p_kernel=None,
        get_zero_order=None):

        self.ds = data_server
        self.fold_k = fold_k
        self.max_rounds = max_rounds
        self.epsilon = epsilon

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
        self.theta = None
        self.objectives = None

    def get_parameters(self):

        return self.theta

    def run(self):

        self.zero_order.run()

        (c, gamma, delta, beta1, beta2, eta0) = self.zero_order.get_parameters()
        model = L2016SVMPWMPI(
            c, 
            gamma, 
            self.o_kernel, 
            self.p_kernel)
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

        (c, gamma, delta, beta1, beta2, eta0) = hyperparameters
        (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing) = self.data
        num_missing = X_o_missing.shape[0]
        num_not_missing = X_o_missing.shape[0]
        evaluations = []

        # Need to adapt this to missing and non-missing data
        for (fold, holdout) in self.folds:
            (fold_missing, fold_not_missing) = (
                fold[fold < num_missing],
                fold[fold >= num_missing] - num_missing)
            fold_num_missing = fold_missing.shape[0]
            fold_num_not_missing = fold_not_missing.shape[0]
            fold_data = (
                X_o_missing[fold_missing,:], 
                X_o_not_missing[fold_not_missing,:],
                X_p[fold_not_missing,:], 
                y_missing[fold_missing,:],
                y_not_missing[fold_not_missing,:])
            model = L2016SVMPWMPI(
                c, 
                gamma, 
                self.o_kernel, 
                self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                beta1,
                beta2,
                eta0)

            fold_optimizer.run()

            # Holdout indexing
            (holdout_missing, holdout_not_missing) = (
                holdout[holdout < num_missing],
                holdout[holdout >= num_missing] - num_missing)
            holdout_X_o_missing = X_o_missing[holdout_missing,:]
            holdout_X_o_not_missing = X_o_not_missing[holdout_not_missing,:]
            holdout_y_missing = y_missing[holdout_missing,:]
            holdout_y_not_missing = y_not_missing[holdout_not_missing,:]

            # Retrieve alphas
            dual_params = fold_optimizer.get_parameters()
            alpha_missing = dual_params[:fold_num_missing,:]
            alpha_not_missing = dual_params[fold_num_missing:fold.shape[0],:]
            alphas = np.vstack([alpha_missing, alpha_not_missing])
            print('\tnum nonzeros:', np.count_nonzero(alphas))

            # TODO: make this work for non-linear kernel
            # Retrieve primal parameters
            fold_y = np.vstack([fold_data[-2], fold_data[-1]])
            fold_X_o = np.vstack([fold_data[0], fold_data[1]])
            params = np.dot((alphas * fold_y).T, fold_X_o).T

            # Retrieve and evaluate predictions on holdout
            holdout_X_o = np.vstack([
                holdout_X_o_missing, 
                holdout_X_o_not_missing])
            holdout_y = np.vstack([
                holdout_y_missing, 
                holdout_y_not_missing])
            y_hat = np.sign(np.dot(holdout_X_o, params))
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
        get_grad = lambda theta: model.get_gradient(
            data, theta)
        get_proj = lambda theta: model.get_projected(
            data, theta)
        num_coords = data[0].shape[0] + data[1].shape[0] * 2
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
            batch_size=batch_size,
            max_rounds=num_iters*100)

        return dao

    def get_hyperparameter_sample(self):

        c = np.random.uniform(low=1, high=10)
        gamma = log_uniform(low=-4, high=-1)
        delta = log_uniform(low=-5, high=-2)
        beta1 = np.random.uniform(low=0.99, high=0.99999)
        beta2 = np.random.uniform(low=0.99, high=0.99999)
        eta0 = log_uniform(low=-6, high=-5)

        return (c, gamma, delta, beta1, beta2, eta0)

class Li2016WMPISDCATester:

    def __init__(self,
        data_server, 
        fold_k=5,
        num_batches=1,
        max_rounds=10000,
        epsilon=10**(-5),
        o_kernel=None, 
        p_kernel=None,
        get_zero_order=None):

        self.ds = data_server
        self.fold_k = fold_k
        self.num_batches = num_batches
        self.max_rounds = max_rounds
        self.epsilon = epsilon

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
        self.theta = None
        self.objectives = None

    def get_parameters(self):

        return self.theta

    def run(self):

        self.zero_order.run()

        (c, gamma, delta, beta1, beta2, eta0) = self.zero_order.get_parameters()
        model = L2016SVMPWMPI(
            c, 
            gamma, 
            self.o_kernel, 
            self.p_kernel)
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

        (c, gamma, delta, beta1, beta2, eta0) = hyperparameters
        (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing) = self.data
        num_missing = X_o_missing.shape[0]
        num_not_missing = X_o_missing.shape[0]
        evaluations = []

        # Need to adapt this to missing and non-missing data
        for (fold, holdout) in self.folds:
            print('new fold')
            (fold_missing, fold_not_missing) = (
                fold[fold < num_missing],
                fold[fold >= num_missing] - num_missing)
            fold_num_missing = fold_missing.shape[0]
            fold_num_not_missing = fold_not_missing.shape[0]
            fold_data = (
                X_o_missing[fold_missing,:], 
                X_o_not_missing[fold_not_missing,:],
                X_p[fold_not_missing,:], 
                y_missing[fold_missing,:],
                y_not_missing[fold_not_missing,:])
            model = L2016SVMPWMPI(
                c, 
                gamma, 
                self.o_kernel, 
                self.p_kernel)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                delta,
                beta1,
                beta2,
                eta0)

            fold_optimizer.run()

            # Holdout indexing
            (holdout_missing, holdout_not_missing) = (
                holdout[holdout < num_missing],
                holdout[holdout >= num_missing] - num_missing)
            holdout_X_o_missing = X_o_missing[holdout_missing,:]
            holdout_X_o_not_missing = X_o_not_missing[holdout_not_missing,:]
            holdout_y_missing = y_missing[holdout_missing,:]
            holdout_y_not_missing = y_not_missing[holdout_not_missing,:]

            # Retrieve alphas
            dual_params = fold_optimizer.get_parameters()
            alpha_missing = dual_params[:fold_num_missing,:]
            alpha_not_missing = dual_params[fold_num_missing:fold.shape[0],:]
            alphas = np.vstack([alpha_missing, alpha_not_missing])
            print('\tnum nonzeros:', np.count_nonzero(alphas))

            # TODO: make this work for non-linear kernel
            # Retrieve primal parameters
            fold_y = np.vstack([fold_data[-2], fold_data[-1]])
            fold_X_o = np.vstack([fold_data[0], fold_data[1]])
            params = np.dot((alphas * fold_y).T, fold_X_o).T

            # Retrieve and evaluate predictions on holdout
            holdout_X_o = np.vstack([
                holdout_X_o_missing, 
                holdout_X_o_not_missing])
            holdout_y = np.vstack([
                holdout_y_missing, 
                holdout_y_not_missing])
            y_hat = np.sign(np.dot(holdout_X_o, params))
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
        num_coords = data[0].shape[0] + data[1].shape[0] * 2
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
        gamma = log_uniform(low=-4, high=-1)
        delta = log_uniform(low=-5, high=-2)
        beta1 = np.random.uniform(low=0.99, high=0.99999)
        beta2 = np.random.uniform(low=0.99, high=0.99999)
        eta0 = log_uniform(low=-6, high=-5)

        return (c, gamma, delta, beta1, beta2, eta0)
