import numpy as np

from models.regression import L2RegularizedLogisticRegressionModel as L2RLRM
from models.cca import AppGradModel as AGM
from prinfo.models import CCAPenalizedLUPIModel as CCAPLUPIM
from fitterhappier.zeroorder import HyperBandOptimizer
from whitehorses.utils import get_random_k_folds as get_rkf
from drrobert.stats import get_binary_classification_eval as get_bce

class CCAPenalizedLogisticRegressionLUPIFullAdaGradTesterWithHPSelection:

    def __init__(self,
        data_server,
        fold_k=5,
        max_rounds=10000,
        epsilon=10**(-5),
        zero_order=None):

        self.ds = data_server
        self.fold_k = fold_k
        self.max_rounds = max_rounds

        if zero_order is None:
            max_iter = int(self.max_rounds / 1000)
            zero_order = HyperBandOptimizer(
                self.get_hyperparameter_sample,
                self.get_validation_loss,
                max_iter=max_iter)

        self.zero_order = zero_order

        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.theta = None
        self.objectives = None

    def get_parameters(self):

        return self.theta

    def run(self):

        self.zero_order.run()

        hps = self.zero_order.get_parameters()
        model = self._get_model(hps)
        optimizer = self._get_fold_optimizer(
            model,
            self.data,
            self.max_rounds,
            hps)

        optimizer.run()

        self.theta = optimizer.get_parameters()
        self.objectives = optimizer.objectives

    def get_validation_loss(self, hyperparameters, num_iters):

        print('new hps', hyperparameters)
        (X_o, X_p, y) = self.data
        evaluations = []

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = self._get_model(hyperparameters)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data,
                num_iters,
                hps)

            fold_optimizer.run()

            evaluation = self._get_evaluation(
                fold_optimizer,
                X_o[holdout,:],
                y[holdout,:])

            evaluations.append(evaluation['auroc'])

        return sum(evaluations) / self.fold_k

    def _get_evaluation(self, fold_optimizer, X_o, y):

        (w_o, _, Phi_o, _) = fold_optimizer.get_parameters()
        projected = np.dot(X, Phi_o)
        denom = 1 + np.exp(-np.dot(projected, w))
        y_hat = (np.power(denom, -1) > 0.5).astype(float)

        return get_bce(y, y_hat)

    def _get_model(self, hyperparameters):

        (lambda_s, lambda_p, gamma_o, gamma_p) = hyperparameters[:-3]
        o_model = BL2RLRM(self.k, gamma_o)
        p_model = BL2RLRM(self.k, gamma_p)
        (d_o, d_p) = self.ds.cols()
        s_model = AGM(d_o, d_p, self.k)

        return CCAPLUPIM(
            o_model,
            p_model,
            s_model,
            lambda_s,
            lambda_p)

    def _get_fold_optimizer(self, 
        model, 
        data, 
        num_iter,
        hps):

        (bs, deltas, eta0s) = hps[-3:]
        get_obj = lambda ws: model.get_objective(
            data, ws)

        def get_grad(ws):

            indexes = np.random.choice(
                data[0].shape[0], 
                size=bs, 
                replace=False)
            mb = (
                data[0][indexes,:],
                data[1][indexes,:],
                data[2][indexes,:])

            return model.get_gradient(mb, ws)

        get_proj = lambda ws: model.get_projected(
            data, ws)

        return FullAdaGradBlockCoordinateOptimizer(
            model.get_parameter_shape(),
            get_obj,
            get_grad,
            get_proj,
            eta0s=eta0s,
            deltas=deltas,
            epsilon=self.epsilon,
            max_rounds=num_iters*1000)

    def get_hyperparameter_sample(self):

        # TODO: double check that these ranges are reasonable
        lambda_s = log_uniform(low=-2, high=1)
        lambda_p = log_uniform(low=-2, high=1)
        gamma_o = log_uniform(low=-5, high=-1)
        gamma_p = log_uniform(low=-5, high=-1)
        bs = int(log_uniform(low=0, high=2))
        deltas = [log_uniform(low=-3, high=-1) for _ in range(4)]
        eta0s = [log_uniform(low=-3, high=-1) for _ in range(4)]

        return (lambda_s, lambda_p, gamma_o, gamma_p, bs, deltas, eta0s)

class CCAPenalizedLogisticRegressionLUPIFullAdaGradTester:

    def __init__(self,
        data_server,
        delta=10**(-3),
        eta0=1,
        theta=1,
        fold_k=5,
        batch_size=10,
        max_rounds=float('inf'),
        epsilon=10**(-3)):

        self.ds = data_server
        self.delta = delta
        self.eta0 = eta0
        self.fold_k = fold_k
        self.bs = batch_size
        self.max_rounds = max_rounds
        self.epsilon = epsilon

        self.folds = get_rkf(self.ds.rows(), fold_k)
        self.data = self.ds.get_data()
        self.ws = None
        self.objectives = []
        self.evaluations = []

    def run(self):

        (X_o, X_p, y) = self.data

        for (fold, holdout) in self.folds:
            fold_data = (X_o[fold,:], X_p[fold,:], y[fold,:])
            model = self._get_model(
                lambda_s,
                lambda_p,
                gamma_o,
                gamma_p)
            fold_optimizer = self._get_fold_optimizer(
                model,
                fold_data)

            fold_optimizer.run()

            self.objectives.append(
                fold_optimizer.objectives[-1])

            evaluation = self._get_evaluation(
                fold_optimizer,
                X_o[holdout,:],
                y[holdout,:])

            self.evaluations.append(evaluation)

        print('Mean AUROC:', 
            sum([e['auroc'] for e in self.evaluations]) / self.fold_k)
        print('Final objective values:', self.objectives)

    def _get_evaluation(self, fold_optimizer, X_o, y):

        (w_o, _, Phi_o, _) = fold_optimizer.get_parameters()
        projected = np.dot(X, Phi_o)
        denom = 1 + np.exp(-np.dot(projected, w))
        y_hat = (np.power(denom, -1) > 0.5).astype(float)

        return get_bce(y, y_hat)

    def _get_model(self, lambda_s, lambda_p, gamma_o, gamma_p):

        o_model = BL2RLRM(self.k, gamma_o)
        p_model = BL2RLRM(self.k, gamma_p)
        (d_o, d_p) = self.ds.cols()
        s_model = AGM(d_o, d_p, self.k)

        return CCAPLUPIM(
            o_model,
            p_model,
            s_model,
            lambda_s,
            lambda_p)

    def _get_fold_optimizer(self, model, data):
        
        get_obj = lambda ws: model.get_objective(
            data, ws)
        get_proj = lambda ws: model.get_projected(
            data, ws)

        def get_grad(ws):

            indexes = np.random.choice(
                data[0].shape[0], 
                size=self.bs, 
                replace=False)
            mb = (
                data[0][indexes,:],
                data[1][indexes,:],
                data[2][indexes,:])

            return model.get_gradient(mb, ws)

        return FullAdaGradBlockCoordinateOptimizer(
            model.parameter_shape()
            get_obj,
            get_grad,
            get_proj,
            eta0s=self.eta0s,
            deltas=self.deltas,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds)
