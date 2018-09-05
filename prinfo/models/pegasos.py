import numpy as np

from ..utils import get_thresholded

class PegasosHingeLossSVMModel:

    def __init__(self, p, i=None, lam=10**(-5)):

        self.p = p
        self.idn = i
        self.lam = lam

        self.num_rounds = 0

    def get_gradient(self, data, params):

        self.num_rounds += 1

        (X, y) = data
        k = X.shape[0]
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag
        data_term = np.sum(
            y[y_prod < 1] * X[y_prod < 1],
            axis=0) / (k * self.lam)

        return params - data_term

    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)   
        data_term = np.sum(residuals) / residuals.shape[0]
        reg_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return reg_term + data_term

    def get_residuals(self, data, params):

        (X, y) = data
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag 
        threshd = get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        (X, y) = data
        x_i = X[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return (x_i, y_i)

    def get_projected(self, data, params):

        norm = np.linalg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)
