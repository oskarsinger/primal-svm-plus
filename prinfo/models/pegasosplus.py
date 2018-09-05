import numpy as np

from ..utils import get_thresholded

class PegasosHingeLossSVMPlusWithSlacksModel:

    def __init__(self, 
        d_o,
        d_p,
        gamma=0.5, 
        c=10,
        theta=1, 
        i=None):

        self.d_o = d_o
        self.d_p = d_p
        self.d = self.d_o + self.d_p
        self.gamma = gamma
        self.c = c
        self.theta = theta
        self.lam_o = 1.0 / (self.c * self.theta)
        self.lam_p = self.gamma * self.lam_o
        self.i = i

        self.num_rounds = 0

    def get_gradient(self, data, params):

        self.num_rounds += 1
        
        # Extract data and params
        (X_o, X_p, y) = data
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]

        # Compute hinge loss terms
        y_hat_mag = np.dot(X_o, w_o)
        y_prod = y * y_hat_mag
        X_p_prod = np.dot(X_p, w_p)
        y_and_X_p_prods = y_prod + X_p_prod
        lt_one = (y_and_X_p_prods < 1)[:,0]

        # Compute w_o gradient
        data_o_term = np.mean(
            y[lt_one,:] * X_o[lt_one,:],
            axis=0)[:,np.newaxis] / self.lam_o
        w_o_grad = w_o - data_o_term

        # Compute w_p gradient
        data_p_term1 = np.mean(
            X_p[lt_one,:],
            axis=0)[:,np.newaxis] / self.lam_p
        data_p_term2 = np.mean(
            X_p[(X_p_prod < 0)[:,0],:],
            axis=0)[:,np.newaxis] / (self.theta * self.lam_p)
        w_p_grad = w_p - data_p_term1 - data_p_term2

        return np.vstack([w_o_grad, w_p_grad])

    def get_objective(self, data, params):

        # Extract parameters
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]

        # Computer regularization terms
        reg_o_term = self.lam_o * np.linalg.norm(w_o)**2 / 2
        reg_p_term = self.lam_p * np.linalg.norm(w_p)**2 / 2

        # Compute residual terms
        residuals = self.get_residuals(data, params)   
        data_term = np.mean(residuals)

        return reg_o_term + reg_p_term + data_term

    def get_residuals(self, data, params):

        # Extract parameters and data
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]
        (X_o, X_p, y) = data

        # Compute first hinge term residuals
        y_hat_mag = np.dot(X_o, w_o)
        y_prod = y * y_hat_mag 
        X_p_prod = np.dot(X_p, w_p)
        y_and_X_p_prods = y_prod + X_p_prod
        threshd1 = get_thresholded(
            1 - y_and_X_p_prods, lower=0)

        # Compute second hinge term residuals
        threshd2 = get_thresholded(
            X_p_prod, lower=0)

        return threshd1 + threshd2 / self.theta

    def get_datum(self, data, i):

        (X_o, X_p, y) = data
        X_o_i = X_o[i,:][np.newaxis,:]
        X_p_i = X_p[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return (X_o_i, X_p_i, y_i)

    def get_projected(self, data, params):

        # Extract params
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]

        # Extract scales
        scale_o = (np.linalg.norm(w_o) * np.sqrt(self.lam_o))**(-1)
        scale_p = (np.linalg.norm(w_p) * np.sqrt(self.lam_p))**(-1)

        # Project w_o and w_p
        w_o_projected = min([1, scale_o]) * w_o
        w_p_projected = min([1, scale_p]) * w_p

        return np.vstack([w_o_projected, w_p_projected])
