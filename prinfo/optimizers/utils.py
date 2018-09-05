import numpy as np

from ..utils import get_thresholded

def get_shrunk_and_thresholded(x, lower=0):

    sign = np.sign(x)
    threshed = get_thresholded(
        np.absolute(x) - lower, lower=0)

    return sign * threshed

def get_mirror_update(
    parameters, 
    eta, 
    search_direction, 
    get_dual, 
    get_primal):

    dual_parameters = get_dual(parameters)
    dual_update = dual_parameters - eta * search_direction
    primal_parameters = get_primal(dual_update)

    return primal_parameters
