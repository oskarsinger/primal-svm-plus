import numpy as np

from drrobert.stats import get_binary_classification_eval as get_bce

# TODO: make all of these work for non-linear kernel

def get_dual_evaluation(
    fold_optimizer, 
    train_X, 
    train_y, 
    test_X, 
    test_y):

    alphas = fold_optimizer.get_parameters()[:train_X.shape[0],:]

    print('\tnum nonzeros:', np.count_nonzero(alphas))

    params = np.dot((alphas * train_y).T, train_X).T
    y_hat = np.sign(np.dot(test_X, params))

    return get_bce(test_y, y_hat)

def get_primal_evaluation(
    fold_optimizer, 
    test_X, 
    test_y):

    w = fold_optimizer.get_parameters()[:test_X.shape[1],:]
    y_hat = np.sign(np.dot(test_X, w))

    return get_bce(test_y, y_hat)
