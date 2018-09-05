import numpy as np
import numpy.random as npr

from sklearn.metrics import roc_auc_score

def get_binary_classification_eval(y, y_hat):

    get_fc = lambda x: float(np.count_nonzero(x))
    p = get_fc(y > 0)
    p_hat = get_fc(y_hat > 0)
    n = get_fc(y < 0)
    n_hat = get_fc(y_hat < 0)
    tp = get_fc(np.logical_and(y_hat == y, y_hat > 0))
    tn = get_fc(np.logical_and(y_hat == y, y_hat < 0))
    accuracy = (tp + tn) / y.shape[0]
    sensitivity = tp / p
    specificity = tn / n
    precision = 1 if p_hat == 0 else tp / p_hat
    f1 = 0 if precision + sensitivity == 0 else \
        2 * precision * sensitivity / (precision + sensitivity)
    auroc = roc_auc_score(y, y_hat)

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auroc': auroc}

def get_random_k_folds(N, k):

    indexes = np.arange(N)
    
    np.random.shuffle(indexes)

    size = int(N / k)
    holdouts = [indexes[size*i:size*(i+1)]
                for i in range(k)]
    folds = [np.hstack(holdouts[:i] + holdouts[i+1:])
             for i in range(k)]

    return list(zip(folds, holdouts))
    
def log_uniform(upper, lower, size=1):

    log_u = np.log(upper)
    log_l = np.log(lower)
    logs = npr.uniform(
        low=log_l, high=log_u, size=size)

    if size == 1:
        logs = logs[0]

    return np.exp(logs)

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

    return get_binary_classification_eval(test_y, y_hat)

def get_primal_evaluation(
    fold_optimizer, 
    test_X, 
    test_y):

    w = fold_optimizer.get_parameters()[:test_X.shape[1],:]
    y_hat = np.sign(np.dot(test_X, w))

    return get_binary_classification_eval(test_y, y_hat)
