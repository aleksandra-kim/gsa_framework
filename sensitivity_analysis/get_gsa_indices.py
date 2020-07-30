import numpy as np


########################
### 1. Sobol indices ###
########################
def separate_output_values(y, n_samples, n_inputs, calc_second_order):
    AB = np.zeros((n_samples, n_inputs))
    BA = np.zeros((n_samples, n_inputs)) if calc_second_order else None
    step = 2 * n_inputs + 2 if calc_second_order else n_inputs + 2

    A = y[0:y.size:step]
    B = y[(step - 1):y.size:step]
    for j in range(D):
        AB[:, j] = y[(j + 1):y.size:step]
        if calc_second_order:
            BA[:, j] = y[(j + 1 + n_inputs):y.size:step]

    return A,B,AB,BA


def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A,B,AB], axis=0)
#     return np.mean(B * (AB - A), axis=0) # in the paper


def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A,AB], axis=0)


def sobol_indices(y, n_samples, n_inputs, calc_second_order=False):
    A,B,AB,BA = separate_output_values(y, n_samples, n_inputs, calc_second_order)
    first =first_order(A, AB, B)
    total = total_order(A, AB, B)
    sa_dict = {'first': first, 'total': total}
    return sa_dict


###################################
### 2. Correlation coefficients ###
###################################
def correlation_coefficients(dict_):

    X = dict_.get('X')
    y = dict_.get('y')

    from scipy.stats import spearmanr
    spearman, pval_spearman = spearmanr(X, y)
    spearman = spearman[:-1, -1]

    X_temp = np.hstack([X, y.reshape(X.shape[0], -1)]).T
    pearson = np.corrcoef(X_temp)
    pearson = pearson[:-1, -1]

    sa_dict = {
        'spearman': spearman,
        'pval_spearman': pval_spearman,
        'pearson': pearson,
    }
    return sa_dict