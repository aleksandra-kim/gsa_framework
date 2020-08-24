import numpy as np
import json
import os
from copy import deepcopy


########################
### 1. Sobol indices ###
########################
def separate_output_values(y, num_params):
    iterations_per_param = y.shape[0] // (num_params+2)
    AB = np.zeros((iterations_per_param, num_params))
    step = num_params + 2
    A = y[0::step].reshape(iterations_per_param, -1)
    B = y[(step - 1)::step].reshape(iterations_per_param, -1)
    for j in range(num_params):
        AB[:, j] = y[(j + 1)::step]
    return A,B,AB

def first_order(A, AB, B):
    # First order estimator normalized by sample variance
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
#     return np.mean(B * (AB - A), axis=0) # in the paper

def total_order(A, AB, B):
    # Total order estimator normalized by sample variance
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)

def sobol_indices(dict_):
    '''
    Function that computes estimations of Sobol' first and total order indices
    ---------------
    Source:
        Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index
        Saltelli A., Annoni P., Azzini I., Campolongo F., Ratto M., Tarantola S., 2010
        https://doi.org/10.1016/j.cpc.2009.09.018
    Links:
        https://github.com/SALib/SALib/blob/master/src/SALib/analyze/sobol.py

    Parameters
    ----------
    dict_ : dict
        Dictionary that contains model outputs `y` obtained by running model on saltelli samples
        and number of parameters num_params

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices
    '''
    y = dict_.get('y')
    num_params = dict_.get('num_params')
    A,B,AB = separate_output_values(y, num_params)
    first = first_order(A, AB, B)
    total = total_order(A, AB, B)
    sa_dict = {'first': first, 'total': total}
    return sa_dict


###################################
### 2. Correlation coefficients ###
###################################
def kendalltau_mat(X, y):
    from scipy.stats import kendalltau
    num_params = X.shape[1]
    kendall, pval_kendall = np.zeros(num_params), np.zeros(num_params)
    kendall[:], pval_kendall[:] = np.nan, np.nan
    for i in range(num_params):
        kendall_tuple = kendalltau(X[:,i], y)
        kendall[i] = kendall_tuple[0]
        pval_kendall[i] = kendall_tuple[1]
    return kendall, pval_kendall

def correlation_coefficients(dict_):

    X = dict_.get('X')
    y = dict_.get('y')

    from scipy.stats import spearmanr
    spearman, pval_spearman = spearmanr(X, y)
    spearman = spearman[:-1, -1]
    kendall, pval_kendall = kendalltau_mat(X, y)

    X_temp = np.hstack([X, y.reshape(X.shape[0], -1)]).T
    pearson = np.corrcoef(X_temp)
    pearson = pearson[:-1, -1]

    sa_dict = {
        'pearson': pearson,
        'spearman': spearman,
        'pval_spearman': pval_spearman,
        'kendall': kendall,
        'pval_kendall': pval_kendall,
    }
    return sa_dict

Z_ALPHA_2 = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.98: 2.326,
    0.99: 2.576,
}
n0_DEFAULT = 10

def get_corrcoef_num_iterations(theta=None, interval_width=0.1, confidence_level=0.99):
    '''
    References
    ----------
    Sample size requirements for estimating Pearson, Kendall and Spearman correlations
    Bonett, Douglas G and Wright, Thomas A, 2000
    http://doi.org/10.1007/BF02294183
    `num_iterations` should agree with the values from Table 1. This was tested in the tests.
    Sometimes there's a difference of +-1 iteration. I think this is due to numerical problems,
    because this paper is very old.
    '''

    z_alpha_2 = Z_ALPHA_2[confidence_level]

    corrcoeff_constants = {
        'pearson':  {'b': 3, 'c': 1, 'theta': theta or 0.95}, # "hardest" correlation value to estimate
        'spearman': {'b': 3, 'theta': theta or 0.8},
        'kendall':  {'b': 4, 'c': (0.437)**0.5, 'theta': theta or 0.8},
    }
    corrcoeff_constants['spearman']['c'] = (1 + corrcoeff_constants['spearman']['theta']**2/2) ** 0.5

    compute_n0 = lambda b, c, theta: np.round(4 * c**2 * (1 - theta**2)**2 * (z_alpha_2 / interval_width)**2 + b)
    compute_L1 = lambda b, c, theta, n: 0.5 * (np.log(1+theta) - np.log(1-theta)) - c*z_alpha_2 / (n-b)**0.5
    compute_L2 = lambda b, c, theta, n: 0.5 * (np.log(1+theta) - np.log(1-theta)) + c*z_alpha_2 / (n-b)**0.5
    compute_limit = lambda L: (np.exp(2*L)-1) / (np.exp(2*L)+1)
    compute_n = lambda b, n0, w0: np.round((n0-b) * (w0/interval_width)**2 + b)

    for val in corrcoeff_constants.values():
        b,c,theta = val['b'], val['c'], val['theta']
        # First stage approximation
        val['n0'] = max( compute_n0(b,c,theta), n0_DEFAULT )
        val['L1'] = compute_L1(b,c,theta,val['n0'])
        val['L2'] = compute_L2(b,c,theta,val['n0'])
        val['lower_limit'] = compute_limit(val['L1'])
        val['upper_limit'] = compute_limit(val['L2'])
        assert val['upper_limit'] > val['lower_limit']
        val['w0'] = val['upper_limit']  - val['lower_limit']

        # Second stage approximation
        val['num_iterations'] = max( compute_n(b,val['n0'],val['w0']), n0_DEFAULT )

    return corrcoeff_constants


############################
### 3. Gradient boosting ###
############################
def xgboost_scores(dict_):
    # 1. Preparations
    X = dict_.get('X')
    y = dict_.get('y')
    num_params = dict_.get('num_params')
    iterations = dict_.get('iterations')
    train_test_ratio = dict_.get('train_test_ratio', 0.8)
    write_dir = dict_.get('write_dir')

    # 2. Read xgboost parameters
    filename = os.path.join(write_dir, 'xgboost_params.json')
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
    except:
        params = {}

    # 3. Prepare training and testing sets for  gradient boosting trees
    import xgboost as xgb
    n_split = int(train_test_ratio * iterations)
    X_train, X_test = X[:n_split, :], X[n_split:iterations, :]
    y_train, y_test = y[:n_split], y[n_split:iterations]
    dtrain = xgb.DMatrix(X_train, y_train)
    X_dtest = xgb.DMatrix(X_test)

    # 4. Train the model
    model = xgb.train(params, dtrain)

    # 5. make predictions and compute prediction score
    from sklearn.metrics import r2_score, explained_variance_score
    y_pred = model.predict(X_dtest)
    r2 = r2_score(y_test, y_pred)
    exp_v_score = explained_variance_score(y_test, y_pred)

    # 6. Save importance scores
    fscores_inf = model.get_fscore()
    fscores_dict = {int(key[1:]): val for key, val in fscores_inf.items()}
    fscores_all = np.array([fscores_dict.get(i,0) for i in range(num_params)])

    sa_dict = {
        'fscores': fscores_all,
        'r2': r2,
        'explained_variance': exp_v_score,
    }
    print('r2 = ' + str(r2))
    return sa_dict


#################################
### 4. Dissimilarity measures ###
#################################
def dissimilarity_measure(dict_):
    num_params = dict_.get('num_params')
    iterations = dict_.get('iterations')
    iterations_per_parameter = iterations // (num_params + 1)
    y = dict_.get('y')

    # Bin model output `y` to obtain normalized histogram counts (aka densities)
    # - choose number of bins, other options can be tried out: Sturges' formula, Rice rule, Doane's formula, etc
    num_bins = int(np.round(np.sqrt(iterations_per_parameter)))
    _, bins = np.histogram(y, bins=num_bins)

    # Bin model output `y_base`
    y_base = y[:iterations_per_parameter]
    y_base_count_normalized = get_hist_count(y_base, bins)

    from scipy.stats import entropy
    # Include all dissimilarity measures on output here
    diss_measures = {
        'means_difference': means_difference,
    }
    # Include all dissimilarity measures on output densities here
    diss_measures_on_densities = {
        'entropy': entropy,
        'my_entropy': my_entropy,
    }

    # Compute dissimilarity between two distributions
    array = np.zeros(num_params)
    array[:] = np.nan
    sa_dict = {diss_measure_str: deepcopy(array) for diss_measure_str in diss_measures.keys()}
    sa_dict.update({diss_measure_str: deepcopy(array) for diss_measure_str in diss_measures_on_densities.keys()})

    for i in range(num_params):
        y_per_input = y[(i+1)*iterations_per_parameter : (i+2)*iterations_per_parameter]
        # Bin model output `y_per_input`
        y_per_input_count_normalized = get_hist_count(y_per_input, bins)
        for diss_measure_str, diss_measure_fnc in diss_measures.items():
            sa_dict[diss_measure_str][i] = diss_measure_fnc(y_base, y_per_input)
        for diss_measure_density_str, diss_measure_density_fnc in diss_measures_on_densities.items():
            sa_dict[diss_measure_density_str][i] = diss_measure_density_fnc(y_per_input_count_normalized, y_base_count_normalized)

    return sa_dict

def get_hist_count(data, bins):
    '''given histogram `bins`, get frequency count for `data`'''
    nbins = bins.shape[0]
    count = np.zeros(nbins, dtype=np.int)
    for i in range(nbins-1):
        if i != nbins-2:
            count[i] = np.sum(np.logical_and(data >= bins[i], data < bins[i+1]) )
        else:
            count[i] = np.sum(np.logical_and(data >= bins[i], data <= bins[i + 1]))
    return count / data.shape[0]

def means_difference(p1,p2):
    '''Sobol first order'''
    return (np.mean(p1) - np.mean(p2))**2

def my_entropy(p,q):
    sum_ = 0
    for i in range(len(p)):
        if q[i] != 0 and p[i] != 0:
            sum_ += p[i] * np.log(p[i] / q[i])
    return sum_


### 4.1 Mutual information / Kullback–Leibler divergence
########################################################


#####################################
### 5. Moment independent measure ###
#####################################

##########################################
### 6. Sequential selection algorithms ###
##########################################

##########################
### 7. Shapley effects ###
##########################











