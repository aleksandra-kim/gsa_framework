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











