from ..utils import get_z_alpha_2, read_hdf5_array
import numpy as np
from scipy.stats import kendalltau, spearmanr

n0_DEFAULT = 10

def kendalltau_mat(X, y):
    """Compute Kendall tau-a correlation coefficients for all parameters."""
    num_params = X.shape[1]
    kendall, pval_kendall = np.zeros(num_params), np.zeros(num_params)
    kendall[:], pval_kendall[:] = np.nan, np.nan
    for i in range(num_params):
        kendall_tuple = kendalltau(X[:,i], y)
        kendall[i] = kendall_tuple[0]
        pval_kendall[i] = kendall_tuple[1]
    return kendall, pval_kendall


def correlation_coefficients(gsa_dict):
    """Compute estimations of different correlation coefficients, such as Pearson, Kendall and Spearman.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains parameter sampling matrix ``X`` and model outputs ``y``.

    Returns
    -------

    Dictionary that contains computed sensitivity indices.

    TODO should be X or X rescaled?

    """

    y = read_hdf5_array(gsa_dict['filename_y'])
    y = y.flatten()
    X = read_hdf5_array(gsa_dict['filename_X_rescaled'])

    spearman, pval_spearman = spearmanr(X, y)
    spearman = spearman[:-1, -1]
    kendall, pval_kendall = kendalltau_mat(X, y)

    X_temp = np.hstack([X, y.reshape(X.shape[0], -1)]).T
    pearson = np.corrcoef(X_temp)
    pearson = pearson[:-1, -1]

    return {
        'pearson': pearson,
        'spearman': spearman,
        # 'pval_spearman': pval_spearman,
        'kendall': kendall,
        # 'pval_kendall': pval_kendall,
    }

# +
# def correlation_coefficients_chunks(gsa_dict):
#     with h5py.File(filename, 'r') as f:
#         array = np.array(f['dataset'][:])
# -

def get_corrcoef_num_iterations(theta=None, interval_width=0.1, confidence_level=0.99):
    """Computes number of iterations for confident estimation of correlation coefficient  ``theta``.

    Parameters
    ----------
    theta : float
        "True" correlation coefficient value that the estimator should approach. Can be Pearson, Kendall or Spearman.
    interval_width : float
        Desired width of the confidence interval.
    confidence_level : float
        Desired confidence level.

    Returns
    -------
    corrcoeff_constants : dict
        Dictionary with all constants that were used for the calculation of the number of iterations.

    References
    ----------
    Paper:
        Sample size requirements for estimating Pearson, Kendall and Spearman correlations.
        Bonett, Douglas G and Wright, Thomas A, 2000
        http://doi.org/10.1007/BF02294183
    Remark for testing:
        ``num_iterations`` should agree with the values from Table 1 of the paper. Part of the table is tested in tests.
        Sometimes there is a difference of +-1 iteration. I think this is due to minor numerical imprecision.

    """

    z_alpha_2 = get_z_alpha_2(confidence_level)

    corrcoeff_constants = {
        'pearson':  {'b': 3, 'c': 1, 'theta': theta or 0.95}, # "hardest" correlation value to estimate
        'spearman': {'b': 3, 'theta': theta or 0.8},
        'kendall':  {'b': 4, 'c': (0.437)**0.5, 'theta': theta or 0.8},
    }
    corrcoeff_constants['spearman']['c'] = (1 + corrcoeff_constants['spearman']['theta']**2/2) ** 0.5

    # It seems like these could be proper functions instead of anonymous ones.

    compute_n0 = lambda b, c, theta: np.round(4 * c**2 * (1 - theta**2)**2 * (z_alpha_2 / interval_width)**2 + b)
    compute_L1 = lambda b, c, theta, n: 0.5 * (np.log(1+theta) - np.log(1-theta)) - c*z_alpha_2 / (n-b)**0.5
    compute_L2 = lambda b, c, theta, n: 0.5 * (np.log(1+theta) - np.log(1-theta)) + c*z_alpha_2 / (n-b)**0.5
    compute_limit = lambda L: (np.exp(2*L)-1) / (np.exp(2*L)+1)
    compute_n = lambda b, n0, w0: np.round((n0-b) * (w0/interval_width)**2 + b)

    for val in corrcoeff_constants.values():
        b,c,theta = val['b'], val['c'], val['theta']
        # First stage approximation

        # Might be OK, but feels weird to store intermediate values in result dict

        val['n0'] = max( compute_n0(b,c,theta), n0_DEFAULT )
        val['L1'] = compute_L1(b,c,theta,val['n0'])
        val['L2'] = compute_L2(b,c,theta,val['n0'])
        val['lower_limit'] = compute_limit(val['L1'])
        val['upper_limit'] = compute_limit(val['L2'])
        assert val['upper_limit'] > val['lower_limit']
        val['w0'] = val['upper_limit']  - val['lower_limit']
        # Second stage approximation
        val['num_iterations'] = int(max( compute_n(b,val['n0'],val['w0']), n0_DEFAULT ))
    return corrcoeff_constants
