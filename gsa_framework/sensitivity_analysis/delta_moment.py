import numpy as np
from ..utils import read_hdf5_array
from scipy.stats import gaussian_kde, rankdata


def delta_moment(
    filepath_Y, filepath_X, iterations, num_params, seed=None, num_resamples=10
):
    if not seed is None:
        np.random.seed(seed)

    Y = read_hdf5_array(filepath_Y).flatten()[:iterations]
    X = read_hdf5_array(filepath_X)[:iterations, :]

    # equal frequency partition
    exp = 2 / (7 + np.tanh((1500 - iterations) / 500))
    M = int(np.round(min(int(np.ceil(iterations ** exp)), 48)))
    m = np.linspace(0, iterations, M + 1)
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

    S_dict = {"delta": [np.nan] * num_params}

    try:
        for i in range(num_params):
            X_i = X[:, i]
            S_dict["delta"][i] = bias_reduced_delta(Y, Ygrid, X_i, m, num_resamples)
    except np.linalg.LinAlgError:
        msg = "Singular matrix detected\n"
        msg += "This may be due to the sample size ({}) being too small\n".format(
            Y.size
        )
        msg += "If this is not the case, check Y values or raise an issue with the\n"
        msg += "SALib team"
        raise np.linalg.LinAlgError(msg)
    return S_dict


def bias_reduced_delta(Y, Ygrid, X, m, num_resamples=10):
    """Plischke et al. 2013 bias reduction technique (eqn 30)"""
    d = np.zeros(num_resamples)
    d_hat = calc_delta(Y, Ygrid, X, m)

    N = len(Y)
    r = np.random.randint(N, size=(num_resamples, N))
    for i in range(num_resamples):
        r_i = r[i, :]
        d[i] = calc_delta(Y[r_i], Ygrid, X[r_i], m)

    d = 2 * d_hat - d
    return d.mean()


def calc_delta(Y, Ygrid, X, m):
    """Plischke et al. (2013) delta index estimator (eqn 26) for d_hat."""
    N = len(Y)
    fy = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    abs_fy = np.abs(fy)
    xr = rankdata(X, method="ordinal")

    d_hat = 0
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)

        Y_ix = Y[ix]
        if not np.all(np.equal(Y_ix, Y_ix[0])):
            fyc = gaussian_kde(Y_ix, bw_method="silverman")(Ygrid)
            fy_ = np.abs(fy - fyc)
        else:
            fy_ = abs_fy

        d_hat += (nm / (2 * N)) * np.trapz(fy_, Ygrid)

    return d_hat
