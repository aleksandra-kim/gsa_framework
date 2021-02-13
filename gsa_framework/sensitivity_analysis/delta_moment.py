import numpy as np
from ..utils import read_hdf5_array, get_z_alpha_2
from scipy.stats import gaussian_kde, rankdata
import multiprocessing
import h5py

# def delta_moment(
#     filepath_Y, filepath_X, iterations, num_params, seed=None, num_resamples=10
# ):
#     if not seed is None:
#         np.random.seed(seed)
#
#     Y = read_hdf5_array(filepath_Y).flatten()[:iterations]
#     X = read_hdf5_array(filepath_X)[:iterations, :]
#
#     # equal frequency partition
#     exp = 2 / (7 + np.tanh((1500 - iterations) / 500))
#     M = int(np.round(min(int(np.ceil(iterations ** exp)), 48)))
#     m = np.linspace(0, iterations, M + 1)
#     Ygrid = np.linspace(np.min(Y), np.max(Y), 100)
#
#     S_dict = {"delta": [np.nan] * num_params}
#
#     try:
#         for i in range(num_params):
#             X_i = X[:, i]
#             S_dict["delta"][i] = bias_reduced_delta(Y, Ygrid, X_i, m, num_resamples)
#     except np.linalg.LinAlgError:
#         msg = "Singular matrix detected\n"
#         msg += "This may be due to the sample size ({}) being too small\n".format(
#             Y.size
#         )
#         msg += "If this is not the case, check Y values or raise an issue with the\n"
#         msg += "SALib team"
#         raise np.linalg.LinAlgError(msg)
#     return S_dict


def compute_delta_parameters(Y, iterations):
    exp = 2 / (7 + np.tanh((1500 - iterations) / 500))
    M = int(np.round(min(int(np.ceil(iterations ** exp)), 48)))
    m = np.linspace(0, iterations, M + 1)
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)
    return Ygrid, m


def bias_reduced_delta_many_chunks(Y, X, num_resamples, conf_level):
    iterations = len(Y)
    num_params_current = X.shape[1]
    Ygrid, m = compute_delta_parameters(Y, iterations)
    Xr = rankdata(X, method="ordinal", axis=0)
    fy_full = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    d_hat = np.array([])
    for i in range(num_params_current):
        X_i = X[:, i]
        Xr_i = Xr[:, i]
        d_hat = np.hstack(
            [
                d_hat,
                bias_reduced_delta(
                    Y, Ygrid, X_i, m, num_resamples, conf_level, fy=fy_full, xr=Xr_i
                ),
            ]
        )
    return d_hat


def delta_parallel(Y, filepath_X, num_resamples, conf_level, cpus):
    with h5py.File(filepath_X, "r") as f:
        X = np.array(f["dataset"][:1, :])
        num_params = X.shape[1]
    del X
    num_jobs = int(np.ceil(num_params / cpus))
    chunks = list(range(0, num_params + num_jobs, num_jobs))
    cpus_needed = len(chunks) - 1
    results_all_delta = np.array([])
    results_all_delta_conf = np.array([])
    with h5py.File(filepath_X, "r") as f:
        X = np.array(f["dataset"][:])
        with multiprocessing.Pool(processes=cpus_needed) as pool:
            results = pool.starmap(
                bias_reduced_delta_many_chunks,
                [
                    (Y, X[:, chunks[i] : chunks[i + 1]], num_resamples, conf_level)
                    for i in range(cpus_needed)
                ],
            )
        results_array_delta = np.array([])
        results_array_delta_conf = np.array([])
        for res in results:
            results_array_delta = np.hstack([results_array_delta, res[::2]])
            results_array_delta_conf = np.hstack([results_array_delta_conf, res[1::2]])
        results_all_delta = np.hstack([results_all_delta, results_array_delta])
        results_all_delta_conf = np.hstack(
            [results_all_delta_conf, results_array_delta_conf]
        )
    return results_all_delta, results_all_delta_conf


def delta_parallel_stability(Y, X, num_resamples, conf_level, cpus):
    num_params = X.shape[1]
    num_jobs = int(np.ceil(num_params / cpus))
    chunks = list(range(0, num_params + num_jobs, num_jobs))
    cpus_needed = len(chunks) - 1
    results_all_delta = np.array([])
    results_all_delta_conf = np.array([])
    with multiprocessing.Pool(processes=cpus_needed) as pool:
        results = pool.starmap(
            bias_reduced_delta_many_chunks,
            [
                (Y, X[:, chunks[i] : chunks[i + 1]], num_resamples, conf_level)
                for i in range(cpus_needed)
            ],
        )
    results_array_delta = np.array([])
    results_array_delta_conf = np.array([])
    for res in results:
        results_array_delta = np.hstack([results_array_delta, res[::2]])
        results_array_delta_conf = np.hstack([results_array_delta_conf, res[1::2]])
    results_all_delta = np.hstack([results_all_delta, results_array_delta])
    results_all_delta_conf = np.hstack(
        [results_all_delta_conf, results_array_delta_conf]
    )
    return results_all_delta, results_all_delta_conf


def delta_moment(
    filepath_Y,
    filepath_X_rescaled,
    num_resamples=1,
    conf_level=0.95,
    seed=None,
    cpus=None,
):
    """Compute estimations of different correlation coefficients, such as Pearson and Spearman.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains parameter sampling matrix ``X`` and model outputs ``y``.

    Returns
    -------

    Dictionary that contains computed sensitivity indices.

    """
    np.random.seed(seed)
    Y = read_hdf5_array(filepath_Y).flatten()
    cpus = min(
        # There has to be a way to make this more elegant, -> S: Set default cpus to inf?
        cpus or multiprocessing.cpu_count(),
        multiprocessing.cpu_count(),
    )
    results_delta, results_delta_conf = delta_parallel(
        Y, filepath_X_rescaled, num_resamples, conf_level, cpus
    )
    return {
        "delta": results_delta,
        "stat.delta_conf": results_delta_conf,
    }


def calc_delta(Y, Ygrid, X, m, fy=None, xr=None):
    """Plischke et al. (2013) delta index estimator (eqn 26) for d_hat."""
    N = len(Y)
    if fy is None:
        fy = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    if xr is None:
        xr = rankdata(X, method="ordinal")
    abs_fy = np.abs(fy)

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


def bias_reduced_delta(Y, Ygrid, X, m, num_resamples, conf_level, fy=None, xr=None):
    """Plischke et al. 2013 bias reduction technique (eqn 30)"""

    d_hat = calc_delta(Y, Ygrid, X, m, fy=fy, xr=xr)

    if num_resamples > 0:
        d = np.zeros(num_resamples)
        N = len(Y)
        r = np.random.randint(N, size=(num_resamples, N))
        for i in range(num_resamples):
            r_i = r[i, :]
            d[i] = calc_delta(Y[r_i], Ygrid, X[r_i], m)

        d = 2 * d_hat - d
        return d.mean(), get_z_alpha_2(conf_level) * d.std(ddof=1)
    else:
        return d_hat, np.nan


def delta_moment_parallel_stability(
    Y,
    X_rescaled,
    num_resamples=1,
    conf_level=0.95,
    seed=None,
    cpus=None,
):
    """Compute estimations of different correlation coefficients, such as Pearson and Spearman.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains parameter sampling matrix ``X`` and model outputs ``y``.

    Returns
    -------

    Dictionary that contains computed sensitivity indices.

    """
    np.random.seed(seed)
    cpus = min(
        # There has to be a way to make this more elegant, -> S: Set default cpus to inf?
        cpus or multiprocessing.cpu_count(),
        multiprocessing.cpu_count(),
    )
    results_delta, results_delta_conf = delta_parallel_stability(
        Y, X_rescaled, num_resamples, conf_level, cpus
    )
    return {
        "delta": results_delta,
        "stat.delta_conf": results_delta_conf,
    }


def delta_moment_stability(
    Y,
    X_rescaled,
    num_resamples=1,
    conf_level=0.95,
    seed=None,
):
    """Compute estimations of different correlation coefficients, such as Pearson and Spearman.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains parameter sampling matrix ``X`` and model outputs ``y``.

    Returns
    -------

    Dictionary that contains computed sensitivity indices.

    """
    np.random.seed(seed)
    results = bias_reduced_delta_many_chunks(
        Y,
        X_rescaled,
        num_resamples=num_resamples,
        conf_level=conf_level,
    )
    return {
        "delta": results[::2],
        "stat.delta_conf": results[1::2],
    }
