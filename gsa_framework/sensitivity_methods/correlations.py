from ..utils import get_z_alpha_2, read_hdf5_array
import numpy as np
from scipy.stats import kendalltau, spearmanr
import h5py
import multiprocessing

n0_DEFAULT = 10
OPTIMAL_CHUNK_SIZE_PEARSON = (
    500  # somewhat optimal chunk size for numpy.corrcoef that computes Pearson coeff.
)
get_chunk_size_pearson = lambda num_params: min(OPTIMAL_CHUNK_SIZE_PEARSON, num_params)
OPTIMAL_CHUNK_SIZE_SPEARMAN = 100  # somewhat optimal chunk size for scipy.stats.spearmanr that computes Spearman coeff.
get_chunk_size_spearman = lambda num_params: min(
    OPTIMAL_CHUNK_SIZE_SPEARMAN, num_params
)


def kendalltau_mat(X, y):
    """Compute Kendall tau-a correlation coefficients for all parameters."""
    num_params = X.shape[1]
    kendall, pval_kendall = np.zeros(num_params), np.zeros(num_params)
    kendall[:], pval_kendall[:] = np.nan, np.nan
    for i in range(num_params):
        kendall_tuple = kendalltau(X[:, i], y)
        kendall[i] = kendall_tuple[0]
        pval_kendall[i] = kendall_tuple[1]
    return kendall, pval_kendall


def pearson_one_chunk(X, y):
    """Compute Pearson correlation coefficient between all columns of X and y, set nan coefficients to 0."""
    X_temp = np.hstack([X, y.reshape(X.shape[0], -1)]).T
    pearson = np.corrcoef(X_temp)
    pearson = pearson[:-1, -1]
    pearson[np.isnan(pearson)] = 0
    return pearson


def spearman_one_chunk(X, y):
    """Compute Spearman correlation coefficient between all columns of X and y, set nan coefficients to 0."""
    spearman = np.zeros(shape=X.shape[1])
    var_X = np.var(X, axis=0)
    var_0_where = np.where(var_X == 0)[0]
    var_non0_where = np.setdiff1d(np.arange(X.shape[1]), var_0_where)
    X_non0_var = X[:, var_non0_where]
    # Spearman
    spearman_non0, _ = spearmanr(X_non0_var, y)
    spearman_non0 = spearman_non0[:-1, -1]
    spearman[var_non0_where] = spearman_non0
    return spearman


def corrcoef_many_chunks(X, y, option):
    """Compute correlation coefficient given by ``option`` between all columns of X and y efficiently.

    This function computes correlations sequentially, by partitioning X into optimal number of columns
    that was determined heuristically for ``option`` chosen as ``pearson`` or ``spearman``.

    """

    if option == "pearson":
        corrcoef_func = pearson_one_chunk
        get_chunk_size = get_chunk_size_pearson
    elif option == "spearman":
        corrcoef_func = spearman_one_chunk
        get_chunk_size = get_chunk_size_spearman
    num_params = X.shape[1]
    chunk_size = get_chunk_size(num_params)
    num_chunks = int(np.ceil(num_params / chunk_size))
    corrcoef = np.array([])
    for i in range(num_chunks):
        X_partial = X[:, i * chunk_size : (i + 1) * chunk_size]
        corrcoef = np.hstack([corrcoef, corrcoef_func(X_partial, y)])
    return corrcoef


def corrcoef_parallel(y, filepath_X, cpus, option, selected_iterations=None):
    """Compute correlation coefficient efficiently in parallel, using multiprocessing with one job per worker.

    ``option`` can be ``pearson`` or ``spearman``.

    """

    if option == "pearson":
        get_chunk_size = get_chunk_size_pearson
    elif option == "spearman":
        get_chunk_size = get_chunk_size_spearman
    with h5py.File(filepath_X, "r") as f:
        X = np.array(f["dataset"][:1, :])
        num_params = X.shape[1]
    del X
    chunk_size = get_chunk_size(num_params)
    num_jobs = int(np.ceil(np.ceil(num_params / chunk_size) / cpus))
    chunks = list(range(0, num_params + num_jobs * chunk_size, num_jobs * chunk_size))
    cpus_needed = len(chunks) - 1
    results_all = np.array([])
    if selected_iterations is None:
        selected_iterations = np.arange(len(y))
    with h5py.File(filepath_X, "r") as f:
        X = np.array(f["dataset"][selected_iterations, :])
        with multiprocessing.Pool(processes=cpus_needed) as pool:
            results = pool.starmap(
                corrcoef_many_chunks,
                [
                    (
                        X[selected_iterations, chunks[i] : chunks[i + 1]],
                        y[selected_iterations],
                        option,
                    )
                    for i in range(cpus_needed)
                ],
            )
        results_array = np.array([])
        for res in results:
            results_array = np.hstack([results_array, res])
        results_all = np.hstack([results_all, results_array])
    return results_all


def corrcoef_parallel_stability_spearman(y, X, cpus=None):
    """Compute correlation coefficient efficiently in parallel, using multiprocessing with one job per worker.

    ``option`` can be ``pearson`` or ``spearman``.

    """

    get_chunk_size = get_chunk_size_spearman
    num_params = X.shape[1]
    chunk_size = get_chunk_size(num_params)
    cpus = min(
        # There has to be a way to make this more elegant, -> S: Set default cpus to inf?
        cpus or multiprocessing.cpu_count(),
        multiprocessing.cpu_count(),
    )
    num_jobs = int(np.ceil(np.ceil(num_params / chunk_size) / cpus))
    chunks = list(range(0, num_params + num_jobs * chunk_size, num_jobs * chunk_size))
    cpus_needed = len(chunks) - 1
    with multiprocessing.Pool(processes=cpus_needed) as pool:
        results = pool.starmap(
            corrcoef_many_chunks,
            [
                (
                    X[:, chunks[i] : chunks[i + 1]],
                    y,
                    "spearman",
                )
                for i in range(cpus_needed)
            ],
        )
    results_array = np.array([])
    for res in results:
        results_array = np.hstack([results_array, res])

    S_dict = {"spearman": results_array}
    return S_dict


def correlation_coefficients(
    filepath_Y, filepath_X_rescaled, cpus=None, selected_iterations=None
):
    """Compute estimates of Pearson and Spearman correlation coefficients between vector ``Y`` and all columns of ``X``.

    Parameters
    ----------
    filepath_Y : Path or str
        Filepath to model outputs saved in `.hdf5` format.
    filepath_X_rescaled : Path or str
        Filepath to rescaled model inputs sampling in `.hdf5` format.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains Pearson and Spearman correlation coefficients.

    """

    y = read_hdf5_array(filepath_Y)
    y = y.flatten()
    cpus = min(
        # There has to be a way to make this more elegant, -> S: Set default cpus to inf?
        cpus or multiprocessing.cpu_count(),
        multiprocessing.cpu_count(),
    )
    sa_dict = {
        "pearson": corrcoef_parallel(
            y, filepath_X_rescaled, cpus, "pearson", selected_iterations
        ),
        "spearman": corrcoef_parallel(
            y, filepath_X_rescaled, cpus, "spearman", selected_iterations
        ),
    }
    return sa_dict


def get_corrcoef_num_iterations(theta=None, interval_width=0.01, confidence_level=0.95):
    """Computes number of iterations for confident estimation of correlation coefficient with true value equal to ``theta``.

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
    iterations_dict : dict
        Dictionary with number of iterations for Pearson, Kendall and Spearman coefficients.

    References
    ----------
    Paper:
        :cite:ts:`bonett2000sample`
    Remark for testing:
        ``num_iterations`` should agree with the values from Table 1 of the paper. Part of the table is tested in tests.
        Sometimes there is a difference of +-1 iteration. I think this is due to minor numerical imprecision.

    """

    z_alpha_2 = get_z_alpha_2(confidence_level)

    corrcoeff_constants = {
        "pearson": {
            "b": 3,
            "c": 1,
            "theta": theta or 0.95,
        },  # "hardest" correlation value to estimate
        "spearman": {"b": 3, "theta": theta or 0.8},
        "kendall": {"b": 4, "c": (0.437) ** 0.5, "theta": theta or 0.8},
    }
    corrcoeff_constants["spearman"]["c"] = (
        1 + corrcoeff_constants["spearman"]["theta"] ** 2 / 2
    ) ** 0.5

    # It seems like these could be proper functions instead of anonymous ones.

    compute_n0 = lambda b, c, theta: np.round(
        4 * c ** 2 * (1 - theta ** 2) ** 2 * (z_alpha_2 / interval_width) ** 2 + b
    )
    compute_L1 = (
        lambda b, c, theta, n: 0.5 * (np.log(1 + theta) - np.log(1 - theta))
        - c * z_alpha_2 / (n - b) ** 0.5
    )
    compute_L2 = (
        lambda b, c, theta, n: 0.5 * (np.log(1 + theta) - np.log(1 - theta))
        + c * z_alpha_2 / (n - b) ** 0.5
    )
    compute_limit = lambda L: (np.exp(2 * L) - 1) / (np.exp(2 * L) + 1)
    compute_n = lambda b, n0, w0: np.round((n0 - b) * (w0 / interval_width) ** 2 + b)

    iterations_dict = {}
    for key, val in corrcoeff_constants.items():
        b, c, theta = val["b"], val["c"], val["theta"]
        # First stage approximation
        n0 = max(compute_n0(b, c, theta), n0_DEFAULT)
        L1 = compute_L1(b, c, theta, n0)
        L2 = compute_L2(b, c, theta, n0)
        lower_limit = compute_limit(L1)
        upper_limit = compute_limit(L2)
        assert upper_limit > lower_limit
        w0 = upper_limit - lower_limit
        # Second stage approximation
        iterations_dict[key] = int(max(compute_n(b, n0, w0), n0_DEFAULT))
    return iterations_dict


def get_corrcoef_interval_width(theta=None, iterations=100, confidence_level=0.95):
    """
    Computes confidence interval width given number of iterations, "true" value of correlation coefficient ``theta``
    and ``confidence_level``.

    Parameters
    ----------
    theta : float
        "True" correlation coefficient value that the estimator should approach. Can be Pearson, Kendall or Spearman.
    iterations : int
        Number of iterations.
    confidence_level : float
        Desired confidence level.

    Returns
    -------
    interval_width_dict : dict
        Dictionary with analytical confidence interval width for Pearson, Kendall and Spearman coefficients.

    References
    ----------
    Paper:
        :cite:ts:`bonett2000sample`

    """

    z_alpha_2 = get_z_alpha_2(confidence_level)

    corrcoeff_constants = {
        "pearson": {
            "b": 3,
            "c": 1,
            "theta": theta or 0.95,
        },  # "hardest" correlation value to estimate
        "spearman": {"b": 3, "theta": theta or 0.8},
        "kendall": {"b": 4, "c": (0.437) ** 0.5, "theta": theta or 0.8},
    }
    corrcoeff_constants["spearman"]["c"] = (
        1 + corrcoeff_constants["spearman"]["theta"] ** 2 / 2
    ) ** 0.5

    compute_L1 = (
        lambda b, c, theta, n: 0.5 * (np.log(1 + theta) - np.log(1 - theta))
        - c * z_alpha_2 / (n - b) ** 0.5
    )
    compute_L2 = (
        lambda b, c, theta, n: 0.5 * (np.log(1 + theta) - np.log(1 - theta))
        + c * z_alpha_2 / (n - b) ** 0.5
    )
    compute_limit = lambda L: (np.exp(2 * L) - 1) / (np.exp(2 * L) + 1)

    interval_width_dict = {}
    for k, val in corrcoeff_constants.items():
        b, c, theta = val["b"], val["c"], val["theta"]
        L1 = compute_L1(b, c, theta, iterations)
        L2 = compute_L2(b, c, theta, iterations)
        w0 = compute_limit(L2) - compute_limit(L1)
        interval_width_dict[k] = w0
    return interval_width_dict
