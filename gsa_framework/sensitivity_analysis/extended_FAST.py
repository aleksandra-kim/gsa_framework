from ..sampling.get_samples import get_omega_eFAST
import numpy as np
from ..utils import read_hdf5_array


def eFAST_first_order(Y, M, omega):
    """Sobol first order index estimator."""
    N = Y.shape[0]
    f = np.fft.fft(Y)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    D1 = 2 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
    return D1 / V


def eFAST_total_order(Y, omega):
    """Sobol total order index estimator."""
    N = Y.shape[0]
    f = np.fft.fft(Y)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    Dt = 2 * sum(Sp[np.arange(int(omega / 2))])
    return 1 - Dt / V


def eFAST_indices(filepath_Y, iterations, num_params, M=4, selected_iterations=None):
    """Compute estimations of Sobol' first and total order indices.

    High values of the Sobol first order index signify important parameters, while low values of the  total indices
    point to non-important parameters. First order computes main effects only, total order takes into account
    interactions between parameters.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains model outputs ``y`` obtained by running model on Saltelli samples,
        number of Monte Carlo iterations ``iterations``, and number of parameters ``num_params``.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices.

    References
    ----------
    Paper:
        A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output.
        Saltelli A., Tarantola S., Chan K. P.-S.
        https://doi.org/10.1080/00401706.1999.10485594
    Link with the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/analyze/fast.py

    """

    y = read_hdf5_array(filepath_Y)
    y = y.flatten()
    if selected_iterations is not None:
        y = y[selected_iterations]
    iterations_per_param = iterations // num_params
    # Recreate the vector omega used in the sampling
    omega = get_omega_eFAST(num_params, iterations_per_param, M)
    # Calculate and Output the First and Total Order Values
    first = np.zeros(num_params)
    total = np.zeros(num_params)
    first[:], total[:] = np.nan, np.nan
    if selected_iterations is not None:
        iterations_per_param_current = len(y) // num_params
        assert iterations_per_param == len(y) / num_params
    else:
        iterations_per_param_current = iterations_per_param
    for i in range(num_params):
        l = np.arange(i * iterations_per_param, (i + 1) * iterations_per_param)[
            :iterations_per_param_current
        ]
        first[i] = eFAST_first_order(y[l], M, omega[0])
        total[i] = eFAST_total_order(y[l], omega[0])
    S_dict = {
        "First order": first,
        "Total order": total,
    }
    return S_dict
