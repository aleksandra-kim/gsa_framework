import numpy as np
from ..utils import get_z_alpha_2, read_hdf5_array

# TODO confidence intervals


def separate_output_values(y, num_params):
    """Separate model output into values obtained from the sampling matrices A, B and AB."""
    iterations_per_param = y.shape[0] // (num_params + 2)
    AB = np.zeros((iterations_per_param, num_params))
    step = num_params + 2
    A = y[0::step].reshape(iterations_per_param, -1)
    B = y[(step - 1) :: step].reshape(iterations_per_param, -1)
    for j in range(num_params):
        AB[:, j] = y[(j + 1) :: step]
    return A, B, AB


def sobol_first_order(A, AB, B):
    """First order estimator normalized by sample variance."""
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
    # return np.mean(B * (AB - A), axis=0) # in the paper and in SALib


def sobol_total_order(A, AB, B):
    """Total order estimator normalized by sample variance."""
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)


def confidence_interval(std, N, confidence_level=0.95):
    """TODO change ``N`` to be consistent with the selected coefficients"""
    z_alpha_2 = get_z_alpha_2(confidence_level)
    return z_alpha_2 * std / np.sqrt(N)


def sobol_indices(filepath_Y, num_params, selected_iterations=None):
    """Compute estimations of Sobol' first and total order indices.

    High values of the Sobol first order index signify important parameters, while low values of the  total indices
    point to non-important parameters. First order computes main effects only, total order takes into account
    interactions between parameters.

    Parameters
    ----------
    gsa_dict : dict
        Dictionary that contains model outputs ``y`` obtained by running model on Saltelli samples,
        and number of parameters ``num_params``.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices.

    References
    ----------
    Paper:
        Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index
        Saltelli A., Annoni P., Azzini I., Campolongo F., Ratto M., Tarantola S., 2010
        https://doi.org/10.1016/j.cpc.2009.09.018
    Link with the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/analyze/sobol.py

    """

    y = read_hdf5_array(filepath_Y)
    y = y.flatten()
    if selected_iterations is not None:
        y = y[selected_iterations]
    A, B, AB = separate_output_values(y, num_params)
    first = sobol_first_order(A, AB, B)
    total = sobol_total_order(A, AB, B)
    sa_dict = {
        "First order": first,
        "Total order": total,
    }
    return sa_dict


def sobol_indices_stability(
    Y, num_params, iterations_for_convergence, num_bootstrap, stability_seeds=None
):
    S_dict = {}
    for i, iterations_current in enumerate(iterations_for_convergence):
        Ycurrent = Y[:iterations_current]
        iterations_per_param = Ycurrent.shape[0] // (num_params + 2)
        A, B, AB = separate_output_values(Ycurrent, num_params)
        all_first = np.zeros((0, num_params))
        all_total = np.zeros((0, num_params))
        for j in range(num_bootstrap):
            if stability_seeds is not None:
                stability_seed = stability_seeds[i, j]
                np.random.seed(stability_seed)
            choice = np.random.choice(
                np.arange(iterations_per_param), size=iterations_per_param, replace=True
            )
            Aj = A[choice]
            Bj = B[choice]
            ABj = AB[choice]
            first = sobol_first_order(Aj, ABj, Bj)
            total = sobol_total_order(Aj, ABj, Bj)
            all_first = np.vstack([all_first, first])
            all_total = np.vstack([all_total, total])
        S_dict[iterations_current] = {
            "first": all_first,
            "total": all_total,
        }
    return S_dict
