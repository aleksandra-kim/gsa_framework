from copy import deepcopy
from scipy.stats import entropy
import numpy as np


def dissimilarity_measure(dict_):
    """Naive implementation of dissimilarity measures computation between distributions.

    The parameter sampling matrix ``X`` that generated model outputs ``y`` was constructed in such a way that
    first all parameters vary and ``y_base`` is obtained, then each parameter is varied independently while others are
    fixed, and the ``y_per_parameter`` is computed. Finally, various distances or dissimilarity measures between
    ``y_base`` and ``y_per_parameter`` are computed.

    Parameters
    ----------
    dict_ : dict
        Dictionary that contains model outputs ``y``, number of iterations ``iterations`` and number of parameters
        ``num_params``.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices.

    TODO add references

    """

    num_params, iterations, y = dict_["num_params"], dict["iterations"], dict["y"]
    iterations_per_parameter = iterations // (num_params + 1)

    # Bin model output ``y`` to obtain normalized histogram counts (aka densities).
    # Choose number of bins as follows:
    num_bins = int(
        np.round(np.sqrt(iterations_per_parameter))
    )  # other options: Sturges', Rice, Doane's formulas, etc
    _, bins = np.histogram(y, bins=num_bins)
    # Bin model output ``y_base``
    y_base = y[:iterations_per_parameter]
    y_base_count_normalized = get_hist_count(y_base, bins)

    # Include all dissimilarity measures on output here
    diss_measures = {
        "means_difference": means_difference_squared,
    }
    # Include all dissimilarity measures on output densities here
    diss_measures_on_densities = {
        "entropy": entropy,
        "entropy_without_zeros": entropy_without_zeros,
    }

    # Compute dissimilarity between two distributions
    array = np.zeros(num_params)
    array[:] = np.nan
    sa_dict = {
        diss_measure_str: deepcopy(array) for diss_measure_str in diss_measures.keys()
    }
    sa_dict.update(
        {
            diss_measure_str: deepcopy(array)
            for diss_measure_str in diss_measures_on_densities.keys()
        }
    )
    for i in range(num_params):
        y_per_parameter = y[
            (i + 1) * iterations_per_parameter : (i + 2) * iterations_per_parameter
        ]
        # Bin model output ``y_per_parameter``
        y_per_parameter_count_normalized = get_hist_count(y_per_parameter, bins)
        for diss_measure_str, diss_measure_fnc in diss_measures.items():
            sa_dict[diss_measure_str][i] = diss_measure_fnc(y_base, y_per_parameter)
        for (
            diss_measure_density_str,
            diss_measure_density_fnc,
        ) in diss_measures_on_densities.items():
            sa_dict[diss_measure_density_str][i] = diss_measure_density_fnc(
                y_per_parameter_count_normalized, y_base_count_normalized
            )
    return sa_dict


def get_hist_count(data, bins):
    """Given histogram ``bins``, get frequency count for ``data``."""
    nbins = bins.shape[0]
    count = np.zeros(nbins, dtype=np.int)
    for i in range(nbins - 1):
        if i != nbins - 2:
            count[i] = np.sum(np.logical_and(data >= bins[i], data < bins[i + 1]))
        else:
            count[i] = np.sum(np.logical_and(data >= bins[i], data <= bins[i + 1]))
    return count / data.shape[0]


def means_difference_squared(p, q):
    """Compute squared difference of means between the histograms ``p`` and ``q``."""
    return (np.mean(p) - np.mean(q)) ** 2


def entropy_without_zeros(p, q):
    """Compute entropy between histograms ``p`` and ``q``, which have same bins, by disregarding 0 values."""
    sum_ = 0
    for i in range(len(p)):
        if q[i] != 0 and p[i] != 0:
            sum_ += p[i] * np.log(p[i] / q[i])
    return sum_

    # I think this should be OK
    return sum(x[i] * np.log(x[i] / y[i] for x, y in zip(p, q) if x > 0 and y > 0))
