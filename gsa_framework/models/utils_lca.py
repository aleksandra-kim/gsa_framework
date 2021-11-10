import numpy as np
import stats_arrays as sa
from copy import deepcopy


def get_amounts_shift(uncertain_params_all_exchanges, shift_median=True):
    amounts_all_exchanges = {}
    for exchange_type, uncertain_params in uncertain_params_all_exchanges.items():
        # 1. Lognormal
        lognormal_where = np.where(
            uncertain_params["uncertainty_type"] == sa.LognormalUncertainty.id
        )[0]
        lognormal = uncertain_params[lognormal_where]
        m = lognormal["loc"]
        s = lognormal["scale"]
        lognormal_mean = np.exp(m + s ** 2 / 2)
        lognormal_median = np.exp(m)

        # 2. Normal
        normal_where = np.where(
            uncertain_params["uncertainty_type"] == sa.NormalUncertainty.id
        )[0]
        normal = uncertain_params[normal_where]
        m = normal["loc"]
        normal_mean = m
        normal_median = normal_mean

        # 2. Uniform
        uniform_where = np.where(
            uncertain_params["uncertainty_type"] == sa.UniformUncertainty.id
        )[0]
        uniform = uncertain_params[uniform_where]
        a = uniform["minimum"]
        b = uniform["maximum"]
        uniform_mean = (a + b) / 2
        uniform_median = uniform_mean

        # 4. Triangular
        triangular_where = np.where(
            uncertain_params["uncertainty_type"] == sa.TriangularUncertainty.id
        )[0]
        triangular = uncertain_params[triangular_where]
        c = triangular["loc"]
        a = triangular["minimum"]
        b = triangular["maximum"]
        triangular_mean = (a + b + c) / 3
        triangular_median = np.empty(triangular.shape[0])
        triangular_median[:] = np.nan
        case1 = np.where(c >= (a + b) / 2)[0]
        triangular_median[case1] = a[case1] + np.sqrt(
            (b[case1] - a[case1]) * (c[case1] - a[case1]) / 2
        )
        case2 = np.where(c < (a + b) / 2)[0]
        triangular_median[case2] = b[case2] - np.sqrt(
            (b[case2] - a[case2]) * (b[case2] - c[case2]) / 2
        )

        amounts = deepcopy(uncertain_params["amount"])
        if shift_median:
            amounts[lognormal_where] = lognormal_median
            amounts[normal_where] = normal_median
            amounts[uniform_where] = uniform_median
            amounts[triangular_where] = triangular_median
        else:
            amounts[lognormal_where] = lognormal_mean
            amounts[normal_where] = normal_mean
            amounts[uniform_where] = uniform_mean
            amounts[triangular_where] = triangular_mean
        amounts = np.sign(uncertain_params["amount"]) * np.abs(amounts)
        assert np.all(np.sign(uncertain_params["amount"]) == np.sign(amounts))

        amounts_all_exchanges[exchange_type] = amounts

    return amounts_all_exchanges


def get_lca_score_shift(new_amounts, uncertain_where, lca):
    lca_new = deepcopy(lca)
    exchange_types = list(uncertain_where.keys())
    for exchange_type in exchange_types:
        if exchange_type == "tech":
            default_amount = deepcopy(lca_new.tech_params["amount"])
            default_amount[uncertain_where[exchange_type]] = new_amounts[exchange_type]
            lca_new.rebuild_technosphere_matrix(default_amount)
        elif exchange_type == "bio":
            default_amount = deepcopy(lca_new.bio_params["amount"])
            default_amount[uncertain_where[exchange_type]] = new_amounts[exchange_type]
            lca_new.rebuild_biosphere_matrix(default_amount)
        elif exchange_type == "cf":
            default_amount = deepcopy(lca_new.cf_params["amount"])
            default_amount[uncertain_where[exchange_type]] = new_amounts[exchange_type]
            lca_new.rebuild_characterization_matrix(default_amount)
    lca_new.redo_lci()
    lca_new.redo_lcia()
    return lca_new.score
