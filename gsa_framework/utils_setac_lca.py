import numpy as np
from copy import deepcopy
import stats_arrays as sa
from gsa_framework.validation import COLORS_DICT
import plotly.graph_objects as go
import os, pickle


def get_amounts_shift(tech_params, shift_median=True):
    # 1. Lognormal
    lognormal_where = np.where(
        tech_params["uncertainty_type"] == sa.LognormalUncertainty.id
    )[0]
    lognormal = tech_params[lognormal_where]
    m = lognormal["loc"]
    s = lognormal["scale"]
    lognormal_mean = np.exp(m + s ** 2 / 2)
    lognormal_median = np.exp(m)

    # 2. Normal
    normal_where = np.where(tech_params["uncertainty_type"] == sa.NormalUncertainty.id)[
        0
    ]
    normal = tech_params[normal_where]
    m = normal["loc"]
    normal_mean = m
    normal_median = normal_mean

    # 2. Normal
    uniform_where = np.where(
        tech_params["uncertainty_type"] == sa.UniformUncertainty.id
    )[0]
    uniform = tech_params[uniform_where]
    a = uniform["minimum"]
    b = uniform["maximum"]
    uniform_mean = (a + b) / 2
    uniform_median = uniform_mean

    # 4. Triangular
    triangular_where = np.where(
        tech_params["uncertainty_type"] == sa.TriangularUncertainty.id
    )[0]
    triangular = tech_params[triangular_where]
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
    triangular_mean = triangular_mean
    triangular_median = triangular_median

    amounts = deepcopy(tech_params["amount"])
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
    amounts = np.sign(tech_params["amount"]) * amounts
    assert np.all(np.sign(tech_params["amount"]) == np.sign(amounts))
    return amounts


def get_score_shift(new_amounts, uncertain_where, lca):
    lca_new = deepcopy(lca)
    default_amounts = deepcopy(lca_new.tech_params["amount"])
    default_amounts[uncertain_where] = new_amounts
    lca_new.rebuild_technosphere_matrix(default_amounts)
    lca_new.redo_lci()
    lca_new.redo_lcia()
    return lca_new.score


# From GSA ecoinvent
def get_lsa_scores_pickle(path):
    files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and "all" not in f
        and "DS" not in f
        and "meta" not in f
    ]
    starts = [int(f.split("_")[2]) for f in files]
    ind_sort = np.argsort(starts)

    files_sorted = [files[i] for i in ind_sort]

    scores = {}
    for file in files_sorted:
        filepath = os.path.join(path, file)
        with open(filepath, "rb") as f:
            temp = pickle.load(f)
        temp_int = {int(k): v["scores"] for k, v in temp.items()}
        scores.update(temp_int)

    return scores


def get_LSA_3_with_base_score(path, lca):
    """
    add base scores between the computed LSA scores
    """
    scores_dict = get_lsa_scores_pickle(path)
    vals_2, vals_3, keys_2, keys_3 = [], [], [], []
    keys = np.array(list(scores_dict.keys()))

    for k, v in scores_dict.items():
        if v.shape[0] == 2:
            vals_2.append(list(v))
            keys_2.append(k)
        elif v.shape[0] == 3:
            vals_3.append(list(v))
            keys_3.append(k)

    vals_2 = np.array(vals_2)
    keys_2 = np.array(keys_2)
    vals_2 = np.vstack(
        [
            vals_2[:, 0],
            np.tile(lca.score, [1, vals_2.shape[0]]),
            vals_2[:, 1],
        ]
    ).T

    vals_3 = np.array(vals_3)
    keys_3 = np.array(keys_3)

    complete_scores_dict = {keys_2[i]: vals_2[i] for i in range(len(keys_2))}
    complete_scores_dict.update({keys_3[i]: vals_3[i] for i in range(len(keys_3))})

    return complete_scores_dict


def get_nonzero_params(scores_dict, var_threshold=0.0001):
    keys = np.array(list(scores_dict.keys()))
    vals = np.array(list(scores_dict.values()))

    # Variance of LSA scores for each input / parameter
    var = np.var(vals, axis=1)
    where = np.where(var > var_threshold)[0]

    params_yes = keys[where]
    params_no = np.setdiff1d(keys, params_yes)
    params_yes.sort(), params_no.sort()

    return params_no, params_yes


def get_where_tech(path, lca):
    """
    where tech that still need to be computed
    """
    scores = get_lsa_scores_pickle(path)
    computed_where_tech = np.array(list(scores.keys()))
    all_where_tech = np.where(lca.tech_params["uncertainty_type"] != 0)[0]

    where_tech = np.setdiff1d(all_where_tech, computed_where_tech)
    return where_tech


def get_where_tech_nonzero(path_curr, path_prev, lca, th=1e-24):
    """
    where tech that still need to be computed
    """
    scores = get_lsa_scores_pickle(path_curr)
    computed_where_tech = np.array(list(scores.keys()))
    scores_dict = get_LSA_3_with_base_score(path_prev, lca)
    _, all_where_tech = get_nonzero_params(scores_dict, var_threshold=th)

    where_tech = np.setdiff1d(all_where_tech, computed_where_tech)
    return where_tech


def get_xgboost_params(path_model_dir, params_yes_0):
    path_model = path_model_dir / "model"
    path_params_yes_where = path_model_dir / "params_yes_where.pickle"

    with open(path_model, "rb") as f:
        model = pickle.load(f)
    with open(path_params_yes_where, "rb") as f:
        params_yes_where = pickle.load(f)

    fscore = model.get_score()
    fscore_max = max(fscore.values())
    for feature, fscore_val in fscore.items():
        fscore[feature] = fscore_val / fscore_max
    fscore_sorted = {
        k: v for k, v in sorted(fscore.items(), key=lambda item: item[1], reverse=True)
    }

    features_inf = np.array([int(i.replace("f", "")) for i in fscore_sorted.keys()])
    importance_inf = np.array(list(fscore_sorted.values()))
    n_inf = len(fscore)
    params_yes_where_inf = params_yes_where[features_inf]

    params_yes_xgboost = params_yes_0[params_yes_where_inf]
    importance_dict = {params_yes_xgboost[i]: importance_inf[i] for i in range(n_inf)}

    return model, params_yes_xgboost, importance_dict
