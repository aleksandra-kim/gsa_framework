import numpy as np
from copy import deepcopy
import stats_arrays as sa


def get_amounts_means(tech_params):
    lognormal_where = np.where(
        tech_params["uncertainty_type"] == sa.LognormalUncertainty.id
    )[0]
    lognormal = tech_params[lognormal_where]
    lognormal_means = np.exp(lognormal["loc"] + lognormal["scale"] ** 2 / 2)
    triangular_where = np.where(
        tech_params["uncertainty_type"] == sa.TriangularUncertainty.id
    )[0]
    triangular = tech_params[triangular_where]
    triangular_means = (
        triangular["loc"] + triangular["minimum"] + triangular["maximum"]
    ) / 3
    amounts = deepcopy(tech_params["amount"])
    amounts[lognormal_where] = lognormal_means
    amounts[triangular_where] = triangular_means
    uncertain_where = np.where(tech_params["uncertainty_type"] > 1)[0]
    return amounts[uncertain_where], uncertain_where


def get_static_score(new_amounts, uncertain_where, lca):
    all_amounts = deepcopy(lca.tech_params["amount"])
    all_amounts[uncertain_where] = new_amounts
    lca.rebuild_technosphere_matrix(all_amounts)
    lca.redo_lci()
    lca.redo_lcia()
    return lca.score


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
