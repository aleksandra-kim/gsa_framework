import numpy as np
import pickle
from copy import deepcopy
from pathlib import Path
from pypardiso import spsolve

from gsa_framework.utils import read_pickle, write_pickle

WHERE_BIO_REVERSE_DICT = 2


def bio_params_local_sa(lca, write_dir, const_factors=(0.1, 10)):
    """Local SA for biosphere parameters."""

    path_lsa_bio = Path(write_dir) / "LSA_scores_bio.pickle"

    if not path_lsa_bio.exists():
        tech_x_demand = spsolve(lca.technosphere_matrix, lca.demand_array)
        characterization_vector = sum(lca.characterization_matrix)

        path_include_inds_bio = Path(write_dir) / "include_inds_bio.pickle"
        inds_uncertain = np.where(lca.bio_params["uncertainty_type"] > 1)[0]

        if not path_include_inds_bio.exists():
            uncertain_bio_params_temp = lca.bio_params[inds_uncertain]
            # Exclude bio exchanges that are not selected with the demand vector
            exclude_cols = np.where(tech_x_demand == 0)[0]
            exclude_inds = np.array([])
            for col in exclude_cols:
                exclude_inds = np.hstack(
                    [exclude_inds, np.where(uncertain_bio_params_temp["col"] == col)[0]]
                )
            # Exclude bio exchanges that are not included in the given lcia method
            exclude_rows = np.where(characterization_vector.toarray()[0] == 0)[0]
            for row in exclude_rows:
                exclude_inds = np.hstack(
                    [exclude_inds, np.where(uncertain_bio_params_temp["row"] == row)[0]]
                )
            print(
                "Excluding {}/{} biosphere exchanges".format(
                    len(exclude_inds), len(uncertain_bio_params_temp)
                )
            )
            exclude_inds = np.sort(exclude_inds)
            include_inds_temp = np.setdiff1d(
                np.arange(len(uncertain_bio_params_temp)), exclude_inds
            )
            write_pickle(include_inds_temp, path_include_inds_bio)
        else:
            include_inds_temp = read_pickle(path_include_inds_bio)

        include_inds = inds_uncertain[include_inds_temp]
        uncertain_bio_params = lca.bio_params[include_inds]
        nbio = len(uncertain_bio_params)
        rows = uncertain_bio_params["row"]
        cols = uncertain_bio_params["col"]
        bio_reverse_dict = lca.reverse_dict()[2]
        tech_reverse_dict = lca.reverse_dict()[0]

        lsa_scores_bio = {}
        for i, param in enumerate(uncertain_bio_params):
            if i % 1000 == 0:
                print("{}/{}".format(i, nbio))
            row, col = rows[i], cols[i]
            input_ = bio_reverse_dict[row]
            output_ = tech_reverse_dict[col]
            scores = []
            for const_factor in const_factors:
                biosphere_matrix = deepcopy(lca.biosphere_matrix)
                biosphere_matrix[row, col] *= const_factor
                score = characterization_vector * (biosphere_matrix * tech_x_demand)
                scores.append(score[0])
            lsa_scores_bio[include_inds[i]] = {
                "input": input_,
                "output": output_,
                "scores": np.array(scores),
            }
        write_pickle(lsa_scores_bio, path_lsa_bio)
    else:
        lsa_scores_bio = read_pickle(path_lsa_bio)

    return lsa_scores_bio


def cf_params_local_sa(lca, write_dir, const_factors=(0.1, 10)):
    """Local SA for characterization factors."""

    path_lsa_cf = Path(write_dir) / "LSA_scores_cf.pickle"

    if not path_lsa_cf.exists():

        bio_x_tech_x_demand = lca.biosphere_matrix * spsolve(
            lca.technosphere_matrix, lca.demand_array
        )

        path_lsa_include_inds_cf = Path(write_dir) / "include_inds_cf.pickle"
        inds_uncertain = np.where(lca.cf_params["uncertainty_type"] > 1)[0]

        if not path_lsa_include_inds_cf.exists():
            uncertain_cf_params_temp = lca.cf_params[inds_uncertain]
            # Exclude characterization factors that are not affected by given demand
            exclude_flows = np.where(bio_x_tech_x_demand == 0)[0]
            exclude_inds = np.array([])
            for flow in exclude_flows:
                exclude_inds = np.hstack(
                    [exclude_inds, np.where(uncertain_cf_params_temp["row"] == flow)[0]]
                )
            include_inds_temp = np.setdiff1d(
                np.arange(len(uncertain_cf_params_temp)), exclude_inds
            )
            write_pickle(include_inds_temp, path_lsa_include_inds_cf)
        else:
            include_inds_temp = read_pickle(path_lsa_include_inds_cf)

        include_inds = inds_uncertain[include_inds_temp]
        uncertain_cf_params = lca.cf_params[include_inds]
        flows = uncertain_cf_params["row"]
        bio_reverse_dict = lca.reverse_dict()[WHERE_BIO_REVERSE_DICT]

        lsa_scores_cf = {}
        for i, param in enumerate(uncertain_cf_params):
            flow = flows[i]
            input_ = bio_reverse_dict[flow]
            scores = []
            for const_factor in const_factors:
                characterization_vector = sum(deepcopy(lca.characterization_matrix))
                characterization_vector[0, flow] *= const_factor
                score = characterization_vector * bio_x_tech_x_demand
                scores.append(score[0])
            lsa_scores_cf[include_inds[i]] = {
                "input": input_,
                "scores": np.array(scores),
            }

        write_pickle(lsa_scores_cf, path_lsa_cf)

    else:
        lsa_scores_cf = read_pickle(path_lsa_cf)

    return lsa_scores_cf
