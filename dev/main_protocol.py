from gsa_framework.models import LCAModel
from pathlib import Path
import time
import numpy as np
import bw2data as bd
import bw2calc as bc


def setup_lca_model_protocol(path_base, num_params=None, write_dir=None):
    # LCA model
    bd.projects.set_current("GSA for protocol")
    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    # num_params
    if num_params is None:
        lca = bc.LCA(demand, method)
        lca.lci()
        lca.lcia()
        print("LCA score is {}".format(lca.score))
        n_uncertain_tech = len(lca.tech_params[lca.tech_params["uncertainty_type"] > 1])
        n_uncertain_bio = len(lca.bio_params[lca.bio_params["uncertainty_type"] > 1])
        n_uncertain_cf = len(lca.cf_params[lca.cf_params["uncertainty_type"] > 1])
        num_params_stats = n_uncertain_tech + n_uncertain_bio + n_uncertain_cf
        print("Total number of uncertain exchanges is {}".format(num_params_stats))
        print(
            "   tech={}, bio={}, cf={}".format(
                n_uncertain_tech, n_uncertain_bio, n_uncertain_cf
            )
        )
    # Define some variables
    if write_dir is None:
        write_dir = path_base / "protocol_gsa"
    model = LCAModel(demand, method, write_dir, num_params=num_params)
    gsa_seed = 4000238
    return model, write_dir, gsa_seed


if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )

    # path_base = Path('/data/user/kim_a/protocol_gsa')
    num_params = None
    flag_generate_scores_dict = False
    model, write_dir, gsa_seed = setup_lca_model_protocol(
        path_base,
        num_params=num_params,
        write_dir=None,
    )

    iterations = 10
    if num_params is None:
        num_params = len(model)

    np.random.seed(gsa_seed)
    X = np.random.rand(iterations, num_params)
    X_rescaled = model.rescale(X)
    lca_scores = model(X_rescaled)

    print(model)
