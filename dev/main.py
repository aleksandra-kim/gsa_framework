from gsa_framework.test_functions import *
from gsa_framework.lca import LCAModel
from gsa_framework.methods.extended_FAST import eFAST
import numpy as np
from pathlib import Path
import brightway2 as bw

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')
    seed = 340

    # 1. Models
    num_params = 400
    num_influential = 40
    model_morris = Morris(num_params=num_params, num_influential=num_influential)
    model_sobolG = SobolGstar(num_params=num_params, num_influential=num_influential)
    # LCA model
    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    act = co.search("average consumption")[0]
    demand = {act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")
    write_dir_lca = path_base / "lca_model"
    model_lca = LCAModel(demand, method, write_dir_lca)
    models = [
        model_morris,
        model_sobolG,
        model_lca,
    ]

    model = model_morris
    write_dir = path_base / "morris_model"
    gsa = eFAST(M=4, iterations=500000, model=model, write_dir=write_dir, seed=seed)
    S_dict = gsa.perform_gsa()
    gsa.plot_sa_results(S_dict, S_dict_analytical=model.sensitivity_indices)
    # S_boolean = np.hstack(
    #     [
    #         np.ones(model.num_influential),
    #         np.zeros(model.num_params-model.num_influential)
    #     ]
    # )
    # gsa.plot_sa_results(S_dict, S_boolean=S_boolean)
