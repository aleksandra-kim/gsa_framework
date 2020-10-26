import brightway2 as bw
from pathlib import Path
from gsa_framework import LCAModel
from gsa_framework.validation import Validation
from gsa_framework.utils_setac_lca import *


if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/setac_gsa/')

    # LCA model
    bw.projects.set_current("GSA for setac")
    co = bw.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]][0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")
    write_dir = path_base / "setac_lca"
    # var_threshold = 100
    model = LCAModel(demand, method, write_dir)

    # Define some variables
    seed = 923458
    num_params = len(model)
    iterations_validation = 500
    bin_min, bin_max = 2300, 3300

    default_amounts, uncertain_tech_params_where = get_amounts_means(
        model.lca.tech_params
    )
    static_mean_score = get_static_score(
        default_amounts, uncertain_tech_params_where, model.lca
    )
    validation = Validation(
        model,
        iterations=iterations_validation,
        seed=seed,
        default_x=default_amounts,
        write_dir=write_dir,
    )

    # path_lsa_3 = path_base / "LSA_scores" / "3"
    # scores_dict = get_LSA_3_with_base_score(path_lsa_3, model.lca)
    # _, all_where_tech = get_nonzero_params(scores_dict, var_threshold=1e-16)

    influential_params_where = uncertain_tech_params_where[:10]
    influential_Y = validation.get_influential_Y_from_parameter_choice(
        influential_params_where
    )
    validation.plot_histogram(base_y=validation.base_Y, influential_y=influential_Y)

    # 1. Base_Y scores histogram
    # fig = validation.plot_base_Y(
    #     validation.base_Y,
    #     bin_min=2300,
    #     bin_max=3300,
    #     save_fig=True,
    #     default_y=static_mean_score
    # )

    # TODO in the LCAModel: change uncertain_tech_params definition back to the one with threshold
    # TODO in the validation, uncomment last line in the init, regarding influential_y
    # TODO uncomment GSA indices
