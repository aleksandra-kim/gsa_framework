import brightway2 as bw
from pathlib import Path
from gsa_framework import LCAModel
from gsa_framework.validation import Validation
from gsa_framework.utils_setac_lca import *
from gsa_framework.utils import read_hdf5_array


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
    write_dir = path_base / "setac_gsa"
    # var_threshold = 100
    model = LCAModel(demand, method, write_dir)
    lca_init = deepcopy(model.lca)

    # Define some variables
    seed = 923458
    num_params = len(model)
    iterations_validation = 2000
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

    # 1. Base_Y scores histogram
    # base_Y = read_hdf5_array(validation.filepath_base_Y).flatten()
    # fig = validation.plot_base_Y(
    #     base_Y,
    #     bin_min=bin_min,
    #     bin_max=bin_max,
    #     save_fig=True,
    #     default_y=static_mean_score,
    # )

    # 2. Influential_Y after LSA_3
    # --> When running the following code, num_influential is not the same due to numerics
    # path_lsa_3 = write_dir / "LSA_scores" / "3"
    # scores_dict = get_LSA_3_with_base_score(path_lsa_3, lca_init)
    # _, all_where_tech = get_nonzero_params(scores_dict, var_threshold=1e-16)
    # num_influential_lsa_3 = all_where_tech.shape[0]
    # --> So I have to hardcode....
    num_influential_lsa_3 = 5474
    base_Y = read_hdf5_array(validation.filepath_base_Y).flatten()
    filepath_influential_Y = (
        write_dir
        / "arrays"
        / validation.create_influential_model_output_filepath(num_influential_lsa_3)
    )
    influential_Y = read_hdf5_array(filepath_influential_Y).flatten()
    validation.plot_base_influential_Y(
        base_y=validation.base_Y,
        influential_y=influential_Y,
        num_influential=num_influential_lsa_3,
        bin_min=bin_min,
        bin_max=bin_max,
        save_fig=True,
    )

    # 3. Validation with GSA indices
    num_influential_lsa_3 = 5474
    base_Y = read_hdf5_array(validation.filepath_base_Y).flatten()
    filepath_influential_Y = (
        write_dir
        / "arrays"
        / validation.create_influential_model_output_filepath(num_influential_lsa_3)
    )
    influential_Y = read_hdf5_array(filepath_influential_Y).flatten()
    validation.plot_correlation(
        base_Y,
        influential_Y,
        num_influential=num_influential_lsa_3,
        start=0,
        end=50,
        save_fig=True,
    )

    # TODO in the LCAModel: change uncertain_tech_params definition back to the one with threshold
    # TODO in the validation, uncomment last line in the init, regarding influential_y
    # TODO uncomment GSA indices
