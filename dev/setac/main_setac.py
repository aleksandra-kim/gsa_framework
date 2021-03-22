import brightway2 as bw
from pathlib import Path
from gsa_framework import LCAModel
from gsa_framework.convergence_robustness_validation.validation import Validation

# from gsa_framework.utils_setac_lca import *
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
    lca_model = LCAModel(demand, method, write_dir)
    # Define some variables
    # seed = 923458
    seed = 10447
    num_params = len(lca_model)
    iterations_validation = 2000
    bin_min, bin_max = 2300, 3300

    validation = Validation(
        lca_model,
        iterations=iterations_validation,
        seed=seed,
        default_x_rescaled=lca_model.default_uncertain_amounts,
        write_dir=write_dir,
    )
    diff_mean = {
        34191: -0.06663386251557313,
        60: -115.68080082127426,
        "12.narrow": -33.89648819029526,
        "36.narrow": -64.88346733066737,
        "60.narrow": -43.92363283667646,
    }

    # 1. Validation plot base_Y
    # validation.plot_histogram_base_Y(default_Y=lca_model.models.score, bin_min=bin_min, bin_max=bin_max, save_fig=True)
    # validation.plot_histogram_base_Y(default_Y=None, bin_min=bin_min, bin_max=bin_max, save_fig=True)

    # 2. Influential_Y after LSA_3 and regression
    # --> When running the following code, num_influential is not the same due to numerics
    # path_lsa_3 = write_dir / "LSA_scores" / "3"
    # scores_dict = get_LSA_3_with_base_score(path_lsa_3, lca_init)
    # _, all_where_tech = get_nonzero_params(scores_dict, var_threshold=1e-16)
    # num_influential_lsa_3 = all_where_tech.shape[0]
    # --> So I have to hardcode....
    # tag = 60
    # filepath_influential_Y = (
    #     write_dir / "arrays" / validation.create_influential_model_output_filepath(tag)
    # )
    # influential_Y = read_hdf5_array(filepath_influential_Y).flatten() - diff_mean[tag]
    #
    # validation.plot_histogram_base_Y_influential_Y(
    #     influential_Y, tag=tag, save_fig=True, bin_min=bin_min, bin_max=bin_max
    # )
    #
    tag = 60
    filepath_influential_Y = (
        write_dir / "arrays" / validation.create_influential_model_output_filepath(tag)
    )
    influential_Y = read_hdf5_array(filepath_influential_Y).flatten() - diff_mean[tag]

    # validation.plot_histogram_base_Y_influential_Y(
    #     influential_Y, tag=tag, save_fig=True, bin_min=bin_min, bin_max=bin_max
    # )
    validation.plot_correlation_base_Y_influential_Y(
        influential_Y, tag=tag, save_fig=True
    )

    # Narrow
    # iterations = 2000
    # num_params_narrow = 60
    # scaling_factor = 8
    # tag = "{}.div{}".format(num_params_narrow, scaling_factor)
    # filename_Y_narrow = "validation.narrow.Y.{}.{}.div{}.{}.hdf5".format(
    #     iterations, num_params_narrow, scaling_factor, seed
    # )
    # print(filename_Y_narrow)
    # filepath_Y_narrow = write_dir / "arrays" / filename_Y_narrow
    # narrow_Y = read_hdf5_array(filepath_Y_narrow).flatten() #- diff_mean[tag]
    #
    # fig = validation.plot_histogram_base_Y_narrow_Y(
    #     narrow_Y, tag=tag, save_fig=True, bin_min=bin_min, bin_max=bin_max
    # )

    # TODO in the LCAModel: change uncertain_tech_params definition back to the one with threshold
    # TODO in the validation, uncomment last line in the init, regarding influential_y
    # TODO uncomment GSA indices
