from gsa_framework.lca import LCAModel
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.methods.delta_moment import DeltaMoment
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import brightway2 as bw
import time
import numpy as np
from gsa_framework.plotting import histogram_Y1_Y2
from gsa_framework.utils import read_hdf5_array

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # LCA model
    bw.projects.set_current("GSA for oases")
    co = bw.Database("CH consumption 1.0")
    demand_act = [
        act for act in co if "Food and non-alcoholic beverages sector" in act["name"]
    ][0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")

    # Define some variables
    num_params = 10000
    num_influential = num_params // 100
    iterations_validation = 2000
    write_dir = path_base / "lca_model_food_{}".format(num_params)
    model = LCAModel(demand, method, write_dir, num_params=num_params)
    gsa_seed = 3403
    validation_seed = 7043
    fig_format = ["html", "pickle"]

    parameter_inds_convergence_plot = [1, 2, 3]  # TODO choose for convergence
    num_steps = 50

    # TODO Choose which GSA to perform
    flag_correlation = 0
    flag_sobol = 1
    flag_xgboost = 0
    flag_delta = 0

    if flag_correlation:
        iterations = 4 * num_params
        gsa = CorrelationCoefficients(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
        )
        # S_dict = gsa.perform_gsa()
        S_dict = gsa.generate_gsa_indices()
        pearson = S_dict["pearson"]
        spearman = S_dict["spearman"]
        # gsa.plot_sa_results(S_dict, fig_format=fig_format)

        # t0 = time.time()
        # val = Validation(
        #     model=model,
        #     iterations=iterations_validation,
        #     seed=validation_seed,
        #     default_x_rescaled=None,
        #     write_dir=write_dir,
        # )
        # tag = "SpearmanIndex"
        # influential_Y = val.get_influential_Y_from_gsa(
        #     spearman, num_influential, tag=tag
        # )
        # t1 = time.time()
        # print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        # val.plot_histogram_Y_all_Y_inf(
        #     influential_Y, num_influential, tag=tag, fig_format=fig_format
        # )

        num_convergence_plot = 10
        parameter_inds_convergence_plot = np.hstack(
            [
                np.argsort(spearman)[::-1][:num_convergence_plot],
                np.argsort(spearman)[::-1][-num_convergence_plot:],
            ]
        )

        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds_convergence_plot,
            fig_format=fig_format,
        )

    if flag_sobol:
        iterations = 40 * num_params
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.generate_gsa_indices()  # generate_gsa_indices
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(
            S_dict,
            fig_format=fig_format,
        )
        # t0 = time.time()
        # val = Validation(
        #     model=model,
        #     iterations=iterations_validation,
        #     seed=validation_seed,
        #     default_x_rescaled=None,
        #     write_dir=write_dir,
        # )
        # tag = "SaltelliTotalIndex"
        # influential_Y = val.get_influential_Y_from_gsa(total, num_influential, tag=tag)
        # t1 = time.time()
        # print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        # val.plot_histogram_Y_all_Y_inf(
        #     influential_Y, num_influential, tag=tag, fig_format=fig_format
        # )

        parameter_inds_convergence_plot = np.argsort(total)[::-1][:num_influential]
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds_convergence_plot, fig_format=fig_format
        )

    if flag_xgboost:
        num_boost_round = 400
        tuning_parameters = {
            "max_depth": 6,
            "eta": 0.1,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "refresh_leaf": True,
            "subsample": 0.6,
            "min_child_weight": 0.5,
        }
        iterations = 2 * num_params
        gsa = GradientBoosting(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
            tuning_parameters=tuning_parameters,
            num_boost_round=num_boost_round,
            xgb_model=None,
        )
        S_dict, r2, ev = gsa.perform_gsa(flag_save_S_dict=True, return_stats=True)
        #     S_dict = gsa.perform_gsa(flag_save_S_dict=True, return_stats=True)
        print(r2, ev)
        fscores = S_dict["fscores"]
        # gsa.plot_sa_results(
        #     S_dict,
        #     fig_format=fig_format,
        # )
        #
        # t0 = time.time()
        # val = Validation(
        #     model=model,
        #     iterations=iterations_validation,
        #     seed=validation_seed,
        #     default_x_rescaled=None,
        #     write_dir=write_dir,
        # )
        # tag = "FscoresIndex"
        # influential_Y = val.get_influential_Y_from_gsa(fscores, num_influential, tag=tag)
        # t1 = time.time()
        # print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        # val.plot_histogram_Y_all_Y_inf(
        #     influential_Y, num_influential, tag=tag, fig_format=fig_format
        # )

        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds_convergence_plot,
            fig_format=fig_format,
        )

    if flag_delta:
        iterations = 2 * num_params
        num_resamples = 1
        gsa = DeltaMoment(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            num_resamples=num_resamples,
            seed=gsa_seed,
        )
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps,
        )
        parameter_inds = [0, 1, 2]
        conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)
