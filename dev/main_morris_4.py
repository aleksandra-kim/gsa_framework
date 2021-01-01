from gsa_framework.test_functions import Morris4
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.methods.delta_moment import DeltaMoment
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import time
import numpy as np

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. Models
    num_params = 100
    if num_params < 500:
        num_influential = 10
    else:
        num_influential = num_params // 100
    iterations_validation = 2000
    num_influential_validation = 2 * num_influential
    write_dir = path_base / "morris_model4_{}_{}".format(num_params, num_influential)
    model = Morris4(num_params=num_params, num_influential=num_influential)
    gsa_seed = 3407
    validation_seed = 7043
    num_steps_convergence = 50

    fig_format = ["html", "pickle"]  # can have elements "pdf", "html", "pickle"

    # TODO Choose which GSA to perform
    flag_sobol = 0
    flag_correlation = 0
    flag_eFAST = 0
    flag_xgboost = 0
    flag_delta = 0

    parameter_inds = np.arange(num_params)

    if flag_sobol:
        iterations = max(min(2 * num_params, 10000), 40000)
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        # S_dict = gsa.generate_gsa_indices()
        S_dict = gsa.perform_gsa(flag_save_S_dict=False)
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
        # influential_Y = val.get_influential_Y_from_gsa(total, num_influential_validation, tag=tag)
        # t1 = time.time()
        # print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        # val.plot_histogram_Y_all_Y_inf(
        #     influential_Y, num_influential, tag=tag, fig_format=fig_format
        # )
        #
        # conv = Convergence(
        #     gsa.filepath_Y,
        #     gsa.num_params,
        #     gsa.generate_gsa_indices,
        #     gsa.gsa_label,
        #     write_dir,
        #     num_steps=num_steps_convergence,
        # )
        # conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)

    if flag_correlation:
        iterations = max(min(2 * num_params, 10000), 20000)  # * num_params
        gsa = CorrelationCoefficients(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
        )
        S_dict = gsa.perform_gsa()
        pearson = S_dict["pearson"]
        spearman = S_dict["spearman"]
        gsa.plot_sa_results(S_dict, fig_format=fig_format)

        t0 = time.time()
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "SpearmanIndex"
        influential_Y = val.get_influential_Y_from_gsa(
            spearman, num_influential_validation, tag=tag
        )
        t1 = time.time()
        print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        val.plot_histogram_Y_all_Y_inf(
            influential_Y, num_influential, tag=tag, fig_format=fig_format
        )

        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps_convergence,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds,
            fig_format=fig_format,
        )

    if flag_eFAST:
        iterations = 100 * num_params
        M = 4
        gsa = eFAST(
            M=M, iterations=iterations, model=model, write_dir=write_dir, seed=gsa_seed
        )
        S_dict = gsa.perform_gsa()
        # S_dict = gsa.generate_gsa_indices()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(
            S_dict,
            fig_format=fig_format,
        )

        t0 = time.time()
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "eFastTotalIndex"
        influential_Y = val.get_influential_Y_from_gsa(
            total, num_influential_validation, tag=tag
        )
        t1 = time.time()
        print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        val.plot_histogram_Y_all_Y_inf(
            influential_Y, num_influential, tag=tag, fig_format=fig_format
        )

        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps_convergence,
            M=M,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds,
            fig_format=fig_format,
        )

    if flag_xgboost:
        if num_params == 1000 or 100:
            num_boost_round = 300
            tuning_parameters = {
                "max_depth": 2,  # higher than 10 is definitely not good
                "eta": 0.25,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "refresh_leaf": True,
                "subsample": 0.35,
                "min_child_weight": 0.5,
            }
        elif num_params == 5000:
            num_boost_round = 300
            tuning_parameters = {
                "max_depth": 2,  # higher than 10 is definitely not good
                "eta": 0.25,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "refresh_leaf": True,
                "subsample": 0.65,
                "min_child_weight": 0.5,
            }
        elif num_params == 10000:
            num_boost_round = 300
            tuning_parameters = {
                "max_depth": 2,  # higher than 10 is definitely not good
                "eta": 0.25,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "refresh_leaf": True,
                "subsample": 0.65,
                "min_child_weight": 0.5,
            }
        iterations = max(min(2 * num_params, 10000), 40000)
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
        print(r2, ev)
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(
            S_dict,
            fig_format=fig_format,
        )

        t0 = time.time()
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "FscoresIndex"
        influential_Y = val.get_influential_Y_from_gsa(
            fscores, num_influential_validation, tag=tag
        )
        t1 = time.time()
        print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        val.plot_histogram_Y_all_Y_inf(
            influential_Y, num_influential, tag=tag, fig_format=fig_format
        )

        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps_convergence,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds,
            fig_format=fig_format,
        )

    if flag_delta:
        iterations = max(min(2 * num_params, 10000), 40000)  # 2 * num_params
        num_resamples = 1
        gsa = DeltaMoment(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            num_resamples=num_resamples,
            seed=gsa_seed,
        )
        S_dict = gsa.perform_gsa()
        # S_dict = gsa.generate_gsa_indices()
        delta = S_dict["delta"]
        gsa.plot_sa_results(
            S_dict,
            fig_format=fig_format,
        )
        t0 = time.time()
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "DeltaIndex"
        influential_Y = val.get_influential_Y_from_gsa(
            delta, num_influential_validation, tag=tag
        )
        t1 = time.time()
        print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        val.plot_histogram_Y_all_Y_inf(
            influential_Y, num_influential, tag=tag, fig_format=fig_format
        )
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=num_steps_convergence,
        )
        conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)
