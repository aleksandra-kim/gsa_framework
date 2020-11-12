from gsa_framework.test_functions import SobolGstar
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import numpy as np
import time

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path("/data/user/kim_a/paper_gsa/gsa_framework_files")

    # 1. Models
    num_params = 5000
    # num_influential=10
    num_influential = num_params // 100
    iterations_validation = 2000
    write_dir = path_base / "sobol_Gstar_model_{}".format(num_params)
    gsa_seed = 3407
    sobol_g_star_seed = 345897
    validation_seed = 7043
    if num_influential == 10:
        a = np.array(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            + [9] * (num_params - num_influential)
        )
        alpha = np.ones(num_params)  # case 2 from Saltelli paper
    else:
        a = 9 * np.ones(num_params)
        a[:num_influential] = np.linspace(0, 4, num_influential, endpoint=True)
        alpha = np.ones(num_params)
    np.random.seed(sobol_g_star_seed)
    delta = np.random.rand(num_params)  # delta doesn't affect input importance
    np.random.seed(None)
    model = SobolGstar(
        num_params=num_params,
        num_influential=num_influential,
        a=a,
        alpha=alpha,
        delta=delta,
    )
    fig_format = ["html", "pickle"]  # can have elements "pdf", "html", "pickle"

    num_params_corr_plot = 20
    parameter_inds = list(range(num_params_corr_plot)) + list(
        range(num_influential, num_influential + num_params_corr_plot)
    )

    # TODO Choose which GSA to perform
    flag_sobol = 0
    flag_correlation = 0
    flag_eFAST = 0
    flag_xgboost = 1

    if flag_sobol:
        iterations = 100 * num_params
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.generate_gsa_indices()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(
            S_dict,
            S_dict_analytical=model.S_dict_analytical,
            fig_format=fig_format,
        )

        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        frac_inf_total, frac_non_inf_total = val.get_fraction_identified_correctly(
            total, model.influential_params
        )
        frac_inf_first, frac_non_inf_first = val.get_fraction_identified_correctly(
            first, model.influential_params
        )
        print(
            "Fraction of     INFLUENTIAL identified correctly -> total: {0:4.3f}, first: {1:4.3f}".format(
                frac_inf_total, frac_inf_first
            )
        )
        print(
            "Fraction of NON-INFLUENTIAL identified correctly -> total: {0:4.3f}, first: {1:4.3f}".format(
                frac_non_inf_total, frac_non_inf_first
            )
        )

        # conv = Convergence(
        #     gsa.filepath_Y,
        #     gsa.num_params,
        #     gsa.generate_gsa_indices,
        #     gsa.gsa_label,
        #     write_dir,
        #     num_steps=100,
        # )
        # conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)

    if flag_correlation:
        iterations = 200 * num_params
        gsa = CorrelationCoefficients(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
        )
        S_dict = gsa.perform_gsa()
        pearson = S_dict["pearson"]
        spearman = S_dict["spearman"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean, fig_format=fig_format)

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
            spearman, num_influential, tag=tag
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
            num_steps=10,
        )
        conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)

    if flag_eFAST:
        iterations = 6000 * num_params
        M = 4
        gsa = eFAST(
            M=M, iterations=iterations, model=model, write_dir=write_dir, seed=gsa_seed
        )
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(
            S_dict,
            S_dict_analytical=model.S_dict_analytical,
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
        # tag = "eFastTotalIndex"
        # influential_Y = val.get_influential_Y_from_gsa(total, num_influential, tag=tag)
        # t1 = time.time()
        # print("Total validation time  -> {:8.3f} s \n".format(t1 - t0))
        # val.plot_histogram_Y_all_Y_inf(
        #     influential_Y, num_influential, tag=tag, fig_format=fig_format
        # )
        #
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=100,
            M=M,
        )
        conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)

    if flag_xgboost:
        if num_params == 100 or num_params == 1000:
            num_boost_round = 100
            tuning_parameters = {
                "max_depth": 60,
                "eta": 0.1,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "refresh_leaf": True,
                "subsample": 0.6,
            }
        elif num_params == 5000:
            num_boost_round = 100
            max_depth = 60
            tuning_parameters = {
                "max_depth": max_depth,  # try 100
                "eta": 0.1,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "refresh_leaf": True,
                "subsample": 0.6,
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
        t0 = time.time()
        S_dict, r2, explained_var = gsa.perform_gsa(
            flag_save_S_dict=True, return_stats=True, verbose=False
        )
        t1 = time.time()
        fscores = S_dict["fscores"]

        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        frac_inf, frac_non_inf = val.get_fraction_identified_correctly(
            fscores, model.influential_params
        )
        print(
            "Fraction inf {0:4.3f}, fraction non-inf {1:4.3f}".format(
                frac_inf, frac_non_inf
            )
        )
        print(
            "XGBoost training results: \n "
            "  r2={0:4.3f}, explained_variance={1:4.3f} \n".format(r2, explained_var)
        )
        print("GSA indices      -> {:8.3f} s".format(t1 - t0))

        gsa.plot_sa_results(
            S_dict,
            S_boolean=model.S_boolean,
            fig_format=fig_format,
        )

        # conv = Convergence(
        #     gsa.filepath_Y,
        #     gsa.num_params,
        #     gsa.generate_gsa_indices,
        #     gsa.gsa_label,
        #     write_dir,
        #     num_steps=100,
        # )
        # conv.run_convergence(parameter_inds=parameter_inds, fig_format=fig_format)
