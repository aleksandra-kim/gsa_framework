from gsa_framework.test_functions import SobolGstar
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import numpy as np

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. Models
    gsa_seed = 3407
    test_model_seed = 345897
    validation_seed = 7043
    num_params = 1000
    iterations_validation = 2000
    num_influential = num_params // 100
    write_dir = path_base / "sobol_Gstar_model"

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

    np.random.seed(test_model_seed)
    delta = np.random.rand(num_params)  # delta doesn't affect input importance
    np.random.seed(None)

    model = SobolGstar(
        num_params=num_params,
        num_influential=num_influential,
        a=a,
        alpha=alpha,
        delta=delta,
    )

    fig_format = "html"  # can be "pdf", "html", "pdf and html"

    num_params_corr_plot = 10
    parameter_inds = list(range(num_params_corr_plot)) + list(
        range(num_influential, num_influential + num_params_corr_plot)
    )

    # TODO Choose which GSA to perform
    flag_correlation = 0
    flag_sobol = 1
    flag_eFAST = 0
    flag_xgboost = 0

    if flag_sobol:
        iterations = 1000 * num_params
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(
            S_dict,
            S_dict_analytical=model.S_dict_analytical,
            fig_format=fig_format,
        )
        # val = Validation(
        #     model=model,
        #     iterations=iterations_validation,
        #     seed=validation_seed,
        #     default_x_rescaled=None,
        #     write_dir=write_dir,
        # )
        # tag = "SaltelliTotal"
        # influential_Y = val.get_influential_Y_from_gsa(total, num_influential, tag=tag)
        # val.plot_histogram_base_Y_influential_Y(
        #     influential_Y, fig_format=fig_format, tag=tag
        # )
        # conv = Convergence(
        #     gsa.filepath_Y,
        #     gsa.num_params,
        #     gsa.generate_gsa_indices,
        #     gsa.gsa_label,
        #     write_dir,
        #     num_steps=10,
        # )
        # conv.run_convergence(
        #     parameter_inds=parameter_inds, tag=tag, fig_format=fig_format
        # )

    if flag_correlation:
        iterations = 2 * num_params
        gsa = CorrelationCoefficients(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
        )
        S_dict = gsa.perform_gsa()
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean, fig_format=fig_format)
        pearson = S_dict["pearson"]
        spearman = S_dict["spearman"]
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "Spearman"
        influential_Y = val.get_influential_Y_from_gsa(
            spearman, num_influential, tag=tag
        )
        val.plot_histogram_base_Y_influential_Y(
            influential_Y, fig_format=fig_format, tag=tag
        )
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=100,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds, tag=tag, fig_format=fig_format
        )

    if flag_eFAST:
        iterations = 60 * num_params
        M = 2
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
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "eFastTotal.M{}".format(M)
        influential_Y = val.get_influential_Y_from_gsa(total, num_influential, tag=tag)
        val.plot_histogram_base_Y_influential_Y(
            influential_Y, fig_format=fig_format, tag=tag
        )
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir,
            num_steps=10,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds, tag=tag, fig_format=fig_format
        )

    if flag_xgboost:
        gsa = GradientBoosting(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean)
        val = Validation(fscores, model, num_influential=model.num_influential)
        val.generate_plots(plot_histogram=True, plot_correlation=True)
