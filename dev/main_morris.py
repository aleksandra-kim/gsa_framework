from gsa_framework.test_functions import Morris
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import time

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. Models
    num_params = 5000
    num_influential = num_params // 100
    iterations_validation = 2000
    write_dir = path_base / "morris_model_{}".format(num_params)
    model = Morris(num_params=num_params, num_influential=num_influential)
    gsa_seed = 3407
    validation_seed = 7043

    fig_format = ["html", "pickle"]  # can have elements "pdf", "html", "pickle"

    num_params_correlation_plot = 10
    parameter_inds = list(range(num_params_correlation_plot)) + list(
        range(num_influential, num_influential + num_params_correlation_plot)
    )

    # TODO Choose which GSA to perform
    flag_correlation = 1
    flag_sobol = 0
    flag_eFAST = 0
    flag_xgboost = 0

    if flag_sobol:
        iterations = 50 * num_params
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
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
        tag = "SaltelliTotal"
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
            num_steps=100,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds, tag=tag, fig_format=fig_format
        )

    if flag_correlation:
        iterations = 2 * num_params
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
        tag = "Spearman"
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
            num_steps=100,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds,
            fig_format=fig_format,
        )

    if flag_eFAST:
        iterations = 600 * num_params
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

        t0 = time.time()
        val = Validation(
            model=model,
            iterations=iterations_validation,
            seed=validation_seed,
            default_x_rescaled=None,
            write_dir=write_dir,
        )
        tag = "eFastTotalIndex"
        influential_Y = val.get_influential_Y_from_gsa(total, num_influential, tag=tag)
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
            num_steps=100,
            M=M,
        )
        conv.run_convergence(
            parameter_inds=parameter_inds,
            fig_format=fig_format,
        )

    if flag_xgboost:
        gsa = GradientBoosting(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean)
        val = Validation(fscores, model, num_influential=model.num_influential)
        val.generate_plots(plot_histogram=True, plot_correlation=True)
