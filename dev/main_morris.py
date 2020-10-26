from gsa_framework.test_functions import Morris
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. Models
    seed = 3403
    iterations = 200000
    num_params = 2000
    num_influential = 250
    write_dir = path_base / "morris_model"
    model = Morris(num_params=num_params, num_influential=num_influential)

    parameter_inds = [0, 1, 2, 3, 4, 5, 310, 320, 330, 340, 350]

    # TODO Choose which GSA to perform
    flag_correlation = 0
    flag_eFAST = 0
    flag_sobol = 1
    flag_xgboost = 0

    if flag_correlation:
        gsa = CorrelationCoefficients(
            iterations=iterations, model=model, write_dir=write_dir, seed=seed
        )
        S_dict = gsa.perform_gsa()
        pearson = S_dict["pearson"]
        spearman = S_dict["spearman"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean)
        val = Validation(spearman, model, num_influential=model.num_influential)
        val.generate_plots(plot_histogram=True, plot_correlation=True)
        # conv = Convergence(
        #     gsa.filepath_Y,
        #     gsa.num_params,
        #     gsa.generate_gsa_indices,
        #     gsa.gsa_label,
        #     write_dir / "figures",
        #     num_steps=100,
        # )
        # conv.run_convergence()

    if flag_sobol:
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict, S_dict_analytical=model.S_dict_analytical)
        val = Validation(total, model, num_influential=model.num_influential - 50)
        val.generate_plots()
        conv = Convergence(
            gsa.filepath_Y,
            gsa.num_params,
            gsa.generate_gsa_indices,
            gsa.gsa_label,
            write_dir / "figures",
            num_steps=100,
            step_multiplier=2,
        )
        conv.run_convergence(parameter_inds)

    if flag_eFAST:
        gsa = eFAST(
            M=4, iterations=iterations, model=model, write_dir=write_dir, seed=seed
        )
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict, S_dict_analytical=model.S_dict_analytical)
        val = Validation(total, model, num_influential=model.num_influential)
        val.generate_plots()

    if flag_xgboost:
        gsa = GradientBoosting(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean)
        val = Validation(fscores, model, num_influential=model.num_influential)
        val.generate_plots(plot_histogram=True, plot_correlation=True)
