from gsa_framework.test_functions import SobolGstar
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from pathlib import Path

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. Models
    seed = 3403
    iterations = 500000
    num_params = 100
    num_influential = 25
    model = SobolGstar(num_params=num_params, num_influential=num_influential)
    write_dir = path_base / "sobol_Gstar_model"

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
        validation = Validation(spearman, model, num_influential=model.num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_eFAST:
        gsa = eFAST(
            M=4, iterations=iterations, model=model, write_dir=write_dir, seed=seed
        )
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict, S_dict_analytical=model.S_dict_analytical)
        validation = Validation(total, model, num_influential=model.num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_sobol:
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict, S_dict_analytical=model.S_dict_analytical)
        validation = Validation(total, model, num_influential=model.num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_xgboost:
        gsa = GradientBoosting(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(S_dict, S_boolean=model.S_boolean)
        validation = Validation(fscores, model, num_influential=model.num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)
