from gsa_framework.lca import LCAModel
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from pathlib import Path
import brightway2 as bw

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # LCA model
    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    act = co.search("average consumption")[0]
    demand = {act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")
    write_dir = path_base / "lca_model"
    var_threshold = 1e-10
    model = LCAModel(demand, method, write_dir, var_threshold=var_threshold)

    # Define some variables
    seed = 3403
    iterations = 1000
    num_params = len(model)
    num_influential = 100
    iterations_validation = 500

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
        S_pearson = S_dict["pearson"]
        S_spearman = S_dict["spearman"]
        gsa.plot_sa_results(S_dict)
        validation = Validation(
            S_spearman,
            model,
            iterations=iterations_validation,
            num_influential=num_influential,
        )
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_eFAST:
        gsa = eFAST(
            M=4, iterations=iterations, model=model, write_dir=write_dir, seed=seed
        )
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict)
        validation = Validation(
            total,
            model,
            iterations=iterations_validation,
            num_influential=num_influential,
        )
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_sobol:
        gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        first = S_dict["First order"]
        total = S_dict["Total order"]
        gsa.plot_sa_results(S_dict)
        validation = Validation(total, model, num_influential=num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)

    if flag_xgboost:
        gsa = GradientBoosting(iterations=iterations, model=model, write_dir=write_dir)
        S_dict = gsa.perform_gsa()
        fscores = S_dict["fscores"]
        gsa.plot_sa_results(S_dict)
        validation = Validation(fscores, model, num_influential=num_influential)
        validation.generate_plots(plot_histogram=True, plot_correlation=True)
