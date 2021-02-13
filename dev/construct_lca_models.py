from gsa_framework.lca import LCAModel
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.validation import Validation
from pathlib import Path
import brightway2 as bw
import time
import numpy as np
from gsa_framework.plotting import plot_histogram_Y1_Y2
from gsa_framework.utils import read_hdf5_array

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # LCA model
    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    act = [act for act in co if "Food" in act["name"]][0]
    demand = {act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")

    # Define some variables
    num_params = 10000
    iterations_validation = 2000
    write_dir = path_base / "lca_model_{}".format(num_params)
    model = LCAModel(demand, method, write_dir)  # TODO add num_params later
    gsa_seed = 3403
    validation_seed = 7043
    fig_format = ["html", "pickle"]
    parameter_inds_convergence_plot = [0, 1, 2]  # TODO

    # Make sure  that the chosen num_params in LCA are appropriate
    val = Validation(
        model=model,
        iterations=2000,
        seed=4444,
        default_x_rescaled=model.default_uncertain_amounts,
        write_dir=write_dir,
    )
    tag = "numParams{}".format(num_params)
    scores_dict = model.get_lsa_scores_pickle(model.write_dir / "LSA_scores")
    uncertain_tech_params_where_subset, _ = model.get_nonzero_params_from_num_params(
        scores_dict, num_params
    )
    parameter_choice = []
    for u in uncertain_tech_params_where_subset:
        where_temp = np.where(model.uncertain_tech_params_where == u)[0]
        assert len(where_temp) == 1
        parameter_choice.append(where_temp[0])
    parameter_choice.sort()
    Y_subset = val.get_influential_Y_from_parameter_choice(
        parameter_choice=parameter_choice, tag=tag
    )
    val.plot_histogram_Y_all_Y_inf(Y_subset, num_influential=num_params)
