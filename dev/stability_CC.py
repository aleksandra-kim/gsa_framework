from gsa_framework.lca import LCAModel
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.sensitivity_analysis.correlations import corrcoef_parallel
from gsa_framework.convergence import Convergence
from pathlib import Path
import brightway2 as bw
import time
import numpy as np
from gsa_framework.utils import write_pickle

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
    num_influential = num_params // 100
    iterations_validation = 500
    write_dir = path_base / "lca_model_{}".format(num_params)
    model = LCAModel(demand, method, write_dir, num_params=num_params)
    gsa_seed = 3403
    validation_seed = 7043
    fig_format = ["html", "pickle"]

    iterations = 2 * num_params
    gsa = CorrelationCoefficients(
        iterations=iterations,
        model=model,
        write_dir=write_dir,
        seed=gsa_seed,
    )
    #     S_dict = gsa.perform_gsa()
    S_dict = gsa.generate_gsa_indices()
    pearson = S_dict["pearson"]
    spearman = S_dict["spearman"]
    gsa.plot_sa_results(S_dict, fig_format=fig_format)

    conv = Convergence(
        gsa.filepath_Y,
        gsa.num_params,
        gsa.generate_gsa_indices,
        gsa.gsa_label,
        write_dir,
        num_steps=100,
    )
    num_bootstrap = 10
    np.random.seed(gsa_seed)
    stability_seeds = np.random.randint(
        low=0,
        high=2147483647,
        size=(len(conv.iterations_for_convergence), num_bootstrap),
    )
    for i, step in enumerate(conv.iterations_for_convergence):
        print(step)
        for seed in stability_seeds[i, :]:
            filename = "step{}.seed{}.pickle".format(step, seed)
            filepath = (
                write_dir / "stability_intermediate_{}".format(gsa.gsa_label) / filename
            )
            if not filepath.exists():
                X = np.random.rand(step, num_params)
                Xr = model.rescale(X)
                y = model(Xr)
                S = corrcoef_parallel(y, X)
                write_pickle(S, filepath)
            else:
                print("{} already exists".format(filepath.name))
