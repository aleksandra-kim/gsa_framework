import gsa_framework as gf
import brightway2 as bw
from pathlib import Path
import pickle

if __name__ == "__main__":
    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # 1. LCA model
    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    act = co.search("average consumption")[0]
    demand = {act: 1}
    method = ("IPCC 2013", "climate change", "GTP 100a")

    write_dir_lca = path_base / "lca_model"
    model_lca = gf.LCAModel(demand, method, write_dir_lca)
    const_lca = 2
    iterations_lca = model_lca.num_params * const_lca

    # 2. Moon model
    write_dir_moon = path_base / "moon_model"
    model_moon = gf.Moon()
    const_moon = 2
    iterations_moon = model_moon.num_params * const_moon

    # 3. Morris model
    write_dir_morris = path_base / "morris_model"
    model_morris = gf.Morris(num_params=100, num_influential=20)
    const_morris = 130
    iterations_morris = model_morris.num_params * const_morris

    # 3. Sobol-Levitan model
    write_dir_SL = path_base / "sobol_levitan_model"
    model_SL = gf.SobolLevitan(num_params=100)
    const_SL = 2
    iterations_SL = model_SL.num_params * const_SL

    # 4. Problem
    models_dict = {
        "lca": dict(
            model=model_lca,
            write_dir=write_dir_lca,
            iterations=iterations_lca,
            const=const_lca,
        ),
        "moon": dict(
            model=model_moon,
            write_dir=write_dir_moon,
            iterations=iterations_moon,
            const=const_moon,
        ),
        "morris": dict(
            model=model_morris,
            write_dir=write_dir_morris,
            iterations=iterations_morris,
            const=const_morris,
        ),
        "sobol_levitan": dict(
            model=model_SL,
            write_dir=write_dir_SL,
            iterations=iterations_SL,
            const=const_SL,
        ),
    }

    gsa_methods = [
        # "correlation_coefficients",
        #     'sobol_indices',
        "eFAST_indices",
        #     'xgboost',
        # 'dissimilarity_measure',
    ]

    model_names = [
        #     'lca',
        "morris",
        # "sobol_levitan",
    ]

    problem_per_worker = gf.Problem

    model_evals = []
    for model_name in model_names:
        print("\n>>> " + model_name)
        model = models_dict[model_name]["model"]
        write_dir = models_dict[model_name]["write_dir"]
        if not write_dir.exists():
            write_dir.mkdir(parents=True, exist_ok=True)
        for gsa_method in gsa_methods:
            # if gsa_method == "eFAST_indices":
            #     iterations = 130
            if gsa_method == "correlation_coefficients":
                iterations = None
            else:
                iterations = models_dict[model_name]["iterations"]
            print(
                "--- "
                + gsa_method
                + " -> "
                + str(model.num_params)
                + " parameters, "
                + str(iterations)
                + " iterations"
            )
            problem = problem_per_worker(
                sampler="random",
                model=model,
                interpreter=gsa_method,
                write_dir=write_dir,
                iterations=iterations,
                seed=34534,
                X=None,
            )
            model_evals.append(problem)

            # Plotting
            filename_sa_results = problem.gsa_dict["filename_sa_results"]
            with open(filename_sa_results, "rb") as f:
                sa_results = pickle.load(f)
            for sa_index_name, sa_index_values in sa_results.items():
                sa_indices = {sa_index_name: sa_index_values}
                try:
                    problem.plot_sa_results(sa_indices, model.influential_params)
                except:
                    pass
