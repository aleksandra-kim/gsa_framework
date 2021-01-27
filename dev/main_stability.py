from gsa_framework.convergence import Convergence
from pathlib import Path
from gsa_framework.utils import *
from gsa_framework.stability_convergence_metrics import Stability
from gsa_framework.plotting import max_min_band_many, ranking_convergence_many
from setups_paper_gwp import setup_salt, setup_morris4_model, setup_lca_model_paper
from gsa_framework.sensitivity_analysis.saltelli_sobol import *

# 1. Choose which stability dictionaries to include
path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files")

# 1. Models
# models = ["1000_morris4", "5000_morris4", "10000_morris4", "10000_lca"]
# models = ["1000_lca"]
#
# data_dicts, ranking_stats = {}, {}
# for model in models:
#     if "morris" in model:
#         write_dir = path_base / model
#     elif "lca" in model:
#         write_dir = path_base / "lca_model_food_10000"
#     write_arr = write_dir / "arrays"
#     files = [x for x in write_arr.iterdir() if x.is_file() and "stability." in x.name]
#     files = sorted(files)
#     stability_dicts = []
#     for file in files:
#         stability_dict = read_pickle(file)
#         if 'correlations' in file.name:
#             stability_dict_abs = {}
#             for k,v in stability_dict.items():
#                 stability_dict_abs[k] = {
#                     'spearman': np.abs(v['spearman'])
#                 }
#             stability_dict = stability_dict_abs
#         stability_dicts.append(stability_dict)
#     st = Stability(stability_dicts, write_dir)
#     data_dicts[model] = st.confidence_intervals_max
#     ranking_stats[model] = st.stat_ranking_all_steps()
# fig = max_min_band_many(data_dicts)
#
# ranks = st.rankings['delta']['ranks'][48]
# sindices = st.sa_stability_dict['delta']['bootstrap'][48]
# st.stat_ranking(ranks, sindices)
# ranking_convergence_many(ranking_stats)


if __name__ == "__main__":

    # Sobol stability dictionaries
    num_params = 10000
    iterations = 40 * num_params
    num_steps = 50
    num_bootstrap = 60

    gsa = setup_salt(num_params, iterations, setup_lca_model_paper, path_base)

    # Convergence class
    conv = Convergence(
        gsa.filepath_Y,
        gsa.num_params,
        gsa.generate_gsa_indices,
        gsa.gsa_label,
        gsa.write_dir,
        num_steps=num_steps,
    )
    stability_seeds = np.random.randint(
        low=0,
        high=2147483647,
        size=(len(conv.iterations_for_convergence), num_bootstrap),
    )

    filename_S = "stability.S.{}.{}.{}Step{}.{}.{}.pickle".format(
        gsa.gsa_label,
        gsa.sampling_label,
        gsa.iterations,
        conv.iterations_step,
        num_bootstrap,
        gsa.seed,
    )
    filepath_S = gsa.write_dir / "arrays" / filename_S
    if filepath_S.exists():
        print("--> {} already exists".format(filename_S))
        S_dict_stability = read_pickle(filepath_S)
    else:
        Y = read_hdf5_array(gsa.filepath_Y).flatten()
        S_dict_stability = sobol_indices_stability(
            Y,
            num_params,
            conv.iterations_for_convergence,
            num_bootstrap,
            stability_seeds,
        )
        write_pickle(S_dict_stability, filepath_S)
