from gsa_framework.stability_convergence_metrics import *
from pathlib import Path
from gsa_framework.utils import read_pickle
from gsa_framework.stability_convergence_metrics import Stability
from gsa_framework.plotting import max_min_band_many, ranking_convergence_many

# 1. Choose which stability dictionaries to include
path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files")

# 1. Models
models = ["1000_morris4", "5000_morris4", "10000_morris4", "10000_lca"]
# models = ["1000_morris4"]
ranking_stats = {}
for model in models:
    if "morris" in model:
        write_dir = path_base / model
    elif "lca" in model:
        write_dir = path_base / "lca_model_food_10000"
    write_arr = write_dir / "arrays"
    files = [x for x in write_arr.iterdir() if x.is_file() and "stability." in x.name]
    files = sorted(files)
    stability_dicts = []
    for file in files:
        stability_dict = read_pickle(file)
        stability_dicts.append(stability_dict)
    st = Stability(stability_dicts, write_dir)
    # data_dicts[model] = st.confidence_intervals_max
    ranking_stats[model] = st.stat_ranking_all_steps()
# fig = max_min_band_many(data_dicts)

# ranks = st.rankings['delta']['ranks'][48]
# sindices = st.sa_stability_dict['delta']['bootstrap'][48]
# st.stat_ranking(ranks, sindices)

ranking_convergence_many(ranking_stats)
