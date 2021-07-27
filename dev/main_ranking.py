import numpy as np
from pathlib import Path
from gsa_framework.models.test_functions import Morris4
from gsa_framework.utils import read_pickle
from gsa_framework.convergence_robustness_validation import Robustness

path_base = Path("/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/")
num_params = 1000
write_dir = path_base / "{}_morris4".format(num_params)
write_dir_fig = path_base / "paper_figures_review1"
write_dir_arr = write_dir / "arrays"
filepath_stability_dict = (
    write_dir_arr
    / "stability.S.correlationsGsa.randomSampling.4000Step80.120.3407_6000814.pickle"
)
filepath_S = (
    "stability.S.correlationsGsa.randomSampling.4000Step80.120.3407_6000814.pickle"
)
fig_format = ["pdf"]
opacity = 0.6

num_influential = num_params // 100
model = Morris4(num_params=num_params, num_influential=num_influential)
morris_model_name = r"$\underline{\text{Morris model, 1'000 inputs}}$"

S = read_pickle(filepath_S)
sa_names = {
    "spearman": "corr",
}
stability_dict = read_pickle(filepath_stability_dict)
stability_dicts = {"corr": stability_dict}
num_ranks = 4
st = Robustness(
    stability_dicts,
    write_dir,
    num_ranks=num_ranks,
    bootstrap_ranking_tag="paper1_review1",
    ci_type="student",
)
