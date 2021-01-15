from gsa_framework.stability_convergence_metrics import *
from pathlib import Path
from gsa_framework.utils import read_pickle

filename_delta = "stability.S.deltaGsaNr1.latinSampling.8000Step160.60.3407.pickle"
filename_spear = "stability.S.correlationsGsa.randomSampling.4000Step80.60.3407.pickle"
filepath_stability = Path("write_files") / filename_delta
stability_dict = read_pickle(filepath_stability)
parameters = [0, 1, 2, 10, 20, 30]
sb_parameters = get_ci_convergence_per_parameter(
    stability_dict,
    ci_option="student",
    sensitivity_index_names=["delta"],
    parameters=parameters,
)
plot_confidence_convergence(sb_parameters)
