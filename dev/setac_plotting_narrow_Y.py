from gsa_framework.utils import read_hdf5_array, write_hdf5_array
from pathlib import Path
import brightway2 as bw
from copy import deepcopy
from gsa_framework.utils_setac_lca import get_xgboost_params, plot_base_narrow_Y
import pickle
from gsa_framework.lca import LCAModelSetac
import numpy as np

num_params_narrow = 12

path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa/")
path_setac = path_base / "setac_gsa"
path_merlin = path_setac / "merlin"

# 1. LCA setup
# LCA model
iterations = 10
bw.projects.set_current("GSA for setac")
co = bw.Database("CH consumption 1.0")
demand_act = [act for act in co if "Food" in act["name"]][0]
demand = {demand_act: 1}
method = ("IPCC 2013", "climate change", "GTP 100a")
lca = bw.LCA(demand, method)
lca.lci()
lca.lcia()

num_params = len(lca.tech_params[lca.tech_params["uncertainty_type"] > 1])
print(num_params)

seed = 923458
filename_Y_narrow = "validation.narrow.Y.{}.{}.{}.hdf5".format(
    iterations, num_params, seed
)
filepath_Y_narrow = path_setac / "arrays" / filename_Y_narrow
if filepath_Y_narrow.exists():
    narrow_Y = read_hdf5_array(filepath_Y_narrow).flatten()
else:
    tech_params_narrow = deepcopy(lca.tech_params)
    # Read params_yes from xgboost model
    filepath_params_yes_0 = path_merlin / "params_yes_0.pickle"
    with open(filepath_params_yes_0, "rb") as f:
        params_yes_0 = pickle.load(f)
    model_seed = 78997
    path_model_dir = path_setac / "regression" / "{}_model".format(model_seed)
    model, params_yes_xgboost = get_xgboost_params(path_model_dir, params_yes_0)
    params_yes = params_yes_xgboost[:num_params_narrow]
    tech_params_narrow[params_yes]["scale"] = (
        tech_params_narrow[params_yes]["scale"] / 2
    )

    lca_model = LCAModelSetac(demand, method, tech_params_narrow)

    filename_X_unitcube = "validation.base.X.unitcube.{}.{}.{}.hdf5".format(
        iterations, num_params, seed
    )
    filepath_base_X_unitcube = path_setac / "arrays" / filename_X_unitcube
    if filepath_base_X_unitcube.exists():
        X_unitcube = read_hdf5_array(filepath_base_X_unitcube)
    else:
        np.random.rand(seed)
        X_unitcube = np.random.rand(iterations, num_params)
        write_hdf5_array(X_unitcube, filepath_base_X_unitcube)

    filename_X_rescaled_narrow = "validation.narrow.X.rescaled.{}.{}.{}.hdf5".format(
        iterations, num_params, seed
    )
    filepath_X_rescaled_narrow = path_setac / "arrays" / filename_X_rescaled_narrow
    if filepath_X_rescaled_narrow.exists():
        X_rescaled_narrow = read_hdf5_array(filepath_X_rescaled_narrow)
    else:
        X_rescaled_narrow = lca_model.rescale(X_unitcube, filepath_X_rescaled_narrow)
        write_hdf5_array(X_rescaled_narrow, filepath_X_rescaled_narrow)
    narrow_Y = lca_model(X_rescaled_narrow)
    write_hdf5_array(narrow_Y, filepath_Y_narrow)

filename_base_Y = "validation.base.Y.{}.{}.{}.hdf5".format(iterations, num_params, seed)
filepath_base_Y = path_setac / "arrays" / filename_base_Y
base_Y = read_hdf5_array(filepath_base_Y).flatten()
plot_base_narrow_Y(narrow_Y, narrow_Y)

print("asdfasd")
