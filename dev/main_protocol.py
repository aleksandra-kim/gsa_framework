from gsa_framework.models import LCAModel
from pathlib import Path
import time
import numpy as np
import bw2data as bd
import bw2calc as bc

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/protocol_gsa/"
    )

    # LCA model
    bd.projects.set_current("GSA for protocol")
    co = bd.Database("CH consumption 1.0")
    demand_act = [
        act for act in co if "Food and non-alcoholic beverages sector" in act["name"]
    ][0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    # Define some variables
    write_dir = path_base
    iterations = 10
    num_params = 10000

    model = LCAModel(demand, method, write_dir, num_params=num_params)
    X = np.random.rand(iterations, num_params)
    X_rescaled = model.rescale(X)
    lca_scores = model(X_rescaled)

    print(model)
