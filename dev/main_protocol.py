from gsa_framework.models.life_cycle_assessment import LCAModel
from pathlib import Path
import time
import numpy as np
import bw2data as bd
import bw2calc as bc

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )
    # path_base = Path('/data/user/kim_a/protocol_gsa')
    write_dir = path_base / "only_tech"
    bd.projects.set_current("GSA for protocol")
    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a")
    num_params = 10000
    uncertain_exchanges_types = ["tech", "bio", "cf"]
    model = LCAModel(
        demand,
        method,
        write_dir,
        num_params=num_params,
        uncertain_exchanges_types=uncertain_exchanges_types,
    )
    print("sdf")
