import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc
from gsa_framework.utils import read_pickle, write_pickle
from gsa_framework.models.life_cycle_assessment import LCAModel
from dev.utils_graph_traversal import filter_uncertain_technosphere_exchanges

# def filter_technosphere_exchanges(lca, cutoff=0.005, max_calc=1e4):
#     """Use brightway's GraphTraversal to identify the relevant
#     technosphere exchanges in a non-stochastic LCA."""
#     start = time()
#     res = bc.GraphTraversal().calculate(
#         lca.demand, lca.method, cutoff=cutoff, max_calc=max_calc
#     )
#
#     # get all edges
#     technosphere_exchange_indices = []
#     for e in res["edges"]:
#         if e["to"] != -1:  # filter out head introduced in graph traversal
#             technosphere_exchange_indices.append((e["from"], e["to"]))
#     print(
#         "TECHNOSPHERE filtering resulted in {} exchanges and took {} iterations in {} seconds.".format(
#             len(technosphere_exchange_indices),
#             res["counter"],
#             np.round(time() - start, 2),
#         )
#     )
#     return technosphere_exchange_indices

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )
    # path_base = Path('/data/user/kim_a/protocol_gsa')
    write_dir = path_base / "protocol_gsa_paper3"
    write_dir_sct = write_dir / "supply_chain"
    write_dir_sct.mkdir(exist_ok=True, parents=True)

    bd.projects.set_current("GSA for protocol paper 3")
    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    static_lca = bc.LCA(demand, uncertain_method, use_distributions=False)
    static_lca.lci()
    static_lca.lcia()

    lca = bc.LCA(demand, uncertain_method, use_distributions=True)
    lca.lci()
    lca.lcia()

    ei_tech_group = [
        g
        for g in lca.technosphere_mm.groups
        if "tech" in g.label and "ecoinvent" in g.label
    ]
    assert len(ei_tech_group) == 1
    ei_tech_group = ei_tech_group[0]
    tech_params = ei_tech_group.package.data[2]

    cutoff = 0.01
    max_calc = 1e3
    sct_tech_params_filename = "tech_params_cutoff{}_maxcalc{}.pickle".format(
        cutoff, int(max_calc)
    )
    sct_tech_params_filepath = write_dir_sct / sct_tech_params_filename
    if not sct_tech_params_filepath.exists():
        tech_params_sct = filter_uncertain_technosphere_exchanges(lca, cutoff=1e-6)
        write_pickle(tech_params_sct, sct_tech_params_filepath)
    else:
        tech_params_sct = read_pickle(sct_tech_params_filepath)

    # Local SA
    model = LCAModel(
        demand,
        uncertain_method,
        write_dir,
    )

    print()
