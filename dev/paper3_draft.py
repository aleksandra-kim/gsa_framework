# import numpy as np
# from pathlib import Path
# import bw2data as bd
# import bw2calc as bc
#
# from gsa_framework.utils import read_pickle, write_pickle
#
# from dev.utils_graph_traversal import filter_technosphere_exchanges

# def filter_technosphere_exchanges_with_uncertainty(lca, cutoff=0.005, max_calc=1e4):
#     """Use brightway's GraphTraversal to identify the relevant
#     technosphere exchanges in a non-stochastic LCA."""
#     technosphere_exchange_indices = filter_uncertain_technosphere_exchanges(lca, cutoff, max_calc)
#
#     return technosphere_exchange_indices

from pathlib import Path
import bw2io as bi

dir_lca_files = Path("/Users/akim/Documents/LCA_files/")
project_filepath = (
    dir_lca_files / "brightway2-project-GSA-backup.16-November-2021-11-50AM.tar.gz"
)
bi.restore_project_directory(project_filepath)

import bw2data as bd

bd.projects.set_current("GSA")


print(len(ei))

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )
    # path_base = Path('/data/user/kim_a/protocol_gsa')
    write_dir = path_base / "realistic_gsa"
    write_dir_sct = write_dir / "supply_chain"
    write_dir_sct.mkdir(exist_ok=True, parents=True)

    bd.projects.set_current("GSA for realistic models")
    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA(demand, uncertain_method)
    lca.lci()
    lca.lcia()

    ei = bd.Database("ecoinvent 3.8 cutoff")
    diesel_acts = {act["name"] for act in ei if "diesel" in act["name"].lower()}

    cutoff = 0.01
    max_calc = 1e3
    sct_tech_params_filename = "tech_params_cutoff{}_maxcalc{}.pickle".format(
        cutoff, int(max_calc)
    )
    sct_tech_params_filepath = write_dir_sct / sct_tech_params_filename
    if not sct_tech_params_filepath.exists():
        tech_params = filter_technosphere_exchanges(lca, cutoff=0.01, max_calc=1e3)
        write_pickle(tech_params, sct_tech_params_filepath)
    else:
        tech_params = read_pickle(sct_tech_params_filepath)

    tech_group = [
        g
        for g in lca.technosphere_mm.groups
        if "ecoinvent_3.8_cutoff_technosphere_matrix" in g.label
    ]
    assert len(tech_group) == 1
    tech_group = tech_group[0]

    tech_package = [
        p[0][0]
        for p in lca.technosphere_mm.packages.values()
        if "ecoinvent_3.8_cutoff_technosphere_matrix" in p[0].label
    ]
    assert len(tech_package) == 1
    tech_package = tech_package[0]

    # uncertainties = get_uncertatinties()

    print()
