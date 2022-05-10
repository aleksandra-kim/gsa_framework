from pathlib import Path
import numpy as np

import bw2calc as bc
import bw2data as bd
import bw2io as bi
import bw_processing as bwp

from gsa_framework.models.life_cycle_assessment_bw25 import LCAModel25

path_base = Path("/Users/akim/Documents/LCA_files/")
write_dir = Path("/Users/akim/PycharmProjects/akula/dev/write_files/")
fp_datapackages = path_base / "bw_projects"
fp_virtual_markets = fp_datapackages / "virtual-markets.zip"

use_exiobase = False
cutoff = 1e-16

co_name = "swiss consumption 1.0"
co = bd.Database(co_name)
if use_exiobase:
    project = "GSA for archetypes with exiobase"
else:
    project = "GSA for archetypes"
bd.projects.set_current(project)
write_dir_project = write_dir / project.lower().replace(" ", "_")
write_dir_project.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print()


# hh_average = [act for act in co if "ch hh average consumption aggregated" == act['name']]
# assert len(hh_average) == 1
# demand_act = hh_average[0]
# demand = {demand_act: 1}
# method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
#
# write_dir_gsa = write_dir_project / demand_act['name'].lower().replace(" ", "_")
# write_dir_gsa.mkdir(parents=True, exist_ok=True)
# model = LCAModel25(demand, method, write_dir_gsa)
# res = model.get_graph_traversal_params(cutoff=cutoff)
# for g in model.lca.technosphere_mm.groups:
#     g.calculate()
# indices = model.lca.technosphere_mm.input_provenance()

print()
