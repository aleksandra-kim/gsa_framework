import numpy as np
import bw2data as bd
import bw2calc as bc

from dev.utils_graph_traversal import (
    filter_uncertain_technosphere_exchanges,
    filter_uncertain_biosphere_exchanges,
    filter_uncertain_characterization_exchanges,
    collect_uncertain_exchanges,
)


num_params = 200

bd.projects.set_current("GSA for protocol")
co = bd.Database("CH consumption 1.0")
demand_act = [act for act in co if "Food" in act["name"]]
assert len(demand_act) == 1
demand_act = demand_act[0]
demand = {demand_act: 1}
method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
lca = bc.LCA(demand, method)
lca.lci()
lca.lcia()

cutoff = 0.007  # percentage of the total score, cutoff=0.005 means 0.5 percent
score_cutoff = cutoff * lca.score

# tech_inds = filter_technosphere_exchanges(lca, cutoff)
tech_inds_uncertain = filter_uncertain_technosphere_exchanges(lca, cutoff)
# bio_inds  = filter_biosphere_exchanges(lca, cutoff)
bio_inds_uncertain = filter_uncertain_biosphere_exchanges(lca, cutoff)
# ch_inds = filter_characterization_exchanges(lca, cutoff)
cf_inds_uncertain = filter_uncertain_characterization_exchanges(lca, cutoff)

exchanges_dict = {
    "tech": tech_inds_uncertain,
    "bio": bio_inds_uncertain,
    "cf": cf_inds_uncertain,
}

subset_exchanges_dict = collect_uncertain_exchanges(exchanges_dict, num_params)

uncertain_params = {
    "tech": lca.tech_params[lca.tech_params["uncertainty_type"] > 1],
    "bio": lca.bio_params[lca.bio_params["uncertainty_type"] > 1],
    "cf": lca.cf_params[lca.cf_params["uncertainty_type"] > 1],
}
row_ind = 0
col_ind = 1
parameter_choice_dict = {}
for uncertain_exchange_type, exc_list in subset_exchanges_dict.items():
    params = uncertain_params[uncertain_exchange_type]
    where_list = []
    if uncertain_exchange_type != "cf":
        for exc in exc_list:
            where = np.where(
                np.logical_and(
                    params["row"] == exc[row_ind],
                    params["col"] == exc[col_ind],
                )
            )[0]
            assert len(where) == 1
            where_list.append(where[0])
    else:
        for exc in exc_list:
            where = np.where(params["row"] == exc[row_ind])[0]
            assert len(where) == 1
            where_list.append(where[0])
    where_list = sorted(where_list)
    parameter_choice_dict[uncertain_exchange_type] = where_list

print()
