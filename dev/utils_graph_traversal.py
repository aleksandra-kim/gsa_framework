import numpy as np
from time import time
import bw2calc as bc


def filter_technosphere_exchanges(lca, cutoff=0.005, max_calc=1e4):
    """Use brightway's GraphTraversal to identify the relevant
    technosphere exchanges in a non-stochastic LCA."""
    start = time()
    res = bc.GraphTraversal().calculate(
        lca.demand, lca.method, cutoff=cutoff, max_calc=max_calc
    )

    # get all edges
    technosphere_exchange_indices = []
    for e in res["edges"]:
        if e["to"] != -1:  # filter out head introduced in graph traversal
            technosphere_exchange_indices.append((e["from"], e["to"]))
    print(
        "TECHNOSPHERE {} filtering resulted in {} of {} exchanges and took {} iterations in {} seconds.".format(
            res["lca"].technosphere_matrix.shape,
            len(technosphere_exchange_indices),
            res["lca"].technosphere_matrix.getnnz(),
            res["counter"],
            np.round(time() - start, 2),
        )
    )
    return technosphere_exchange_indices


def filter_uncertain_technosphere_exchanges(lca, cutoff=0.005, max_calc=1e4):
    """Use brightway's GraphTraversal to identify the relevant
    technosphere exchanges in a non-stochastic LCA."""
    start = time()
    res = bc.GraphTraversal().calculate(
        lca.demand, lca.method, cutoff=cutoff, max_calc=max_calc
    )
    # get all edges
    uncertain_technosphere_exchange_indices = []
    for e in res["edges"]:
        if e["to"] != -1:  # filter out head introduced in graph traversal
            uncertainty_distribution = lca.tech_params[
                np.logical_and(
                    lca.tech_params["row"] == e["from"],
                    lca.tech_params["col"] == e["to"],
                )
            ]["uncertainty_type"]
            if uncertainty_distribution > 1:
                uncertain_technosphere_exchange_indices.append(
                    (e["from"], e["to"], e["impact"])
                )
    print(
        "TECHNOSPHERE {} filtering resulted in {} of {} UNCERTAIN exchanges and took {} iterations in {} seconds.".format(
            res["lca"].technosphere_matrix.shape,
            len(uncertain_technosphere_exchange_indices),
            res["lca"].technosphere_matrix.getnnz(),
            res["counter"],
            np.round(time() - start, 2),
        )
    )
    return uncertain_technosphere_exchange_indices


def filter_biosphere_exchanges(lca, cutoff=0.005):
    """Reduce biosphere exchanges to those that matter for a given impact
    category in a non-stochastic LCA."""
    start = time()

    # print('LCA score:', lca.score)
    inv = lca.characterized_inventory
    # print('Characterized inventory:', inv.shape, inv.nnz)
    finv = inv.multiply(abs(inv) > abs(lca.score * cutoff))
    # print('Filtered characterized inventory:', finv.shape, finv.nnz)
    biosphere_exchange_indices = list(zip(*finv.nonzero()))
    # print(biosphere_indices[:2])
    explained_fraction = finv.sum() / lca.score
    # print('Explained fraction of LCA score:', explained_fraction)
    print(
        "BIOSPHERE {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.".format(
            inv.shape,
            finv.nnz,
            inv.nnz,
            np.round(explained_fraction * 100, 2),
            np.round(time() - start, 2),
        )
    )
    return biosphere_exchange_indices


def filter_uncertain_biosphere_exchanges(lca, cutoff=0.005):
    """Reduce biosphere exchanges to those that matter for a given impact
    category in a non-stochastic LCA."""
    start = time()

    # print('LCA score:', lca.score)
    inv = lca.characterized_inventory
    # print('Characterized inventory:', inv.shape, inv.nnz)
    finv = inv.multiply(abs(inv) > abs(lca.score * cutoff))
    # print('Filtered characterized inventory:', finv.shape, finv.nnz)
    biosphere_exchange_indices = list(zip(*finv.nonzero()))
    uncertain_biosphere_exchange_indices = []
    explained_impact = 0
    for row, col in biosphere_exchange_indices:
        uncertainty_distribution = lca.bio_params[
            np.logical_and(
                lca.bio_params["row"] == row,
                lca.bio_params["col"] == col,
            )
        ]["uncertainty_type"]
        if uncertainty_distribution > 1:
            impact = finv[row, col]
            uncertain_biosphere_exchange_indices.append((row, col, impact))
            explained_impact += impact
    # print(biosphere_indices[:2])
    explained_fraction = explained_impact / lca.score
    # print('Explained fraction of LCA score:', explained_fraction)
    print(
        "BIOSPHERE {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.".format(
            inv.shape,
            len(uncertain_biosphere_exchange_indices),
            inv.nnz,
            np.round(explained_fraction * 100, 2),
            np.round(time() - start, 2),
        )
    )
    return uncertain_biosphere_exchange_indices


def filter_characterization_exchanges(lca, cutoff=0.005):
    start = time()
    inv_sum = np.array(np.sum(lca.characterized_inventory, axis=1)).squeeze()
    # print('Characterized inventory:', inv.shape, inv.nnz)
    finv_sum = inv_sum * abs(inv_sum) > abs(lca.score * cutoff)
    characterization_exchange_indices = list(finv_sum.nonzero()[0])
    explained_fraction = finv_sum.sum() / lca.score
    # print('Explained fraction of LCA score:', explained_fraction)
    print(
        "CHARACTERIZATION {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.".format(
            inv_sum.shape,
            len(characterization_exchange_indices),
            inv_sum.shape,
            np.round(explained_fraction * 100, 2),
            np.round(time() - start, 2),
        )
    )
    return characterization_exchange_indices


def filter_uncertain_characterization_exchanges(lca, cutoff=0.005):
    start = time()
    inv_sum = np.array(np.sum(lca.characterized_inventory, axis=1)).squeeze()
    # print('Characterized inventory:', inv.shape, inv.nnz)
    mask = abs(inv_sum) > abs(lca.score * cutoff)
    finv_sum = inv_sum * mask
    characterization_exchange_indices = list(finv_sum.nonzero()[0])
    uncertain_characterization_exchange_indices = []
    explained_impact = 0
    for row in characterization_exchange_indices:
        uncertainty_distribution = lca.cf_params[lca.cf_params["row"] == row][
            "uncertainty_type"
        ]
        if uncertainty_distribution > 1:
            impact = inv_sum[row]
            uncertain_characterization_exchange_indices.append((row, None, impact))
            explained_impact += impact
    explained_fraction = explained_impact / lca.score
    # print('Explained fraction of LCA score:', explained_fraction)
    print(
        "CHARACTERIZATION {} filtering resulted in {} of {} exchanges ({}% of total impact) and took {} seconds.".format(
            inv_sum.shape,
            len(uncertain_characterization_exchange_indices),
            inv_sum.shape,
            np.round(explained_fraction * 100, 2),
            np.round(time() - start, 2),
        )
    )
    return uncertain_characterization_exchange_indices


def collect_uncertain_exchanges(exchanges_dict, num_exchanges):
    where_impact = 2
    impacts = [
        exc[where_impact] for exc_list in exchanges_dict.values() for exc in exc_list
    ]
    impacts = np.array(impacts)
    impacts_threshold = np.sort(impacts)[::-1][:num_exchanges][-1]

    exchanges_dict_selected = {}
    num_selected = 0
    for exchange_type, exc_list in exchanges_dict.items():
        exc_list_selected = [
            exc for exc in exc_list if exc[where_impact] >= impacts_threshold
        ]
        exchanges_dict_selected[exchange_type] = exc_list_selected
        num_selected += len(exc_list_selected)
    print("Impact threshold {:5.3f}".format(impacts_threshold))
    print("{:d} exchanges selected".format(num_selected))
    return exchanges_dict_selected
