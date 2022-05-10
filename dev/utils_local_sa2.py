import numpy as np
from time import time
from pathlib import Path
from fs.zipfs import ZipFS
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp

DATA_DIR = Path(__file__).parent.resolve() / "data"


def filter_biosphere_exchanges(lca, cutoff=0.005):
    """Reduce biosphere exchanges to those that matter for a given impact
    category in a non-stochastic LCA."""
    start = time()
    inv = lca.characterized_inventory
    finv = inv.multiply(abs(inv) > abs(lca.score * cutoff))
    biosphere_exchanges = list(zip(*finv.nonzero()))
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
    return biosphere_exchanges


def filter_uncertain_biosphere_exchanges(lca, cutoff=0.005):

    uncertain_biosphere_exchanges = biosphere_exchanges
    return uncertain_biosphere_exchanges


def setup_bw_project_archetypes(project="GSA for archetypes"):
    bd.projects.set_current(project)
    co = bd.Database("swiss consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA(demand, uncertain_method, use_distributions=False)
    lca.lci()
    lca.lcia()
    return lca


def generate_local_sa_biosphere_datapackage(cutoff=1e-4, const_factor=10):

    lca = setup_bw_project_archetypes()
    uncertain_biosphere_exchanges = filter_uncertain_biosphere_exchanges(lca, cutoff)

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "local-sa-biosphere.zip"), write=True),
        name="local sa biosphere",
    )

    amounts = np.array([exc.amount for exc in uncertain_biosphere_exchanges])
    num_samples = len(amounts)
    data_array = np.tile(amounts, num_samples) * (
        np.diag(np.ones(num_samples) * const_factor)
    )

    indices_array = np.array(
        [(exc.input.id, exc.output.id) for exc in uncertain_biosphere_exchanges],
        dtype=bwp.INDICES_DTYPE,
    )

    # All inputs -> all True
    flip_array = np.ones(len(indices_array), dtype=bool)

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="local sa biosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    dp.finalize_serialization()
