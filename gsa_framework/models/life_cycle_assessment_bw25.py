from pathlib import Path
import bw2calc as bc

from .model_base import ModelBase

# from .utils_graph_traversal import filter_uncertain_technosphere_exchanges
from ..utils import read_pickle, write_pickle


class LCAModel25(ModelBase):
    def __init__(
        self,
        demand,
        method,
        write_dir,
    ):
        self.demand = demand
        self.method = method
        self.lca = bc.LCA(demand, method, use_distributions=True)
        self.lca.lci()
        self.lca.lcia()
        self.write_dir = Path(write_dir)

    def __len__(self):
        return

    def rescale(self, X):
        X_rescaled_all = X
        return X_rescaled_all

    def __call__(self, X):
        scores = X
        return scores
