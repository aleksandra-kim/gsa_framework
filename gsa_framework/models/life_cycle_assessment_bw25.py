from pathlib import Path
import bw2calc as bc

from .model_base import ModelBase

# from .utils_graph_traversal import filter_uncertain_technosphere_exchanges
from ..utils import read_pickle, write_pickle


class LCAModel25:
    # TODO pass ModelBase to this class
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

    def get_graph_traversal_params(self, cutoff=1e-16, max_calc=1e16):
        fp_graph_traversal = self.write_dir / self.create_graph_traversal_filename(
            cutoff, max_calc
        )
        if fp_graph_traversal.exists():
            res = read_pickle(fp_graph_traversal)
        else:
            res = bc.GraphTraversal().calculate(
                self.lca, cutoff=cutoff, max_calc=max_calc
            )
            write_pickle(res, fp_graph_traversal)
        return res

    # def get_graph_traversal_uncertain_params(self, cutoff=1e-16, max_calc=1e+16):
    #     res = self.get_graph_traversal_params(cutoff=cutoff, max_calc=max_calc)
    #
    #     if fp_graph_traversal.exists():
    #         uncertain_params = read_pickle(fp_graph_traversal)
    #     else:
    #         uncertain_params = filter_uncertain_technosphere_exchanges(self.lca, cutoff=cutoff, max_calc=max_calc)
    #
    #     return uncertain_params

    @staticmethod
    def create_graph_traversal_filename(cutoff, max_calc):
        return "sct.cutoff_{:1.0e}.maxcalc_{:1.0e}.pickle".format(
            cutoff,
            max_calc,
        )
