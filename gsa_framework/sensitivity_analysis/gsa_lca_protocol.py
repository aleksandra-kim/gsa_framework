from gsa_framework.utils import read_pickle, write_pickle


class GsaLcaProtocol:
    def __init__(self):
        return

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

    def get_ids_from_graph_traversal(self, cutoff=1e-16, max_calc=1e16):
        res = self.get_graph_traversal_params(cutoff, max_calc)
        indices = self.lca.technosphere_mm.input_provenance()
        return

    @staticmethod
    def create_graph_traversal_filename(cutoff, max_calc):
        return "sct.cutoff_{:1.0e}.maxcalc_{:1.0e}.pickle".format(
            cutoff,
            max_calc,
        )
