from gsa_framework.lca import LCAModel
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.extended_FAST import eFAST
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.gradient_boosting import GradientBoosting
from gsa_framework.methods.delta_moment import DeltaMoment
from gsa_framework.validation import Validation
from gsa_framework.convergence import Convergence
from pathlib import Path
import brightway2 as bw
import time
import numpy as np
from gsa_framework.plotting import plot_histogram_Y
from gsa_framework.utils import read_hdf5_array

if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa_framework/dev/write_files/oases_gsa_gwp/"
    )
    # path_base = Path('/data/user/kim_a/paper_gsa/gsa_framework_files')

    # LCA model
    bw.projects.set_current("GSA for oases")
    co = bw.Database("CH consumption 1.0")
    demand_act = [act for act in co if "ch hh average consumption" in act["name"]][0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a")

    iterations = 2000

    lca = bw.LCA(demand, method)
    lca.lci()
    lca.lcia()
    print(lca.score)
    default_score = lca.score
