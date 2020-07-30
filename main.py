from gsa_framework import *
from test_functions import *


interpreter = 'correlation_coefficients'
num_params = 50
iterations = 5
seed = 10
write_dir = 'write_files'

import brightway2 as bw
bw.projects.set_current('GSA for paper')
co = bw.Database('CH consumption 1.0')
act = co.search('average consumption')[0]
demand = {act: 1}
method = ('IPCC 2013', 'climate change', 'GTP 100a')
model = LCAModel(demand, method)

test = Problem(
    sampler = 'random',
    model = model,
    interpreter = interpreter,
    write_dir = write_dir,
    iterations = iterations,
    seed = seed,
    X = None,
)

sa_indices = {"spearman": abs(test.sa_dict['spearman'])}
test.plot_sa_results(sa_indices, model.influential_params)



# lca = bw.LCA(demand, method)
# lca.lci()
# lca.lcia()
#
# tech_params = lca.tech_params
# from stats_arrays import MCRandomNumberGenerator
# mc = MCRandomNumberGenerator(lca.tech_params)