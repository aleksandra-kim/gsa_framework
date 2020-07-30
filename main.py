from gsa_framework import *
from test_functions import *


interpreter = 'correlation_coefficients'
num_params = 50
iterations = 3
seed = 10
write_dir = 'write_files'

# 1. LCA model
import brightway2 as bw
bw.projects.set_current('GSA for paper')
co = bw.Database('CH consumption 1.0')
act = co.search('average consumption')[0]
demand = {act: 1}
method = ('IPCC 2013', 'climate change', 'GTP 100a')
write_dir = 'write_files/paper_gsa'
model = LCAModel(demand, method, write_dir)
num_params = model.__num_input_params__()
X = np.random.rand(iterations, num_params)
Xr = model.__rescale__(X)
scores = model(Xr)
print(scores)

# 2. other models
# model = Moon()
# model = Morris()

test = Problem(
    sampler = 'random',
    model = model,
    interpreter = interpreter,
    write_dir = write_dir,
    iterations = iterations,
    seed = seed,
    X = None,
)

# sa_indices = {"spearman": abs(test.sa_dict['spearman'])}
# test.plot_sa_results(sa_indices, model.influential_params)