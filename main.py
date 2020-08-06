from gsa_framework import *
from test_functions import *
from path_setup import *

# 1. LCA model
import brightway2 as bw
bw.projects.set_current('GSA for paper')
co = bw.Database('CH consumption 1.0')
act = co.search('average consumption')[0]
demand = {act: 1}
method = ('IPCC 2013', 'climate change', 'GTP 100a')

write_dir_lca = os.path.join(path_base,'lca_model')
model_lca = LCAModel(demand, method, write_dir_lca)
const_lca = 2
iterations_lca = model_lca.num_params * const_lca

# 2. Moon model
write_dir_moon = os.path.join(path_base,'moon_model')
model_moon = Moon()
const_moon = 10000
iterations_moon = model_moon.num_params * const_moon

# 3. Morris model
write_dir_morris = os.path.join(path_base,'morris_model')
model_morris = Morris(num_params=100, num_influential=50)
const_morris = 10000
iterations_morris = model_morris.num_params * const_morris

# 4. Problem
choose_model = 'moon'

if choose_model == 'lca':
    model = model_lca
    write_dir = write_dir_lca
    iterations = iterations_lca
    const = const_lca
elif choose_model == 'moon':
    model = model_moon
    write_dir = write_dir_moon
    iterations = iterations_moon
    const = const_moon
elif choose_model == 'morris':
    model = model_morris
    write_dir = write_dir_morris
    iterations = iterations_morris
    const = const_morris

problem = Problem(
    sampler = 'random',
    model = model,
    interpreter = 'sobol_indices',
    write_dir = write_dir,
    iterations = iterations,
    seed = 235235,
    X = None,
)

# y = problem.gsa_dict['y']
#
# import matplotlib.pyplot as plt
# plt.hist(y)
# plt.show()
#
# plt.hist(y[:const])
# plt.show()
#
# plt.hist(y[const:const*2])
# plt.show()

sa_results = problem.gsa_dict['sa_results']
for sa_index_name in sa_results.keys():
    sa_indices = {sa_index_name: problem.gsa_dict['sa_results'][sa_index_name]}
    try:
        problem.plot_sa_results(sa_indices, model.influential_params, filename=sa_index_name+'.pdf')
    except:
        pass

