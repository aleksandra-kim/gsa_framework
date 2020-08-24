# from gsa_framework import *
# from test_functions import *
# from path_setup import *
#
# # 1. LCA model
# import brightway2 as bw
#
# bw.projects.set_current('GSA for paper')
# co = bw.Database('CH consumption 1.0')
# act = co.search('average consumption')[0]
# demand = {act: 1}
# method = ('IPCC 2013', 'climate change', 'GTP 100a')
#
# write_dir_lca = os.path.join(path_base,'lca_model')
# model_lca = LCAModel(demand, method, write_dir_lca)
# const_lca = 2
# iterations_lca = model_lca.num_params * const_lca
#
# # 2. Moon model
# write_dir_moon = os.path.join(path_base,'moon_model')
# model_moon = Moon()
# const_moon = 10000
# iterations_moon = model_moon.num_params * const_moon
#
# # 3. Morris model
# write_dir_morris = os.path.join(path_base,'morris_model')
# model_morris = Morris(num_params=100, num_influential=50)
# const_morris = 10000
# iterations_morris = model_morris.num_params * const_morris
#
# # 3. Sobol-Levitan model
# write_dir_SL = os.path.join(path_base,'sobol_levitan_model')
# model_SL = SobolLevitan()
# const_SL = 1000
# iterations_SL = model_SL.num_params * const_SL
#
# # 4. Problem
# choose_model = 'sobol_levitan'
#
# if choose_model == 'lca':
#     model = model_lca
#     write_dir = write_dir_lca
#     iterations = iterations_lca
#     const = const_lca
# elif choose_model == 'moon':
#     model = model_moon
#     write_dir = write_dir_moon
#     iterations = iterations_moon
#     const = const_moon
# elif choose_model == 'morris':
#     model = model_morris
#     write_dir = write_dir_morris
#     iterations = iterations_morris
#     const = const_morris
# elif choose_model == 'sobol_levitan':
#     model = model_SL
#     write_dir = write_dir_SL
#     iterations = iterations_SL
#     const = const_SL
#
# problem = Problem(
#     sampler = 'random',
#     model = model,
#     interpreter = 'correlation_coefficients',
#     write_dir = write_dir,
#     iterations = iterations,
#     seed = 235235,
#     X = None,
# )
#
# # y = problem.gsa_dict['y']
# # nbins = 40
# #
# # import matplotlib.pyplot as plt
# # plt.hist(y, nbins)
# # plt.show()
# #
# # plt.hist(y[:const], nbins)
# # plt.show()
# #
# # plt.hist(y[const:const*2], nbins)
# # plt.show()
#
# sa_results = problem.gsa_dict['sa_results']
# for sa_index_name in sa_results.keys():
#     sa_indices = {sa_index_name: problem.gsa_dict['sa_results'][sa_index_name]}
#     try:
#         problem.plot_sa_results(sa_indices, model.influential_params, filename=sa_index_name+'.pdf')
#     except:
#         pass
#
#
#
#
from tests.test_get_corrcoef_num_iterations import test_num_iterations
test_num_iterations()