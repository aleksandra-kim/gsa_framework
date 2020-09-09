# import brightway2 as bw
#
# from path_setup import *
# from gsa_framework import *
#
# # Models
# from test_functions import *
# from lca_models import *
#
# # 1. LCA model
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
# const_moon = model_moon.num_params
# iterations_moon = model_moon.num_params * const_moon
#
# # 3. Morris model
# write_dir_morris = os.path.join(path_base,'morris_model')
# model_morris = Morris(num_params=50, num_influential=5)
# const_morris = model_morris.num_params
# iterations_morris = model_morris.num_params * const_morris
#
# # 3. Sobol-Levitan model
# write_dir_SL = os.path.join(path_base,'sobol_levitan_model')
# model_SL = SobolLevitan()
# const_SL = model_SL.num_params
# iterations_SL = model_SL.num_params * const_SL
#
# # 4. Problem
# models_dict = {
#     'lca': dict(
#         model= model_lca,
#         write_dir = write_dir_lca,
#         iterations = iterations_lca,
#         const = const_lca,
#     ),
#     'moon': dict(
#         model = model_moon,
#         write_dir = write_dir_moon,
#         iterations = iterations_moon,
#         const = const_moon,
#     ),
#     'morris': dict(
#         model = model_morris,
#         write_dir = write_dir_morris,
#         iterations = iterations_morris,
#         const = const_morris,
#     ),
#     'sobol_levitan': dict(
#         model = model_SL,
#         write_dir = write_dir_SL,
#         iterations = iterations_SL,
#         const = const_SL,
#     )
# }
#
# gsa_methods = [
#     'correlation_coefficients',
#     'sobol_indices',
#     'eFAST_indices',
#     'xgboost',
#     # 'dissimilarity_measure',
# ]
#
# model_names = [
#     'lca',
#     'morris',
#     'sobol_levitan',
# ]
#
# for model_name in model_names:
#     print('>>> ' + model_name)
#     model = models_dict[model_name]['model']
#     for gsa_method in gsa_methods:
#         print('--- ' + gsa_method)
#         problem = Problem(
#             sampler = 'random',
#             model = model,
#             interpreter = gsa_method,
#             write_dir = models_dict[model_name]['write_dir'],
#             iterations = models_dict[model_name]['iterations'],
#             seed = 34534,
#             X = None,
#         )
#         sa_results = problem.gsa_dict['sa_results']
#         for sa_index_name, sa_index_values in sa_results.items():
#             sa_indices = {sa_index_name: sa_index_values}
#             try:
#                 problem.plot_sa_results(sa_indices, model.influential_params, filename=sa_index_name+'.pdf')
#             except:
#                 pass



from tests.test_sobol_sequence import test_output
from tests.test_get_corrcoef_num_iterations import test_num_iterations
test_output()
# test_num_iterations()