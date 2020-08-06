# Framework for global sensitivity analysis in brightway
# We want to make easy things easy, and hard things possible.

import brightway2 as bw
import numpy as np
import os
import pickle
import h5py
from copy import copy, deepcopy
import plotly.graph_objects as go
from stats_arrays import uncertainty_choices, MCRandomNumberGenerator
# Local files
from sampling.get_samples import *
from sensitivity_analysis.get_gsa_indices import *


sampler_mapping = {
    'saltelli': saltelli_samples,
    'sobol':    sobol_samples,
    'random':   random_samples,
    'custom':   custom_samples,
    'dissimilarity_samples': dissimilarity_samples
}
interpreter_mapping = {
    'correlation_coefficients': correlation_coefficients,
    'sobol_indices': sobol_indices,
    'xgboost': xgboost_scores,
    'dissimilarity_measure': dissimilarity_measure,
}


class Problem:
    """Definition of a global sensitivity analysis problem.

    Can include parameterized models and/or LCA matrix calculations. Can be run locally or remotely.

    ``sampler`` can be a string, in which case the correct function is retrieved from ``sampler_mapping``. 
    ``sampler`` functions should be able to be called with ``(number of parameters, number of iterations requested)``.

    **to be discussed**, could require that sampler is a callable. Strings are nicer, especially for beginners.

    Models must support the ``Problem`` API. They must expose the following methods:

        * __num_input_params__(): Returns number of independent input parameters.
        * __rescale__(vector): Rescales a vector of ``[0-1]`` normalized input parameters to model scale.
        * __call__(vector): Return a model result. Should be an numpy array with one or more elements. ``__call__`` is the generic Python method for callable objects (e.g. functions).

    Args:
        * sampler (str or callable): Generate normalized samples for the correct number of parameters.
        * model (object): Model which supports the ``Problem`` API.
        * interpreter (str or callable): Consume ``self.results`` and generate sensitivity indices.
        * iterations (int, default=number of factors): Number of Monte Carlo iterations to calculate.

    Raises:
        Errors?

    """
    def __init__(self, sampler, model, interpreter, write_dir, iterations=None, seed=None, X=None):

        # General
        self.seed = seed
        self.model = model
        self.num_params = self.model.__num_input_params__()
        self.iterations = iterations or self.guess_iterations()
        self.write_dir = write_dir

        self.gsa_dict = {
            'iterations': self.iterations,
            'num_params': self.num_params,
            'write_dir': self.write_dir,
        }

        # Make necessary directories
        self.make_dirs()

        # Sampling strategy depends on the interpreter
        self.interpreter_str = interpreter
        self.interpreter_fnc = interpreter_mapping.get(self.interpreter_str, 'correlation_coefficients')
        self.sampler_str = sampler

        # Generate samples
        self.gsa_dict.update({
            'sampler_str': self.sampler_str,
            'X': X,
        })
        self.gsa_dict.update({'X': self.generate_samples()})
        self.gsa_dict.update({'X_rescaled': self.rescale_samples()})
        # Run model
        self.gsa_dict.update({'y': self.run_locally()})
        # Compute GSA indices
        self.gsa_dict.update({'sa_results': self.interpret()})

    def make_dirs(self):
        dirs_list = [
            'arrays'
        ]
        for dir in dirs_list:
            dir_path = os.path.join(self.write_dir, dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def guess_iterations(self, CONSTANT=10):
        # Default value for now...
        return self.num_params * CONSTANT

    def generate_samples(self, X=None):
        """Use ``self.sampler`` to generate normalized samples for this problem"""
        self.base_sampler_fnc = 'no_base'
        if self.interpreter_str == 'sobol_indices':
            print('Changing samples to saltelli, because of faster indices convergence')
            self.sampler_str = 'saltelli'
            self.seed = None
        elif self.interpreter_str == 'dissimilarity_measure':
            print('Samples should be adapted for dissimilarity sensitivity measure')
            self.base_sampler_fnc = sampler_mapping.get(self.sampler_str, 'random')
            self.sampler_str = 'dissimilarity_samples'
            self.gsa_dict.update({'base_sampler_fnc': self.base_sampler_fnc})
        else:
            if X != None:
                self.sampler_str = 'custom'
                self.seed = None
        self.sampler_fnc = sampler_mapping.get(self.sampler_str, 'random')
        self.gsa_dict.update({'sampler_fnc': self.sampler_fnc})
        self.gsa_dict.update({'seed': self.seed})

        self.filename_X = os.path.join(
            self.write_dir,
            'arrays',
            'X_' + self.sampler_str + '_' + self.base_sampler_fnc + \
            '_iterations_' + str(self.iterations) + \
            '_num_params_' + str(self.num_params) + \
            '_seed_' + str(self.seed) + '.hdf5',
        )
        if not os.path.exists(self.filename_X):
            X = self.sampler_fnc(self.gsa_dict)
            self.write_hdf5_array(X, self.filename_X)

        return self.filename_X

    def rescale_samples(self):
        path_start = os.path.split(self.filename_X)[0]
        path_end = os.path.split(self.filename_X)[-1]
        self.filename_X_rescaled = os.path.join(path_start, 'X_rescaled' + path_end[1:])
        if not os.path.exists(self.filename_X_rescaled):
            X = self.read_hdf5_array(self.filename_X)
            X_rescaled = self.model.__rescale__(X)
            print(X.shape, X_rescaled.shape)
            self.write_hdf5_array(X_rescaled, self.filename_X_rescaled)
        return self.filename_X_rescaled


    def write_hdf5_array(self, array, filename):
        try:
            n_rows, n_cols = array.shape[0], array.shape[1]
        except IndexError:
            n_rows, n_cols = 1, array.shape[0]

        with h5py.File(filename, 'w') as f:
            d = f.create_dataset('dataset',
                                 (n_rows, n_cols),
                                 maxshape=(n_rows, n_cols),
                                 dtype=array.dtype
                                 )
            d[:] = array


    def read_hdf5_array(self, filename):
        with h5py.File(filename, 'r') as f:
            X = np.array(f['dataset'][:])
            return X


    def run_locally(self):
        path_start = os.path.split(self.filename_X)[0]
        path_end = os.path.split(self.filename_X)[-1]
        self.filename_y = os.path.join(path_start, 'y' + path_end[1:])
        if not os.path.exists(self.filename_y):
            X_rescaled = self.read_hdf5_array(self.filename_X_rescaled)
            y = self.model(X_rescaled)
            self.write_hdf5_array(y, self.filename_y)
        return self.filename_y
    #
    # def run_remotely(self):
    #     """Prepare files for remote execution.
    #
    #     Dispatch could be via a cloud API, dask, multiprocessing, etc.
    #
    #     This function needs to create evaluation batches, e.g. 100 Monte Carlo iterations."""
    #     pass
    #

    def interpret(self):
        y = self.read_hdf5_array(self.filename_y)
        print(y.shape)
        self.gsa_dict.update({'y': y.flatten()})
        return self.interpreter_fnc(self.gsa_dict)


    def plot_sa_results(self, sa_indices, influential_inputs=[], filename=''):
        index_name = list(sa_indices.keys())[0]
        index_vals = list(sa_indices.values())[0]

        sa_indices_influential = np.array([index_vals[f] for f in influential_inputs])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x = np.arange(len(index_vals)),
                y = index_vals,
                mode = 'markers',
                marker = dict(
                    color = 'blue'
                ),
                name='All parameters',
            ),
        )
        if len(influential_inputs)>0:
            fig.add_trace(
                go.Scatter(
                    x = influential_inputs,
                    y = sa_indices_influential,
                    mode = 'markers',
                    marker = dict(
                        color = 'red'
                    ),
                name = 'Known influential parameters',
                ),
            )
        fig.update_layout(
            xaxis_title = "Model parameters",
            yaxis_title = index_name,
        )
        if not filename:
            filename = 'sensitivity_plot.pdf'
        pathname = os.path.join(self.write_dir, filename)
        fig.write_image(pathname)


class LCAModel:
    """A simple LCA model which uses uncertainty in the background database."""
    def __init__(self, func_unit, method, write_dir):
        self.lca = bw.LCA(func_unit, method)
        self.lca.lci()
        self.lca.lcia()

        self.write_dir = write_dir
        self.make_dirs()

        # self.uncertain_tech_params_where = np.where(self.lca.tech_params['uncertainty_type'] > 1)[0]
        # self.uncertain_tech_params = self.lca.tech_params[self.uncertain_tech_params_where]

        self.uncertain_tech_params_where = self.get_LSA_params(var_threshold=100000)
        self.uncertain_tech_params = self.lca.tech_params[self.uncertain_tech_params_where]

        self.num_params = self.__num_input_params__()

        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)


    def make_dirs(self):
        directories = {
            'LSA_scores': os.path.join(self.write_dir, 'LSA_scores')
        }
        # dir_reg = os.path.join(self.write_dir, 'regression')
        for dir in directories.values():
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.directories = directories


    def get_lsa_scores_pickle(self, path):
        filepath_all =  os.path.join(path, 'LSA_scores.pickle')
        if os.path.isfile(filepath_all):
            with open(filepath_all, 'rb') as f:
                scores_ = pickle.load(f)
            scores = {int(k): v for k, v in scores_.items()}
        else:
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                     and 'LSA_scores_' in f]
            starts = [int(f.split('_')[2]) for f in files]
            ind_sort = np.argsort(starts)

            files_sorted = [files[i] for i in ind_sort]

            scores = {}
            for file in files_sorted:
                filepath = os.path.join(path, file)
                with open(filepath, 'rb') as f:
                    temp = pickle.load(f)
                temp_int = {int(k): v['scores'] for k, v in temp.items()}
                scores.update(temp_int)

        return scores


    def get_nonzero_params(self, scores_dict, var_threshold):

        keys = np.array(list(scores_dict.keys()))
        vals = np.array(list(scores_dict.values()))

        # Variance of LSA scores for each input / parameter
        var = np.var(vals, axis=1)
        where = np.where(var > var_threshold)[0]

        params_yes = keys[where]
        params_no = np.setdiff1d(keys, params_yes)
        params_yes.sort(), params_no.sort()

        return params_no, params_yes


    def get_LSA_params(self, var_threshold):
        params_yes_filename = os.path.join(self.directories['LSA_scores'],
                                           'params_yes_' + str(var_threshold) + '.pickle')
        if not os.path.exists(params_yes_filename):
            scores_dict = self.get_lsa_scores_pickle(self.directories['LSA_scores'])
            _, params_yes = self.get_nonzero_params(scores_dict, var_threshold=var_threshold)
            with open(params_yes_filename, 'wb') as f:
                pickle.dump(params_yes, f)
        else:
            with open(params_yes_filename, 'rb') as f:
                params_yes = pickle.load(f)

        return params_yes


    def __num_input_params__(self):
        # self.uncertain_bio_params  = self.lca.bio_params[self.lca.bio_params['uncertainty_type'] > 1]
        # self.uncertain_cf_params   = self.lca.cf_params[self.lca.cf_params['uncertainty_type'] > 1]
        return len(self.uncertain_tech_params) # + len(self.uncertain_bio_params) + len(self.lca.cf_params)

    def __rescale__(self, X):
        iterations, num_params = X.shape[0], X.shape[1]
        assert num_params == self.uncertain_tech_params.shape[0]

        X_reordered = X[:,self.mc.ordering]

        X_rescaled = np.zeros((iterations, num_params))
        X_rescaled[:] = np.nan

        offset = 0
        for uncertainty_type in self.choices:
            num_uncertain_params = self.mc.positions[uncertainty_type]
            if not num_uncertain_params:
                continue
            random_data = uncertainty_type.ppf(
                params = self.mc.params[offset:num_uncertain_params+offset],
                percentages = X_reordered[:, offset:num_uncertain_params+offset].T
            )
            X_rescaled[:,offset:num_uncertain_params+offset] = random_data.T
            offset += num_uncertain_params

        X_rescaled = X_rescaled[:,np.argsort(self.mc.ordering)]

        return X_rescaled

    def __call__(self, X):
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i,x in enumerate(X):
            amounts = deepcopy(self.lca.tech_params['amount'])
            amounts[self.uncertain_tech_params_where] = x
            self.lca.rebuild_technosphere_matrix(amounts)
            self.lca.redo_lci()
            self.lca.redo_lcia()
            scores[i] = self.lca.score

        return scores


# class ParameterizedLCAModel:
#     """A model which does LCA of parameterized models.
#
#     Builds on the ``presamples`` library. For each Monte Carlo iteration, the model is evaluated, and the evaluation result is saved into an ``overrides`` processed array. This array is then used in building the LCA matrices."""
#     pass
