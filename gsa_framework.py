# Framework for global sensitivity analysis in brightway
# We want to make easy things easy, and hard things possible.

import brightway2 as bw
import numpy as np
import os
from copy import copy, deepcopy
import plotly.graph_objects as go
from stats_arrays import uncertainty_choices, MCRandomNumberGenerator
# Local files
from sampling.get_samples import *
from sensitivity_analysis.get_gsa_indices import *


sampler_mapping = {
    # 'saltelli': sampling.saltelli_samples,
    'sobol':    sobol_samples,
    'random':   random_samples,
    'custom':   custom_samples,
}
interpreter_mapping = {
    'correlation_coefficients': correlation_coefficients,
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

        # 1. Sampling
        self.iterations = iterations or self.guess_iterations()
        self.sampler = sampler_mapping.get(sampler, 'random')
        self.seed = seed
        self.model = model
        self.num_params = self.model.__num_input_params__()
        self.write_dir = write_dir

        if X == None:
            np.random.seed(seed)
            self.X = np.random.rand(self.iterations,self.num_params)

        self.sampler_dict = {
            'iterations': self.iterations,
            'num_params': self.num_params,
            'X': self.X,
            'seed': self.seed,
        }

        # 2. Model runs
        self.run_locally()

        # 3. GSA
        self.interpreter = interpreter_mapping.get(interpreter, 'correlation_coefficients')
        self.results_dict = {
            'X': self.X,
            'y': self.results
        }
        self.sa_dict = self.interpret()

    def guess_iterations(self, CONSTANT=10):
        # Default value for now...
        return self.num_params * CONSTANT

    def generate_samples(self):
        """Use ``self.sampler`` to generate normalized samples for this problem"""
        self.samples = self.sampler(self.sampler_dict)

    def run_locally(self):
        self.results = self.model(self.X)
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
        assert hasattr(self, "results")
        return self.interpreter(self.results_dict)


    def plot_sa_results(self, sa_indices, influential_inputs=[]):
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
        filename = os.path.join(self.write_dir, 'write_figures', 'sensitivity_plot.pdf')
        fig.write_image(filename)




from bw2calc import MonteCarloLCA


class LCAModel:
    """A simple LCA model which uses uncertainty in the background database."""
    def __init__(self, func_unit, method):
        self.lca = bw.LCA(func_unit, method)
        self.lca.lci()
        self.lca.lcia()

        self.uncertain_tech_params_where = np.where(self.lca.tech_params['uncertainty_type'] > 1)[0]
        self.uncertain_tech_params = self.lca.tech_params[self.uncertain_tech_params_where]

        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)

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
