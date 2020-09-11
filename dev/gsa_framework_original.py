# Framework for global sensitivity analysis in brightway
# We want to make easy things easy, and hard things possible.


sampler_mapping = {
    "saltelli": some_function_in_salib,
}
interpreter_mapping = {
    "sobol": some_function_in_salib,
}


class Problem:
    """Definition of a global sensitivity analysis problem.

    Can include parameterized models and/or LCA matrix calculations. Can be run locally or remotely.

    ``sampler`` can be a string, in which case the correct function is retrieved from ``sampler_mapping``. ``sampler`` functions should be able to be called with ``(number of parameters, number of iterations requested)``.

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

    def __init__(self, sampler, model, interpreter, iterations=None):
        self.sampler = sampler_mapping.get(sampler, sampler)
        self.inter
        self.model = model
        self.num_params = self.model.__num_input_params__()
        self.iterations = iterations or self.guess_iterations()

    def guess_iterations(self, CONSTANT=10):
        # Default value for now...
        return self.num_params * CONSTANT

    def generate_samples(self):
        """Use ``self.sampler`` to generate normalized samples for this problem"""
        self.samples = self.sampler(self.num_params)

    def run_locally(self):
        self.results = [
            self.model(self.model.__rescale__(vector)) for vector in self.samples
        ]

    def run_remotely(self):
        """Prepare files for remote execution.

        Dispatch could be via a cloud API, dask, multiprocessing, etc.

        This function needs to create evaluation batches, e.g. 100 Monte Carlo iterations."""
        pass

    def interpret(self):
        assert hasattr(self, "results")
        return self.interpreter(self.results)


from bw_calc import MonteCarloLCA


class LCAModel:
    """A simple LCA model which uses uncertainty in the background database."""

    def __init__(self, func_unit, data_objs):
        self.lca = MonteCarloLCA(func_unit, data_objs)
        self.lca.lci()
        self.lca.lcia()

    def __num_input_params__(self):
        return (
            len(self.lca.tech_params)
            + len(self.lca.bio_params)
            + len(self.lca.cf_params)
        )

    def __rescale__(self, vector):
        return do_stuff_here()

    def __call__(self, vector):
        do_something_by_overriding_lca_next()
        return self.lca.score


class ParameterizedLCAModel:
    """A model which does LCA of parameterized models.

    Builds on the ``presamples`` library. For each Monte Carlo iteration, the model is evaluated, and the evaluation result is saved into an ``overrides`` processed array. This array is then used in building the LCA matrices."""

    pass
