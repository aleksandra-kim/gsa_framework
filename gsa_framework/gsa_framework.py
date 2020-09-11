from .sampling.get_samples import *
from .sensitivity_analysis.correlation_coefficients import (
    correlation_coefficients,
    get_corrcoef_num_iterations,
)
from .sensitivity_analysis.dissimilarity_measures import dissimilarity_measure
from .sensitivity_analysis.extended_FAST import eFAST_indices
from .sensitivity_analysis.gradient_boosting import xgboost_scores
from .sensitivity_analysis.sobol_indices import sobol_indices
from .utils import read_hdf5_array, write_hdf5_array
from pathlib import Path
import pickle, json
import plotly.graph_objects as go
import time

# Sampler and Global Sensitivity Analysis (GSA) mapping dictionaries
sampler_mapping = {
    "saltelli": saltelli_samples,
    "sobol": sobol_samples,
    "eFAST": eFAST_samples,
    "random": random_samples,
    "custom": custom_samples,
    "dissimilarity_samples": dissimilarity_samples,
}
interpreter_mapping = {
    "correlation_coefficients": correlation_coefficients,
    "sobol_indices": sobol_indices,
    "eFAST_indices": eFAST_indices,
    "xgboost": xgboost_scores,
    "dissimilarity_measure": dissimilarity_measure,
}


class Problem:
    """Definition of a global sensitivity analysis problem.

    Parameters
    ----------
    sampler : str
        The correct function for generating samples of specific type is retrieved from ``sampler_mapping``.
    model : class
        Can include any model that contain methods:
            ``__num_input_params__`` that outputs number of model input parameters;
            ``__rescale__(X)``       that rescales samples from standard uniform to appropriate distributions;
            ``__call__(X_rescaled)`` that computes model outputs for all samples in ``X_rescaled``.
        Models can be run locally or remotely (TODO).
    interpreter : str
        The correct function for GSA is retrieved from ``interpreter_mapping``.
    write_dir : str
        Directory where intermediate results and plots will be stored.
    iterations : int
        Number of Monte Carlo iterations.
    seed : int
        Random seed.
    X : np.array of size [iterations, num_params]
        Custom parameter sampling matrix in standard uniform [0,1] range.

    Raises
    ------
        Errors?

    """

    def __init__(
        self, sampler, model, interpreter, write_dir, iterations=None, seed=None, X=None
    ):
        # Create necessary directories
        self.write_dir = Path(write_dir)
        self.make_dirs()
        # General
        self.seed = seed
        self.model = model
        self.num_params = self.model.__num_input_params__()
        # Sampling strategy and ``iterations`` depend on the interpreter
        self.interpreter_str = interpreter
        self.interpreter_fnc = interpreter_mapping.get(self.interpreter_str)
        self.sampler_str = sampler
        # Iterations
        self.iterations = self.guess_iterations(iterations)
        # Save some useful info in a GSA dictionary
        self.gsa_dict = {
            "iterations": self.iterations,
            "num_params": self.num_params,
            "write_dir": self.write_dir,
        }
        # Generate samples
        self.gsa_dict.update(
            {
                "sampler_str": self.sampler_str,
                "X": X,
            }
        )
        self.gsa_dict.update({"filename_X": self.generate_samples()})
        self.gsa_dict.update({"filename_X_rescaled": self.rescale_samples()})
        # Run model
        self.gsa_dict.update({"filename_y": self.run_locally()})
        t0 = time.time()
        # Compute GSA indices
        self.gsa_dict.update({"sa_results": self.interpret()})
        t1 = time.time()
        self.save_time(t1 - t0)

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["arrays", "gsa_results", "figures", "computation_time"]
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def guess_iterations(self, iterations, CONSTANT=10):
        """Function that computes number of Monte Carlo iterations depending on the GSA method.

        Returns
        -------
        num_iterations : int
            Number of iterations that should be sufficient for the convergence of GSA indices. For many GSA methods
            this number depends on the desired confidence level and confidence interval width.

        """

        if self.interpreter_str == "correlation_coefficients":
            corrcoef_constants = get_corrcoef_num_iterations()
            computed_iterations = max(
                corrcoef_constants["pearson"]["num_iterations"],
                corrcoef_constants["spearman"]["num_iterations"],
                corrcoef_constants["kendall"]["num_iterations"],
            )
        elif self.interpreter_str == "eFAST_indices":
            M = 4
            computed_iterations = (
                4 * M ** 2 + 1
            )  # Sample size N > 4M^2 is required. M=4 by default.
        else:
            computed_iterations = self.num_params * CONSTANT

        if iterations:
            return max(computed_iterations, iterations)
        else:
            return computed_iterations

    def generate_samples(self, X=None):
        """Use ``self.sampler`` to generate normalized samples for this problem.

        Returns
        -------
        filename_X : str
            Path where parameter sampling matrix ``X`` for standard uniform samples is stored.

        TODO Chris, this function is horrible.

        """

        self.base_sampler_str = "no_base"
        if self.interpreter_str == "sobol_indices":
            # Printing is OK, but not great on clusters. Consider using warnings and/or proper logging
            #             print('Changing samples to saltelli, because indices convergence faster')

            # 1) you don't know if this is actually a change, and
            # 2) if this change is always happening, you could skip the warning message

            self.sampler_str = "saltelli"
            self.seed = None
        elif self.interpreter_str == "eFAST_indices":
            #             print('Changing samples to eFAST, because indices convergence faster')
            self.sampler_str = "eFAST"
        elif self.interpreter_str == "dissimilarity_measure":
            #             print('Samples should be adapted for dissimilarity sensitivity measure')
            self.base_sampler_str = sampler_mapping.get(self.sampler_str, "random")
            self.base_sampler_fnc = sampler_mapping.get(self.base_sampler_str)
            self.sampler_str = "dissimilarity_samples"
            self.gsa_dict.update(
                {
                    "base_sampler_str": self.base_sampler_str,
                    "base_sampler_fnc": self.base_sampler_fnc,
                }
            )
        else:
            if X != None:
                self.sampler_str = "custom"
                self.seed = None
        self.sampler_fnc = sampler_mapping.get(self.sampler_str, "random")
        self.gsa_dict.update({"sampler_fnc": self.sampler_fnc})
        self.gsa_dict.update({"seed": self.seed})

        self.filename_X = (
            self.write_dir
            / "arrays"
            / Path(
                "X_"
                + self.sampler_str
                + "_"
                + self.base_sampler_str
                + "_iterations_"
                + str(self.iterations)
                + "_num_params_"
                + str(self.num_params)
                + "_seed_"
                + str(self.seed)
                + ".hdf5"
            )
        )

        if not self.filename_X.exists():
            X = self.sampler_fnc(self.gsa_dict)
            write_hdf5_array(X, self.filename_X)

        # I don't like this changing global state, and then returning something as well.
        # This is a question of personal preference, but I would set global state on class instantiation, and then
        # change it as little as possible, just pass around the variables needed for each method.

        return self.filename_X

    def rescale_samples(self):
        """Rescale samples from standard uniform to appropriate distributions and write ``X_rescaled`` to a file.

        Returns
        -------
        filename_X_rescaled : str
            Path where parameter sampling matrix ``X_rescaled`` for samples from appropriate distributions is stored.

        """

        self.filename_X_rescaled = self.filename_X.parent / Path(
            "X_rescaled" + self.filename_X.stem[1:] + ".hdf5"
        )
        if not self.filename_X_rescaled.exists():
            X = read_hdf5_array(self.filename_X)
            X_rescaled = self.model.__rescale__(X)
            write_hdf5_array(X_rescaled, self.filename_X_rescaled)
        return self.filename_X_rescaled

    def run_locally(self):
        """Obtain ``model`` outputs from the ``X_rescaled`` parameter sampling matrix.

        Run Monte Carlo simulations and write results to a file.

        Returns
        -------
        filename_y : str
            Path where model outputs ``y`` are stored.

        """

        self.filename_y = self.filename_X.parent / Path(
            "y" + self.filename_X.stem[1:] + ".hdf5"
        )
        if not self.filename_y.exists():
            X_rescaled = read_hdf5_array(self.filename_X_rescaled)
            y = self.model(X_rescaled)
            write_hdf5_array(y, self.filename_y)
        return self.filename_y

    def run_remotely(self):
        """Prepare files for remote execution.

        Dispatch could be via a cloud API, dask, multiprocessing, etc.

        This function needs to create evaluation batches, e.g. 100 Monte Carlo iterations. TODO

        """

        pass

    def interpret(self):
        """Computation of GSA indices.

        Returns
        -------
        gsa_indices_dict : dict
            Keys are GSA indices names, values - sensitivity indices for all parameters.

        """
        #         y = read_hdf5_array(self.filename_y)
        #         X_rescaled = read_hdf5_array(self.filename_X_rescaled)
        #         self.gsa_dict.update({'y': y.flatten()})
        #         self.gsa_dict.update({'X': X_rescaled})
        gsa_indices_dict = self.interpreter_fnc(self.gsa_dict)
        self.filename_gsa_results = (
            self.write_dir
            / "gsa_results"
            / Path(self.interpreter_str + self.filename_X.stem[1:] + ".pickle")
        )

        if not self.filename_gsa_results.exists():
            with open(self.filename_gsa_results, "wb") as f:
                pickle.dump(gsa_indices_dict, f)
        return self.filename_gsa_results

    def save_time(self, elapsed_time):
        time_dict = {"time": str((elapsed_time) / 3600) + " hours"}

        filename_time = (
            self.write_dir
            / "computation_time"
            / Path("time" + self.filename_X.stem[1:] + ".json")
        )
        if not filename_time.exists():
            with open(filename_time, "w") as f:
                json.dump(time_dict, f)

    def plot_sa_results(self, sa_indices, influential_inputs=[]):
        """Simplistic plotting of GSA results of GSA indices vs parameters. Figure is saved in the ``write_dir``.

        Parameters
        ----------
        sa_indices : dict
            Keys are GSA indices names, values - sensitivity indices for all parameters.
        influential_inputs : list
            Parameters that are known to be influential, eg if the model is analytical. Ground truth for GSA validation.
        filename : str
            Filename for saving the plot, otherwise it will be saved under ``sensitivity_plot.pdf``.

        """

        index_name = list(sa_indices.keys())[0]
        index_vals = list(sa_indices.values())[0]

        sa_indices_influential = np.array([index_vals[f] for f in influential_inputs])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(index_vals)),
                y=index_vals,
                mode="markers",
                marker=dict(color="blue"),
                name="All parameters",
            ),
        )
        if len(influential_inputs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=influential_inputs,
                    y=sa_indices_influential,
                    mode="markers",
                    marker=dict(color="red"),
                    name="Known influential parameters",
                ),
            )
        fig.update_layout(
            xaxis_title="Model parameters",
            yaxis_title=index_name,
        )
        pathname = (
            self.write_dir / "figures" / Path(self.filename_gsa_results.stem + ".pdf")
        )
        fig.show()


#         fig.write_image(pathname.as_posix())
