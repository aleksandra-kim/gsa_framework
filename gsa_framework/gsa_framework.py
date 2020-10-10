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
import time
import multiprocessing
import h5py

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    cpus : int
        Number of cpus to use for parallel computations.
    available_memory : float
        Available RAM in GB for storing arrays in variables.

    Raises
    ------
        Errors?

    """

    def __init__(
        self,
        sampler,
        model,
        interpreter,
        write_dir,
        iterations=None,
        seed=None,
        X=None,
        cpus=None,
        available_memory=2,
        use_parallel=True,
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
        # For parallel computations
        ### 1. Chunk sizes limited by available memory
        self.available_memory = available_memory  # GB
        self.bytes_per_entry = 8
        self.chunk_size_memory = min(
            int(
                self.available_memory
                * 1024 ** 3
                / self.bytes_per_entry
                / self.num_params
            ),
            self.iterations,
        )
        self.num_chunks_memory = int(np.ceil(self.iterations / self.chunk_size_memory))
        ### 2. Divide chunks above between available cpus to speed up computations
        self.cpus = min(
            cpus or multiprocessing.cpu_count(), multiprocessing.cpu_count()
        )
        self.use_parallel = use_parallel
        self.num_jobs = self.cpus
        self.chunk_size_per_worker = int(
            np.ceil(self.chunk_size_memory / self.num_jobs)
        )
        # Save some useful info in a GSA dictionary
        self.gsa_dict = {
            "iterations": self.iterations,
            "num_params": self.num_params,
            "write_dir": self.write_dir,
            "cpus": self.cpus,
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
        self.gsa_dict.update({"filename_y": self.run()})
        # Compute GSA indices
        self.gsa_dict.update({"filename_sa_results": self.interpret()})

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["arrays", "gsa_results", "figures", "computation_time"]
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def guess_iterations(self, iterations, CONSTANT=1):
        """Function that computes number of Monte Carlo iterations depending on the GSA method.

        Returns
        -------
        num_iterations : int
            Number of iterations that should be sufficient for the convergence of GSA indices. For many GSA methods
            this number depends on the desired confidence level and confidence interval width.

        """

        # if self.interpreter_str == "correlation_coefficients":
        #     corrcoef_constants = get_corrcoef_num_iterations()
        #     computed_iterations = max(
        #         corrcoef_constants["pearson"]["num_iterations"]*50, #TODO remove 5
        #         corrcoef_constants["spearman"]["num_iterations"]*50,
        #     )
        #     return computed_iterations
        if self.interpreter_str == "eFAST_indices":
            M = 4
            computed_iterations = max(
                (4 * M ** 2 + 1) * self.num_params, iterations
            )  # Sample size N > 4M^2 is required. M=4 by default.
            return computed_iterations
        else:
            if iterations:
                return iterations
            else:
                return self.num_params * CONSTANT

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

    def run_parallel(self):
        """Obtain ``model`` outputs from the ``X_rescaled`` in parallel and write them to a file."""

        results_all = np.array([])
        for i in range(self.num_chunks_memory):
            with h5py.File(self.filename_X_rescaled, "r") as f:
                start = i * self.chunk_size_memory
                end = (i + 1) * self.chunk_size_memory
                X_rescaled = np.array(f["dataset"][start:end, :])
                with multiprocessing.Pool(processes=self.cpus) as pool:
                    results = pool.map(
                        self.model,
                        [
                            X_rescaled[
                                j
                                * self.chunk_size_per_worker : (j + 1)
                                * self.chunk_size_per_worker
                            ]
                            for j in range(self.num_jobs)
                        ],
                    )
            results_array = np.array([])
            for res in results:
                results_array = np.hstack([results_array, res])
            results_all = np.hstack([results_all, results_array])
        write_hdf5_array(results_all, self.filename_y)

    def run_sequential(self):
        """Obtain ``model`` outputs from the ``X_rescaled`` sequentially and write them to a file."""

        X_rescaled = read_hdf5_array(self.filename_X_rescaled)
        y = self.model(X_rescaled)
        write_hdf5_array(y, self.filename_y)

    def run(self):
        """Wrapper function to obtain ``model`` outputs from the ``X_rescaled`` parameter sampling matrix.

        Run Monte Carlo simulations in parallel or sequentially, and write results to a file.

        Returns
        -------
        filename_y : str
            Path where model outputs ``y`` are stored.

        """

        self.filename_y = self.filename_X.parent / Path(
            "y" + self.filename_X.stem[1:] + ".hdf5"
        )
        if not self.filename_y.exists():
            t0 = time.time()
            if self.use_parallel:
                self.run_parallel()
                t1 = time.time()
                print("run_parallel time: " + str(t1 - t0) + " seconds")
            else:
                self.run_sequential()
                t1 = time.time()
                print("run_sequential time: " + str(t1 - t0) + " seconds")
        return self.filename_y

    def interpret(self):
        """Computation of GSA indices.

        Returns
        -------
        gsa_indices_dict : dict
            Keys are GSA indices names, values - sensitivity indices for all parameters.

        """

        self.filename_gsa_results = (
            self.write_dir
            / "gsa_results"
            / Path(self.interpreter_str + self.filename_X.stem[1:] + ".pickle")
        )
        if not self.filename_gsa_results.exists():
            t0 = time.time()
            gsa_indices_dict = self.interpreter_fnc(self.gsa_dict)
            t1 = time.time()
            print("GSA time: " + str(t1 - t0) + " seconds")
            with open(self.filename_gsa_results, "wb") as f:
                pickle.dump(gsa_indices_dict, f)
        return self.filename_gsa_results

    def convergence(self, step, iterations_order):
        y = read_hdf5_array(self.filename_y).flatten()
        sa_convergence_dict_temp = {}
        iterations_blocks = np.arange(step, len(y) + step, step)
        for block_size in iterations_blocks:
            selected_iterations = iterations_order[0:block_size]
            t0 = time.time()
            gsa_indices_dict = self.interpreter_fnc(self.gsa_dict, selected_iterations)
            t1 = time.time()
            print("{0:8d} iterations -> {1:8.3f} s".format(block_size, t1 - t0))
            sa_convergence_dict_temp[block_size] = gsa_indices_dict
        # Put all blocks together
        sa_convergence_dict = {
            key: np.zeros(shape=(0, self.num_params))
            for key in sa_convergence_dict_temp[block_size].keys()
        }
        for sa_dict in sa_convergence_dict_temp.values():
            for key, sa_array in sa_convergence_dict.items():
                new_sa_array = np.vstack([sa_array, sa_dict[key]])
                sa_convergence_dict.update({key: new_sa_array})

        return sa_convergence_dict, iterations_blocks

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

    def plot_convergence(
        self, sa_convergence_dict, iterations_blocks, parameter_inds=None
    ):
        if parameter_inds is None:
            parameter_inds = np.random.randint(
                0, self.num_params, max(10, self.num_params // 10)
            )
        # Assign color to each parameter
        colors = {}
        for parameter in parameter_inds:
            colors[parameter] = "rgb({0},{1},{2})".format(
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )
        # Plot
        fig = make_subplots(
            rows=len(sa_convergence_dict),
            cols=1,
            subplot_titles=list(sa_convergence_dict.keys()),
        )
        for parameter in parameter_inds:
            row = 1
            for sa_index_name, sa_array in sa_convergence_dict.items():
                showlegend = False
                if row == 1:
                    showlegend = True
                fig.add_trace(
                    go.Scatter(
                        x=iterations_blocks,
                        y=sa_array[:, parameter],
                        mode="lines+markers",
                        showlegend=showlegend,
                        marker_color=colors[parameter],
                        name="Parameter " + str(parameter),
                        legendgroup=parameter,
                    ),
                    row=row,
                    col=1,
                )
                row += 1
        fig.show()
        # Save figure
        pathname = (
            self.write_dir
            / "figures"
            / Path("convergence_" + self.filename_gsa_results.stem)
        )
        fig.write_image(pathname.with_suffix(".pdf").as_posix())
        fig.write_html(pathname.with_suffix(".html").as_posix())
