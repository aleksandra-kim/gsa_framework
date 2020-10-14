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


import numpy as np


class SensitivityAnalysisMethod:
    """Base class to define sensitivity analysis methods. Should be subclassed.

    Sensitivity analysis methods should have the following components:

    * A sampling strategy. Can be completely random, or structured, or whatever.
    * A model execution step.
    * An analysis function to calculate various indices.

    This class provides a common interface for these components, and utility functions to save data at each step.

    """
    def __init__(
        self,
        write_dir,
        iterations=None,
        seed=None,
        cpus=None,
        available_memory=2,
        use_parallel=True,
    ):
        self.write_dir = Path(write_dir)

        # This should be done by the Problem class
        # self.make_dirs()

        self.seed = seed
        self.num_params = len(self.model)
        self.iterations = iterations or self.num_params

        # For parallel computations
        ### 1. Chunk sizes limited by available memory
        self.available_memory = available_memory  # GB

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
            # There has to be a way to make this more elegant
            cpus or multiprocessing.cpu_count(), multiprocessing.cpu_count()
        )
        self.use_parallel = use_parallel
        self.num_jobs = self.cpus
        self.chunk_size_per_worker = int(
            np.ceil(self.chunk_size_memory / self.num_jobs)
        )
        # Save some useful info in a GSA dictionary
        # self.gsa_dict = {
        #     "iterations": self.iterations,
        #     "num_params": self.num_params,
        #     "write_dir": self.write_dir,
        #     "cpus": self.cpus,
        # }
        # # Generate samples
        # self.gsa_dict.update(
        #     {
        #         "sampler_str": self.sampler_str,
        #         "X": X,
        #     }
        # )
        # self.gsa_dict.update({"filename_X": self.generate_samples()})
        # self.gsa_dict.update({"filename_X_rescaled": self.rescale_samples()})
        # # Run model
        # self.gsa_dict.update({"filename_y": self.run()})
        # # Compute GSA indices
        # self.gsa_dict.update({"filename_sa_results": self.interpret()})

    def create_normalized_samples_filename(self):
        return "X.{}.{}.{}.{}.hdf5".format(self.label, self.iterations, self.num_params, self.seed)

    def create_rescaled_samples_filename(self):
        # Maybe we need to be more careful here, as this will change according to the model
        return "X.rescaled.{}.{}.{}.{}.hdf5".format(self.label, self.iterations, self.num_params, self.seed)

    @property
    def filepath_X(self):
        return (
            self.write_dir
            / "arrays"
            / self.create_normalized_samples_filename()
        )

    @property
    def filepath_X_rescaled(self):
        return (
            self.write_dir
            / "arrays"
            / self.create_rescaled_samples_filename()
        )

    def generate_normalized_samples(self, X=None):
        raise NotImplementedError

    def rescale_samples(self):
        """Rescale samples from standard uniform to appropriate distributions and write ``X_rescaled`` to a file."""
        if not self.filepath_X_rescaled.exists():
            X = read_hdf5_array(self.filename_X)
            X_rescaled = self.model.rescale(X)
            write_hdf5_array(X_rescaled, self.filename_X_rescaled)
        return X_rescaled

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
