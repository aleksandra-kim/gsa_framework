import numpy as np
from pathlib import Path
import pickle
import time
import multiprocessing
import h5py

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import read_hdf5_array, write_hdf5_array


class SensitivityAnalysisMethod:
    """Base class to define sensitivity analysis methods. Should be subclassed.

    Sensitivity analysis methods should have the following components:

    * A sampling strategy. Can be completely random, or structured, or whatever.
    * A model execution step.
    * An analysis function to calculate various indices.

    This class provides a common interface for these components, and utility functions to save data at each step.

    TODO should filenames have the correct number of iterations or the rounded one?

    """

    sampling_label = "sampling_random"
    gsa_label = "gsa_base"

    def __init__(
        self,
        model,
        write_dir,
        iterations=None,
        seed=None,
        cpus=None,
        available_memory=2,
        bytes_per_entry=8,
        use_parallel=True,
    ):
        self.model = model
        self.write_dir = Path(write_dir)
        self.make_dirs()
        # self.label=label
        self.num_params = len(self.model)
        self.iterations = (
            iterations or self.num_params
        )  # TODO this doesn't work for correlations
        self.seed = seed
        self.cpus = min(
            # There has to be a way to make this more elegant, -> S: Set default cpus to inf?
            cpus or multiprocessing.cpu_count(),
            multiprocessing.cpu_count(),
        )
        self.available_memory = available_memory  # GB
        self.bytes_per_entry = bytes_per_entry
        self.use_parallel = use_parallel

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["arrays", "gsa_results", "figures"]
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_parallel_params(self, across_iterations):
        """Compute parameters necessary for parallel computations, eg chunk_sizes, num_chunks."""
        # 1. Chunk sizes are limited by available memory
        get_chunk_size = lambda constant_dim, dim_to_parallelize: min(
            int(
                self.available_memory * 1024 ** 3 / self.bytes_per_entry / constant_dim
            ),
            dim_to_parallelize,
        )
        if across_iterations:
            # Chunk sizes for iterations while num_params is a constant
            chunk_size = get_chunk_size(self.num_params, self.iterations)
            num_chunks = int(np.ceil(self.iterations / chunk_size))
        else:
            # Chunk sizes for num_params while iterations is a constant
            chunk_size = get_chunk_size(self.iterations, self.num_params)
            num_chunks = int(np.ceil(self.num_params / chunk_size))
        # 2. Divide chunks above between available cpus to speed up computations
        chunk_size_per_worker = int(np.ceil(chunk_size / self.cpus))
        return chunk_size, num_chunks, chunk_size_per_worker

    def create_unitcube_samples_filename(self):
        return "X.unitcube.{}.{}.{}.{}.hdf5".format(
            self.sampling_label, self.iterations, self.num_params, self.seed
        )

    def create_rescaled_samples_filename(self):
        # Maybe we need to be more careful here, as this will change according to the model
        return "X.rescaled.{}.{}.{}.{}.hdf5".format(
            self.sampling_label, self.iterations, self.num_params, self.seed
        )

    def create_model_output_dirname(self):
        dirname = "Y.{}.{}.{}.{}".format(
            self.sampling_label, self.iterations, self.num_params, self.seed
        )
        return dirname

    def create_model_output_dir(self):
        dirpath = self.write_dir / self.dirpath_Y
        dirpath.mkdir(parents=True, exist_ok=True)

    def create_model_output_i_chunk_filename(self, i, start, end):
        return "{}.{}.{}.hdf5".format(i, start, end)

    def create_model_output_filename(self):
        return "Y.{}.{}.{}.{}.hdf5".format(
            self.sampling_label, self.iterations, self.num_params, self.seed
        )

    def create_gsa_results_filename(self):
        return "S.{}.{}.{}.{}.{}.pickle".format(
            self.gsa_label,
            self.sampling_label,
            self.iterations,
            self.num_params,
            self.seed,
        )

    def create_gsa_figure_filename(self):
        return "sensitivity.{}.{}.{}.{}.{}.html".format(
            self.gsa_label,
            self.sampling_label,
            self.iterations,
            self.num_params,
            self.seed,
        )

    @property
    def filepath_X_unitcube(self):
        return self.write_dir / "arrays" / self.create_unitcube_samples_filename()

    @property
    def filepath_X_rescaled(self):
        return self.write_dir / "arrays" / self.create_rescaled_samples_filename()

    @property
    def dirpath_Y(self):
        return self.write_dir / "arrays" / self.create_model_output_dirname()

    @property
    def filepath_Y(self):
        return self.write_dir / "arrays" / self.create_model_output_filename()

    @property
    def filepath_S(self):
        return self.write_dir / "gsa_results" / self.create_gsa_results_filename()

    @property
    def filepath_gsa_figure(self):
        return self.write_dir / "figures" / self.create_gsa_figure_filename()

    def generate_unitcube_samples(self, return_X=True):
        if self.filepath_X_unitcube.exists():
            if return_X:
                X = read_hdf5_array(self.filepath_X_unitcube)
                return X
            else:
                return self.filepath_X_unitcube
        else:
            X = self.generate_unitcube_samples_based_on_method()
            write_hdf5_array(X, self.filepath_X_unitcube)
            if return_X:
                return X
            else:
                return self.filepath_X_unitcube

    def generate_unitcube_samples_based_on_method(self):
        np.random.seed(self.seed)
        X = np.random.rand(self.iterations, self.num_params)
        return X

    def generate_rescaled_samples(self, return_X=True):
        """Rescale samples from standard uniform to appropriate distributions and write ``X_rescaled`` to a file."""
        if self.filepath_X_rescaled.exists():
            if return_X:
                X_rescaled = read_hdf5_array(self.filepath_X_rescaled)
                return X_rescaled
            else:
                return self.filepath_X_rescaled
        else:
            X = read_hdf5_array(self.filepath_X_unitcube)
            X_rescaled = self.model.rescale(X)
            write_hdf5_array(X_rescaled, self.filepath_X_rescaled)
            if return_X:
                return X_rescaled
            else:
                return self.filepath_X_rescaled

    def generate_model_output_from_chunks(self):
        across_iterations = True
        chunk_size, num_chunks, _ = self.get_parallel_params(across_iterations)
        Y = np.zeros(
            shape=(1, 0)
        )  # TODO change for number of outputs for multidimensional output
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            filepath_Y_chunk = (
                self.dirpath_Y
                / self.create_model_output_i_chunk_filename(i, start, end)
            )
            Y_chunk = read_hdf5_array(filepath_Y_chunk)
            Y = np.hstack(
                [Y, Y_chunk]
            )  # TODO change to vstack for multidimensional output
        return Y

    def generate_model_output_parallel(self, return_Y=True):
        """Obtain ``model`` outputs from the ``X_rescaled`` in parallel and write them to a file."""
        self.create_model_output_dir()
        across_iterations = True
        chunk_size, num_chunks, chunk_size_per_worker = self.get_parallel_params(
            across_iterations
        )
        for i in range(num_chunks):
            with h5py.File(self.filepath_X_rescaled, "r") as f:
                start = i * chunk_size
                end = (i + 1) * chunk_size
                X_rescaled = np.array(f["dataset"][start:end, :])
                with multiprocessing.Pool(processes=self.cpus) as pool:
                    Y_chunk = pool.map(
                        self.model,
                        [
                            X_rescaled[
                                j
                                * chunk_size_per_worker : (j + 1)
                                * chunk_size_per_worker
                            ]
                            for j in range(self.cpus)
                        ],
                    )
            Y_array = np.array([])
            for y in Y_chunk:
                Y_array = np.hstack([Y_array, y])
            filepath_Y_chunk = self.create_model_output_i_chunk_filename(i, start, end)
            write_hdf5_array(Y_array, self.dirpath_Y / filepath_Y_chunk)
        Y = self.generate_model_output_from_chunks()
        write_hdf5_array(Y, self.filepath_Y)
        if return_Y:
            return Y
        else:
            return self.filepath_Y

    def generate_model_output_sequential(self, return_Y=True):
        """Obtain ``model`` outputs from the ``X_rescaled`` sequentially and write them to a file."""
        X_rescaled = read_hdf5_array(self.filepath_X_rescaled)
        Y = self.model(X_rescaled)
        write_hdf5_array(Y, self.filepath_Y)
        if return_Y:
            return Y
        else:
            return self.filepath_Y

    def generate_model_output(self, return_Y=True):
        """Wrapper function to obtain ``model`` outputs from the ``X_rescaled`` parameter sampling matrix.

        Run Monte Carlo simulations in parallel or sequentially, and write results to a file.

        Returns
        -------
        filename_y : str
            Path where model outputs ``y`` are stored.

        """
        if self.filepath_Y.exists():
            if return_Y:
                Y = read_hdf5_array(self.filepath_Y)
                return Y
            else:
                return self.filepath_Y
        else:
            t0 = time.time()
            if self.use_parallel:
                Y = self.generate_model_output_parallel()
                t1 = time.time()
                print(
                    "generate_model_output_parallel time: " + str(t1 - t0) + " seconds"
                )
            else:
                Y = self.generate_model_output_sequential()
                t1 = time.time()
                print(
                    "generate_model_output_sequential time: "
                    + str(t1 - t0)
                    + " seconds"
                )
            return Y

    def generate_gsa_indices_based_on_method(self, selected_iterations=None):
        raise NotImplemented

    def generate_gsa_indices(self, selected_iterations=None):
        """Computation of GSA indices.

        Returns
        -------
        gsa_indices_dict : dict
            Keys are GSA indices names, values - sensitivity indices for all parameters.

        """

        if selected_iterations is None:
            if self.filepath_S.exists():
                with open(self.filepath_S, "rb") as f:
                    S_dict = pickle.load(f)
            else:
                t0 = time.time()
                S_dict = self.generate_gsa_indices_based_on_method(selected_iterations)
                t1 = time.time()
                print("GSA time: " + str(t1 - t0) + " seconds")
                with open(self.filepath_S, "wb") as f:
                    pickle.dump(S_dict, f)
        else:
            S_dict = self.generate_gsa_indices_based_on_method(selected_iterations)
        return S_dict

    def perform_gsa(self):
        self.generate_unitcube_samples(return_X=False)
        self.generate_rescaled_samples(return_X=False)
        self.generate_model_output(return_Y=False)
        S_dict = self.generate_gsa_indices()
        return S_dict

    def plot_sa_results(
        self, S_dict, S_boolean=None, S_dict_analytical=None, save_fig=False
    ):
        """Simplistic plotting of GSA results of GSA indices vs parameters. Figure is saved in the ``write_dir``.

        Parameters
        ----------
        S_dict : dict
            Keys are GSA indices names, values - sensitivity indices for all parameters.
        influential_params : list
            Parameters that are known to be influential, eg if the model is analytical. Ground truth for GSA validation.
        filename : str
            Filename for saving the plot, otherwise it will be saved under ``sensitivity_plot.pdf``.

        """
        is_boolean_given = S_boolean is not None
        is_analytical_given = S_dict_analytical is not None
        params = np.arange(self.num_params)
        fig = make_subplots(
            rows=len(S_dict),
            cols=1,
            shared_xaxes=False,
        )
        row = 1
        for gsa_method, gsa_array in S_dict.items():
            showlegend = False
            if row == 1:
                showlegend = True
            fig.add_trace(
                go.Scatter(
                    x=params,
                    y=gsa_array,
                    mode="markers",
                    marker=dict(color="#636EFA"),
                    name="All input parameters of the model",
                    legendgroup="gsa_method",
                    showlegend=showlegend,
                ),
                row=row,
                col=1,
            )
            if is_analytical_given or is_boolean_given:
                if is_analytical_given:
                    influential = S_dict_analytical[gsa_method]
                    params_influential = params
                    marker_symbol = "diamond-wide"
                    marker_color = "#EF553B"
                    name = "Analytical value of the index"
                else:
                    if is_boolean_given:
                        influential_with_zeros = S_boolean * gsa_array
                        non_zero_ind = np.where(influential_with_zeros != 0)[0]
                        influential = influential_with_zeros[non_zero_ind]
                        params_influential = params[non_zero_ind]
                        marker_symbol = "x"
                        marker_color = "#FF6692"
                        name = "Indices that are known to be important"
                fig.add_trace(
                    go.Scatter(
                        x=params_influential,
                        y=influential,
                        mode="markers",
                        marker=dict(
                            symbol=marker_symbol,
                            color=marker_color,
                        ),
                        name=name,
                        legendgroup="known_values",
                        showlegend=showlegend,
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(title_text=gsa_method, row=row, col=1)
            fig.update_xaxes(title_text="Model parameter", row=row, col=1)
            row += 1
        fig.show()
        if save_fig:
            fig.write_html(self.filepath_gsa_figure.as_posix())