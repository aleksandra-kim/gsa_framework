import numpy as np
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .utils import read_hdf5_array, write_hdf5_array, read_pickle, write_pickle


class Convergence:
    def __init__(
        self,
        filepath_Y,
        num_params,
        gsa_func,
        gsa_label,
        write_dir,
        num_steps=10,  # how many gsa indices to compute
    ):
        self.filepath_Y = filepath_Y
        self.Y = read_hdf5_array(
            filepath_Y
        ).flatten()  # assume Y does not occupy too much memory
        self.iterations = self.Y.shape[0]
        self.num_params = num_params
        self.gsa_func = gsa_func
        self.gsa_label = gsa_label
        self.write_dir = write_dir
        self.num_steps = num_steps
        # Compute parameters for convergence
        self.min_block_size = self.generate_min_block_size(
            self.gsa_label
        )  # depends on gsa method
        self.block_factor = max(
            self.iterations // self.min_block_size // self.num_steps, 1
        )
        self.block_size = self.block_factor * self.min_block_size
        self.iterations_order = self.generate_iterations_order()
        self.iterations_blocks = np.arange(
            self.block_size, len(self.Y) + self.block_size, self.block_size
        )
        # self.sampling_label = str(self.filepath_Y).split(".")[
        #     -5
        # ]  # TODO there must be a better way
        self.seed = str(self.filepath_Y).split(".")[-2]

    def generate_min_block_size(self, gsa_label):
        if gsa_label in ["gsa_correlations", "gsa_delta"]:  # TODO change for delta
            min_block_size = 1
        elif gsa_label == "gsa_saltelli":
            min_block_size = self.num_params + 2
        else:
            min_block_size = 1
        return min_block_size

    def generate_iterations_order(self):
        return np.arange(self.iterations)  # for somr gsa methods can be shuffled

    def create_convergence_dict_filepath(self, tag):
        filename = "convergence.block{}.{}.{}.{}.{}.pickle".format(
            self.block_size,
            self.iterations,
            self.num_params,
            self.seed,
            tag,
        )  # TODO add sampling_label and seed that are not dependent on the filepath_Y, to make class more general
        filepath = self.write_dir / "arrays" / filename
        return filepath

    def create_convergence_figure_filepath(self, tag, fig_format):
        filename = "convergence.block{}.{}.{}.{}.{}.{}".format(
            self.block_size,
            self.iterations,
            self.num_params,
            self.seed,
            tag,
            fig_format,
        )  # TODO add sampling_label and seed that are not dependent on the filepath_Y, to make class more general
        filepath = self.write_dir / "figures" / filename
        return filepath

    def run_convergence(self, parameter_inds=None, tag=None, fig_format=[]):
        if tag is None:
            tag = self.gsa_label
        filepath_convergence_dict = self.create_convergence_dict_filepath(tag)
        if filepath_convergence_dict.exists():
            sa_convergence_dict = read_pickle(filepath_convergence_dict)
        else:
            sa_convergence_dict = self.generate_converging_gsa_indices()
            write_pickle(sa_convergence_dict, filepath_convergence_dict)
        fig = self.plot_convergence(
            sa_convergence_dict, parameter_inds, tag, fig_format
        )
        return fig

    def generate_converging_gsa_indices(self):
        """
        gsa_func : function or method
            Corresponds to generate_gsa_indices_based_on_method from the class SensitivityAnalysisMethod.
            Needs to accept an argument ``selected_iterations``
        """
        sa_convergence_dict_temp = {}
        for block_size in self.iterations_blocks:
            selected_iterations = self.iterations_order[0:block_size]
            parameters_convergence_dict = {
                "iterations": block_size,
                "iterations_step": self.block_size,
                "selected_iterations": selected_iterations,
                "flag_convergence": True,
            }
            t0 = time.time()
            gsa_indices_dict = self.gsa_func(**parameters_convergence_dict)
            t1 = time.time()
            print("{0:8d} iterations -> {1:8.3f} s".format(block_size, t1 - t0))
            sa_convergence_dict_temp[block_size] = gsa_indices_dict
        # Put all blocks together
        sa_convergence_dict = {
            key: np.zeros(shape=(0, self.num_params))
            for key in sa_convergence_dict_temp[self.iterations_blocks[0]].keys()
        }
        for sa_dict in sa_convergence_dict_temp.values():
            for key, sa_array in sa_convergence_dict.items():
                new_sa_array = np.vstack([sa_array, sa_dict[key]])
                sa_convergence_dict.update({key: new_sa_array})
        return sa_convergence_dict

    def plot_convergence(
        self,
        sa_convergence_dict,
        parameter_inds=None,
        tag=None,
        fig_format=None,
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
                        x=self.iterations_blocks,
                        y=sa_array[:, parameter],
                        mode="lines+markers",
                        showlegend=showlegend,
                        marker=dict(color=colors[parameter]),
                        name="Parameter " + str(parameter),
                        legendgroup=str(parameter),
                    ),
                    row=row,
                    col=1,
                )
                row += 1
        fig.show()
        if "pdf" in fig_format:
            fig.write_image(
                self.create_convergence_figure_filepath(tag, "pdf").as_posix()
            )
        if "html" in fig_format:
            fig.write_html(
                self.create_convergence_figure_filepath(tag, "html").as_posix()
            )
        return fig
