import numpy as np
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .utils import read_hdf5_array


class Convergence:
    def __init__(
        self,
        filepath_Y,
        num_params,
        gsa_func,
        gsa_label,
        write_dir,
        num_steps=10,
        step_multiplier=None,
        convergence_step=None,
        iterations_order=None,
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
        # Parameters that define how many gsa indices to compute
        self.num_steps = num_steps
        if step_multiplier is None:
            step_multiplier = max(self.iterations // self.num_steps, 1)
        self.step_multiplier = step_multiplier
        if convergence_step is None:
            convergence_step = self.generate_convergence_step(
                self.gsa_label, self.step_multiplier
            )
        self.convergence_step = convergence_step
        if iterations_order is None:
            iterations_order = self.generate_iterations_order(self.gsa_label)
        self.iterations_order = iterations_order
        self.sampling_label = str(self.filepath_Y).split(".")[
            -5
        ]  # TODO there must be a better way
        self.seed = str(self.filepath_Y).split(".")[-2]

    def create_gsa_convergence_filename(self):
        return "convergence.{}.{}.{}.html".format(
            self.gsa_label,
            self.sampling_label,
            self.iterations,
            self.num_params,
            self.seed,
        )  # TODO add sampling_label and seed that are not dependent on the filepath_Y, to make class more general

    def generate_convergence_step(self, gsa_label, step_multiplier):
        if gsa_label == "gsa_saltelli":
            convergence_step = step_multiplier * (self.num_params + 2)
        else:
            convergence_step = max(
                self.iterations // self.num_steps, 1
            )  # default, but also works for gsa_correlations
        return convergence_step

    def generate_iterations_order(self, gsa_label=None):
        # default order, also works for gsa_correlations and gsa_saltelli
        # if gsa_label is None:
        iterations_order = np.arange(self.iterations)
        return iterations_order

    @property
    def filepath_convergence_figure(self):
        return self.write_dir / self.create_gsa_convergence_filename()

    def run_convergence(self, parameter_inds=None):
        sa_convergence_dict, iterations_blocks = self.generate_converging_gsa_indices()
        fig = self.plot_convergence(
            sa_convergence_dict, iterations_blocks, parameter_inds
        )
        return fig

    def generate_converging_gsa_indices(self):
        """
        gsa_func : function or method
            Corresponds to generate_gsa_indices_based_on_method from the class SensitivityAnalysisMethod.
            Needs to accept an argument ``selected_iterations``
        """
        sa_convergence_dict_temp = {}
        iterations_blocks = np.arange(
            self.convergence_step,
            len(self.Y) + self.convergence_step,
            self.convergence_step,
        )
        for block_size in iterations_blocks:
            selected_iterations = self.iterations_order[0:block_size]
            t0 = time.time()
            gsa_indices_dict = self.gsa_func(selected_iterations)
            t1 = time.time()
            print("{0:8d} iterations -> {1:8.3f} s".format(block_size, t1 - t0))
            sa_convergence_dict_temp[block_size] = gsa_indices_dict
        # Put all blocks together
        sa_convergence_dict = {
            key: np.zeros(shape=(0, self.num_params))
            for key in sa_convergence_dict_temp[iterations_blocks[0]].keys()
        }
        for sa_dict in sa_convergence_dict_temp.values():
            for key, sa_array in sa_convergence_dict.items():
                new_sa_array = np.vstack([sa_array, sa_dict[key]])
                sa_convergence_dict.update({key: new_sa_array})

        return sa_convergence_dict, iterations_blocks

    def plot_convergence(
        self,
        sa_convergence_dict,
        iterations_blocks,
        parameter_inds=None,
        save_fig=False,
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
                        marker=dict(color=colors[parameter]),
                        name="Parameter " + str(parameter),
                        legendgroup=str(parameter),
                    ),
                    row=row,
                    col=1,
                )
                row += 1
        fig.show()
        if save_fig:
            fig.write_html(self.filepath_convergence_figure.as_posix())
        return fig
