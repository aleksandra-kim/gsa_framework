import numpy as np
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .utils import read_hdf5_array, read_pickle, write_pickle


class Convergence:
    def __init__(
        self,
        filepath_Y,
        num_params,
        gsa_func,
        gsa_label,
        write_dir,
        num_steps=10,
        **kwargs,
    ):
        self.filepath_Y = filepath_Y
        self.Y = read_hdf5_array(filepath_Y).flatten()
        self.iterations = self.Y.shape[0]
        self.num_params = num_params
        self.gsa_func = gsa_func
        self.gsa_label = gsa_label
        self.write_dir = write_dir
        self.make_dirs()
        self.num_steps = num_steps
        self.iterations_order = self.generate_iterations_order()
        (
            self.iterations_min,
            self.iterations_least_common_multiple,
        ) = self.generate_iterations_min_and_least_common_multiple(
            self.gsa_label, **kwargs
        )  # depends on gsa method
        (
            self.iterations_step,
            self.iterations_for_convergence,
        ) = self.generate_iterations_for_convergence()
        self.sampling_label = str(self.filepath_Y).split(".")[
            1
        ]  # TODO there must be a better way
        self.seed = str(self.filepath_Y).split(".")[-2]

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["arrays", "figures"]
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_iterations_order(self):
        return np.arange(self.iterations)  # for some gsa methods can be shuffled

    def generate_iterations_min_and_least_common_multiple(
        self, gsa_label, **kwargs
    ):  # TODO should be in methods?
        if "saltelliGsa" in gsa_label:
            iterations_least_common_multiple = self.num_params + 2
            iterations_min_not_saltelli = self.iterations // self.num_steps
            iterations_min_saltelli = (
                iterations_min_not_saltelli // (self.num_params + 2) + 1
            ) * (self.num_params + 2)
            iterations_min = max(iterations_min_saltelli, self.num_params + 2)
        elif "eFastGsa" in gsa_label:
            M = kwargs.get("M", 4)
            iterations_least_common_multiple = 4 * M ** 2 + 1
            iterations_min_not_efast = self.iterations // self.num_steps
            iterations_min_efast = (
                iterations_min_not_efast // self.num_params + 1
            ) * self.num_params
            iterations_min = max(
                iterations_min_efast, iterations_least_common_multiple * self.num_params
            )
        else:
            iterations_least_common_multiple = 1
            iterations_min = max(self.iterations // self.num_steps, 20)
        return iterations_min, iterations_least_common_multiple

    def generate_iterations_for_convergence(self):
        factor = max(
            (self.iterations - self.iterations_min)
            // (self.num_steps - 1)
            // self.iterations_least_common_multiple,
            1,
        )
        iterations_step = factor * self.iterations_least_common_multiple
        iterations_for_convergence = np.arange(
            self.iterations_min, self.iterations, iterations_step
        )
        return iterations_step, iterations_for_convergence

    def create_convergence_dict_filepath(self):
        filename = "convergence.S.{}.{}.{}Step{}.{}.pickle".format(
            self.gsa_label,
            self.sampling_label,
            self.iterations,
            self.iterations_step,
            self.seed,
        )  # TODO add sampling_label and seed that are not dependent on the filepath_Y, to make class more general
        filepath = self.write_dir / "arrays" / filename
        return filepath

    def create_figure_convergence_filepath(self, extension):
        filename = "C.{}.{}.{}Step{}.{}.{}".format(
            self.gsa_label,
            self.sampling_label,
            self.iterations,
            self.iterations_step,
            self.seed,
            extension,
        )  # TODO add sampling_label and seed that are not dependent on the filepath_Y, to make class more general
        filepath = self.write_dir / "figures" / filename
        return filepath

    def run_convergence(self, parameter_inds=None, fig_format=()):
        t0 = time.time()
        filepath_convergence_dict = self.create_convergence_dict_filepath()
        if filepath_convergence_dict.exists():
            print("{} already exists".format(filepath_convergence_dict.name))
            sa_convergence_dict = read_pickle(filepath_convergence_dict)
        else:
            sa_convergence_dict = self.generate_converging_gsa_indices()
            write_pickle(sa_convergence_dict, filepath_convergence_dict)
        t1 = time.time()
        print("Total convergence time -> {:8.3f} s".format(t1 - t0))
        fig = self.plot_convergence(sa_convergence_dict, parameter_inds, fig_format)
        return fig

    def generate_converging_gsa_indices(self):
        """
        gsa_func : function or method
            Corresponds to generate_gsa_indices_based_on_method from the class SensitivityAnalysisMethod.
            Needs to accept an argument ``selected_iterations``
        """

        sa_convergence_dict_temp = {}
        for i, iterations_current in enumerate(self.iterations_for_convergence):
            selected_iterations = self.iterations_order[0:iterations_current]
            parameters_convergence_dict = {
                "iterations": iterations_current,
                "iterations_step": self.iterations_step,
                "selected_iterations": selected_iterations,
                "flag_convergence": True,
            }
            t0 = time.time()
            gsa_indices_dict = self.gsa_func(**parameters_convergence_dict)
            t1 = time.time()
            print(
                "{0:4d}. {1:8d} iterations -> {2:8.3f} s".format(
                    i, iterations_current, t1 - t0
                )
            )
            sa_convergence_dict_temp[iterations_current] = gsa_indices_dict

        # Put all blocks together
        sa_convergence_dict = {
            key: np.zeros(shape=(0, self.num_params))
            for key in sa_convergence_dict_temp[
                self.iterations_for_convergence[0]
            ].keys()
        }
        for sa_dict in sa_convergence_dict_temp.values():
            for key, sa_array in sa_convergence_dict.items():
                new_sa_array = np.vstack([sa_array, sa_dict[key]])
                sa_convergence_dict.update({key: new_sa_array})
        # TODO remove intermediate files for sensitivities, eg in correlations
        sa_convergence_dict["iterations"] = np.array(
            list(sa_convergence_dict_temp.keys())
        )
        return sa_convergence_dict

    def plot_convergence(
        self,
        sa_convergence_dict,
        parameter_inds=None,
        fig_format=[],
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
        x = sa_convergence_dict["iterations"]
        sa_convergence_dict.pop("iterations")
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
                        x=x,
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
            fig.write_image(self.create_figure_convergence_filepath("pdf").as_posix())
        if "html" in fig_format:
            fig.write_html(self.create_figure_convergence_filepath("html").as_posix())
        if "pickle" in fig_format:
            filepath = self.create_figure_convergence_filepath("pickle").as_posix()
            write_pickle(fig, filepath)
        return fig
