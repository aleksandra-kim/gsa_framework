import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr, wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from .utils import read_hdf5_array, write_hdf5_array
from gsa_framework.plotting import *

COLORS_DICT = {
    "all": "#636EFA",
    "influential": "#EF553B",
    "scatter": "#00CC96",
}


class Validation:
    def __init__(
        self,
        model,
        iterations=500,
        seed=None,
        default_x_rescaled=None,
        write_dir=None,
    ):
        self.model = model
        self.num_params = len(model)
        if default_x_rescaled is None:
            default_x_rescaled = model.default_uncertain_amounts
        self.default_x_rescaled = default_x_rescaled
        self.iterations = iterations
        self.seed = seed
        np.random.seed(self.seed)

        if write_dir is None:
            write_dir = "."
        self.write_dir = Path(write_dir)
        self.base_X_rescaled, self.base_Y = self.get_base_X_Y()

    def get_base_X_Y(self):
        # Rescaled samples
        if not self.filepath_base_X_rescaled.exists():
            # Unitcube samples
            if not self.filepath_base_X_unitcube.exists():
                X = np.random.rand(self.iterations, self.num_params)
                write_hdf5_array(X, self.filepath_base_X_unitcube)
            else:
                X = read_hdf5_array(self.filepath_base_X_unitcube)
            X_rescaled = self.model.rescale(X)
            write_hdf5_array(X_rescaled, self.filepath_base_X_rescaled)
        else:
            X_rescaled = read_hdf5_array(self.filepath_base_X_rescaled)
        # Model output
        if not self.filepath_base_Y.exists():
            Y = self.model(X_rescaled)
            write_hdf5_array(Y, self.filepath_base_Y)
        else:
            Y = read_hdf5_array(self.filepath_base_Y)
            Y = Y.flatten()
        return X_rescaled, Y

    def get_influential_Y_from_gsa(self, gsa_indices, num_influential, tag=None):
        assert num_influential <= self.num_params
        assert len(gsa_indices) == self.num_params
        if tag is None:
            tag = num_influential
        filepath = self.create_influential_model_output_filepath(tag)
        if filepath.exists():
            influential_Y = read_hdf5_array(filepath).flatten()
        else:
            X_rescaled = read_hdf5_array(
                self.filepath_base_X_rescaled
            )  # read from file instead of using self.X
            non_influential_inds = np.argsort(gsa_indices)[::-1][num_influential:]
            non_influential_inds.sort()
            X_rescaled[:, non_influential_inds] = np.tile(
                self.default_x_rescaled[non_influential_inds], (self.iterations, 1)
            )
            influential_Y = self.model(X_rescaled)
            write_hdf5_array(influential_Y, filepath)
        return influential_Y

    def get_influential_Y_from_parameter_choice(self, parameter_choice, tag=None):
        """Variable ``parameter_choice`` is the indices of influential parameters."""
        num_influential = len(parameter_choice)
        assert num_influential <= self.num_params
        if tag is None:
            tag = num_influential
        filepath = self.create_influential_model_output_filepath(tag)
        if filepath.exists():
            influential_Y = read_hdf5_array(filepath).flatten()
        else:
            X_rescaled = read_hdf5_array(
                self.filepath_base_X_rescaled
            )  # read from file instead of using self.X
            non_influential_inds = np.setdiff1d(
                np.arange(self.num_params), parameter_choice
            )
            non_influential_inds.sort()
            X_rescaled[:, non_influential_inds] = np.tile(
                self.default_x_rescaled[non_influential_inds], (self.iterations, 1)
            )
            influential_Y = self.model(X_rescaled)
            write_hdf5_array(influential_Y, filepath)
        return influential_Y

    def create_base_unitcube_samples_filename(self):
        return "validation.base.X.unitcube.{}.{}.{}.hdf5".format(
            self.iterations, self.num_params, self.seed
        )

    def create_base_rescaled_samples_filename(self):
        return "validation.base.X.rescaled.{}.{}.{}.hdf5".format(
            self.iterations, self.num_params, self.seed
        )

    def create_base_model_output_filename(self):
        return "validation.base.Y.{}.{}.{}.hdf5".format(
            self.iterations, self.num_params, self.seed
        )

    def create_influential_model_output_filepath(self, tag):
        filename = "validation.influential.Y.{}.{}.{}.hdf5".format(
            self.iterations, tag, self.seed
        )
        filepath = self.write_dir / "arrays" / filename
        return filepath

    def create_histogram_base_Y_influential_Y_filepath(self, tag):
        filename = "validation.histogram.base.influential.Y.{}.{}.{}.pdf".format(
            self.iterations, tag, self.seed
        )
        filepath = self.write_dir / "figures" / filename
        return filepath

    def create_correlation_base_Y_influential_Y_filepath(self, tag):
        filename = "validation.correlation.base.influential.Y.{}.{}.{}.pdf".format(
            self.iterations, tag, self.seed
        )
        filepath = self.write_dir / "figures" / filename
        return filepath

    def create_figure_base_Y_histogram_filename(self):
        # Maybe we need to be more careful here, as this will change according to the model
        return "validation.histogram.base_Y.{}.{}.{}.pdf".format(
            self.iterations, self.num_params, self.seed
        )

    def create_figure_histogram_filename(self, num_influential):
        # Maybe we need to be more careful here, as this will change according to the model
        return "validation.histogram.{}.{}.{}.html".format(
            self.iterations, num_influential, self.seed
        )

    def create_figure_correlation_filepath(self, num_influential):
        filename = "validation.correlation.{}.{}.{}.pdf".format(
            self.iterations, num_influential, self.seed
        )
        filepath = self.write_dir / "figures" / filename
        return filepath

        # Maybe we need to be more careful here, as this will change according to the model

    @property
    def filepath_base_X_unitcube(self):
        return self.write_dir / "arrays" / self.create_base_unitcube_samples_filename()

    @property
    def filepath_base_X_rescaled(self):
        return self.write_dir / "arrays" / self.create_base_rescaled_samples_filename()

    @property
    def filepath_base_Y(self):
        return self.write_dir / "arrays" / self.create_base_model_output_filename()

    @property
    def filepath_figure_base_Y_histogram(self):
        return (
            self.write_dir / "figures" / self.create_figure_base_Y_histogram_filename()
        )

    def plot_histogram_base_Y(
        self, bin_min=None, bin_max=None, num_bins=60, save_fig=False
    ):
        fig = histogram_Y(
            Y=self.base_Y,
            default_Y=self.model.static_output,
            bin_min=bin_min,
            bin_max=bin_max,
            num_bins=num_bins,
            color=COLORS_DICT["all"],
            xaxes_title_text=self.model.output_name,
            trace_name="All parameters vary",
            trace_name_default="Static score",
        )
        if save_fig:
            fig.write_image(str(self.filepath_figure_base_Y_histogram))
        return fig

    def plot_histogram_base_Y_influential_Y(
        self,
        influential_Y,
        tag=None,
        save_fig=False,
        bin_min=None,
        bin_max=None,
        num_bins=60,
    ):
        fig = histogram_Y1_Y2(
            self.base_Y,
            influential_Y,
            default_Y=None,
            bin_min=bin_min,
            bin_max=bin_max,
            num_bins=num_bins,
            trace_name1="All parameters vary",
            trace_name2="Only influential vary",
            color1="#636EFA",
            color2="#EF553B",
            color_default_Y="red",
            opacity=0.65,
            xaxes_title_text=self.model.output_name,
        )
        if save_fig:
            filepath = self.create_histogram_base_Y_influential_Y_filepath(tag)
            fig.write_image(str(filepath))
        return fig

    def plot_correlation_base_Y_influential_Y(
        self, influential_Y, tag=None, save_fig=False
    ):
        fig = correlation_Y1_Y2(
            Y1=self.base_Y,
            Y2=influential_Y,
            start=200,
            end=260,
            trace_name1="All parameters vary",
            trace_name2="Only influential vary",
            yaxes1_title_text=self.model.output_name,
            xaxes2_title_text=self.model.output_name,
            yaxes2_title_text=self.model.output_name,
        )
        if save_fig:
            filepath = self.create_correlation_base_Y_influential_Y_filepath(tag)
            fig.write_image(str(filepath))
        return fig

    # def plot_histogram(
    #     self, base_y, influential_y, bin_min=None, bin_max=None, num_bins=60
    # ):
    #     wasserstein_dist = wasserstein_distance(base_y, influential_y)
    #
    #     if bin_min is None:
    #         bin_min = min(np.hstack([base_y, influential_y]))
    #     if bin_max is None:
    #         bin_max = max(np.hstack([base_y, influential_y]))
    #     bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    #     freq_base, bin_base = np.histogram(base_y, bins=bins_)
    #     freq_inf, bin_inf = np.histogram(influential_y, bins=bins_)
    #     opacity_ = 0.65
    #
    #     fig = make_subplots(
    #         rows=2,
    #         cols=1,
    #         shared_xaxes=True,
    #         vertical_spacing=0.05,
    #     )
    #
    #     fig.add_trace(
    #         go.Bar(
    #             x=bin_base,
    #             y=freq_base,
    #             name="All parameters vary",
    #             opacity=opacity_,
    #             marker=dict(color=COLORS_DICT["all"]),
    #         ),
    #         row=1,
    #         col=1,
    #     )
    #     fig.add_trace(
    #         go.Bar(
    #             x=bin_inf,
    #             y=freq_inf,
    #             name="Only influential parameters vary",
    #             opacity=opacity_,
    #             marker=dict(color=COLORS_DICT["influential"]),
    #         ),
    #         row=1,
    #         col=1,
    #     )
    #     fig.add_trace(
    #         go.Box(
    #             x=influential_y,
    #             marker=dict(color=COLORS_DICT["influential"]),
    #             showlegend=False,
    #         ),
    #         row=2,
    #         col=1,
    #     )
    #     fig.add_trace(
    #         go.Box(
    #             x=base_y,
    #             marker=dict(color=COLORS_DICT["all"]),
    #             showlegend=False,
    #         ),
    #         row=2,
    #         col=1,
    #     )
    #     annotations = [
    #         dict(
    #             x=1.02,
    #             y=0.87,
    #             xref="paper",
    #             yref="paper",
    #             text="Wasserstein distance is {:6.3f}".format(wasserstein_dist),
    #             xanchor="left",
    #             yanchor="top",
    #             showarrow=False,
    #         ),
    #     ]
    #     # Overlay both histograms
    #     fig.update_layout(
    #         barmode="overlay",
    #         width=1000,
    #         height=600,
    #         annotations=annotations,
    #         margin=dict(l=20, r=100, t=60, b=20),
    #         legend=dict(x=1, y=1),
    #     )
    #     fig.update_yaxes(title_text="Frequency", row=1, col=1)
    #     fig.update_xaxes(title_text="Model outputs", row=2, col=1)
    #     fig.layout["yaxis2"].update(showticklabels=False)
    #     fig.show()
    #
    #     return fig, wasserstein_dist
