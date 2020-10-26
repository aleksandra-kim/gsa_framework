import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr, wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from .utils import read_hdf5_array, write_hdf5_array

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
        default_x=None,
        write_dir=None,
    ):
        # self.gsa_indices = gsa_indices
        self.model = model
        self.num_params = len(model)
        if default_x is None:
            default_x = np.array([0.5] * self.num_params)
        self.iterations = iterations
        self.seed = seed
        np.random.seed(self.seed)
        self.default_x = default_x
        if write_dir is None:
            write_dir = "."
        self.write_dir = Path(write_dir)
        self.base_X_rescaled, self.base_Y = self.get_base_X_Y()

    def get_base_X_Y(self):
        # Unitcube samples
        if not self.filepath_base_X_unitcube.exists():
            X = np.random.rand(self.iterations, self.num_params)
            write_hdf5_array(X, self.filepath_base_X_unitcube)
        else:
            X = read_hdf5_array(self.filepath_base_X_unitcube)
        # Rescaled samples
        if not self.filepath_base_X_rescaled.exists():
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

    def get_influential_Y_from_gsa(self, gsa_indices, num_influential):
        X_rescaled = read_hdf5_array(
            self.filepath_base_X_rescaled
        )  # read from file instead of using self.X
        non_influential_inds = np.argsort(gsa_indices)[::-1][num_influential:]
        non_influential_inds.sort()
        X_rescaled[:, non_influential_inds] = np.tile(
            self.default_x[non_influential_inds], (self.iterations, 1)
        )
        Y = self.model(X_rescaled)
        return Y

    def get_influential_Y_from_parameter_choice(self, parameter_choice):
        """Variable ``parameter_choice`` is the indices of influential parameters."""
        num_influential = len(parameter_choice)
        filepath = self.create_influential_model_output_filepath(num_influential)
        if filepath.exists():
            print("File ``{}`` already exists".format(filepath.name))
            Y = read_hdf5_array(filepath).flatten()
        else:
            X_rescaled = read_hdf5_array(
                self.filepath_base_X_rescaled
            )  # read from file instead of using self.X
            non_influential_inds = np.setdiff1d(
                np.arange(self.num_params), parameter_choice
            )
            non_influential_inds.sort()
            X_rescaled[:, non_influential_inds] = np.tile(
                self.default_x[non_influential_inds], (self.iterations, 1)
            )
            Y = self.model(X_rescaled)
            write_hdf5_array(Y, filepath)
        return Y

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

    def create_influential_model_output_filepath(self, num_influential):
        filename = "validation.influential.Y.{}.{}.{}.hdf5".format(
            self.iterations, num_influential, self.seed
        )
        filepath = self.write_dir / "arrays" / filename
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

    def create_figure_correlation_filename(self, num_influential):
        # Maybe we need to be more careful here, as this will change according to the model
        return "validation.correlation.{}.{}.{}.html".format(
            self.iterations, num_influential, self.seed
        )

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

    @property
    def filepath_figure_histogram(self):
        return self.write_dir / "figures" / self.create_figure_histogram_filename()

    @property
    def filepath_figure_correlation(self):
        return self.write_dir / "figures" / self.create_figure_correlation_filename()

    def generate_plots(
        self, save_fig=False, plot_histogram=True, plot_correlation=True
    ):
        if plot_histogram:
            fig_hist, _ = self.plot_histogram(self.base_Y, self.influential_y)
        if plot_correlation:
            fig_corr, _, _ = self.plot_correlation(self.base_Y, self.influential_y)
        if save_fig:
            if plot_histogram:
                fig_hist.write_html(self.filepath_figure_histogram)
            if plot_correlation:
                fig_corr.write_html(self.filepath_figure_correlation)

    def plot_base_Y(
        self,
        base_y,
        bin_min=None,
        bin_max=None,
        num_bins=60,
        save_fig=False,
        default_y=None,
    ):
        if bin_min is None:
            bin_min = min(base_y)
        if bin_max is None:
            bin_max = max(base_y)
        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq_base, bin_base = np.histogram(base_y, bins=bins_)
        opacity_ = 0.65

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=bin_base,
                y=freq_base,
                name="All parameters vary",
                opacity=opacity_,
                marker=dict(color=COLORS_DICT["all"]),
                showlegend=True,
            ),
        )
        if default_y is not None:
            fig.add_trace(
                go.Scatter(
                    x=[default_y],
                    y=[0],
                    mode="markers",
                    name="Static score",
                    opacity=opacity_,
                    marker=dict(
                        color="red",
                        size=20,
                        symbol="x",
                    ),
                    showlegend=True,
                ),
            )
        fig.update_layout(
            width=500,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.55, y=0.9),
        )
        fig.update_yaxes(title_text="Frequency")
        fig.update_xaxes(title_text="LCIA scores, [kg CO2-eq]")
        fig.show()
        if save_fig:
            # fig.write_html(self.filepath_figure_base_Y_histogram)
            fig.write_image(str(self.filepath_figure_base_Y_histogram))
        return fig

    def plot_correlation(self, base_y, influential_y, start=0, end=50):
        pearson_correlation = np.corrcoef(base_y, influential_y)[0, 1]
        spearman_correlation, _ = spearmanr(base_y, influential_y)

        x = np.arange(start, end)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=base_y[start:end],
                name="All parameters vary",
                mode="lines+markers",
                marker=dict(color=COLORS_DICT["all"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=influential_y[start:end],
                name="Only influential parameters vary",
                mode="lines+markers",
                marker=dict(color=COLORS_DICT["influential"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=base_y,
                y=influential_y,
                name="Scatter plot between ``base_y`` and ``influential_y``",
                mode="markers",
                marker=dict(color=COLORS_DICT["scatter"]),
            ),
            row=2,
            col=1,
        )
        # Add annotation on the values of pearson and spearman correlation coefficients
        annotations = [
            dict(
                x=1.1,
                y=0.89,
                xref="paper",
                yref="paper",
                text="Pearson correlation coefficient is {:4.3f}".format(
                    pearson_correlation
                ),
                xanchor="left",
                yanchor="middle",
                showarrow=False,
            ),
            dict(
                x=1.1,
                y=0.865,
                xref="paper",
                yref="paper",
                text="Spearman correlation coefficient is {:4.3f}".format(
                    spearman_correlation
                ),
                xanchor="left",
                yanchor="middle",
                showarrow=False,
            ),
        ]
        fig.update_layout(
            width=1200,
            height=1000,
            annotations=annotations,
            legend=dict(x=1.1, y=1),  # on top
            yaxis1=dict(domain=[0.8, 1]),
            yaxis2=dict(domain=[0.0, 0.7]),
        )
        fig.update_xaxes(
            title_text="Subset of model outputs, {0}/{1} datapoints".format(
                end - start, base_y.shape[0]
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Model outputs", row=1, col=1)
        fig.update_xaxes(
            title_text="Model outputs when all parameters vary",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Model outputs when only influential parameters vary",
            row=2,
            col=1,
        )
        fig.show()
        return fig, pearson_correlation, spearman_correlation

    def plot_histogram(
        self, base_y, influential_y, bin_min=None, bin_max=None, num_bins=100
    ):
        wasserstein_dist = wasserstein_distance(base_y, influential_y)

        if bin_min is None:
            bin_min = min(np.hstack([base_y, influential_y]))
        if bin_max is None:
            bin_max = max(np.hstack([base_y, influential_y]))
        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq_base, bin_base = np.histogram(base_y, bins=bins_)
        freq_inf, bin_inf = np.histogram(influential_y, bins=bins_)
        opacity_ = 0.65

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        fig.add_trace(
            go.Bar(
                x=bin_base,
                y=freq_base,
                name="All parameters vary",
                opacity=opacity_,
                marker=dict(color=COLORS_DICT["all"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=bin_inf,
                y=freq_inf,
                name="Only influential parameters vary",
                opacity=opacity_,
                marker=dict(color=COLORS_DICT["influential"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=influential_y,
                marker=dict(color=COLORS_DICT["influential"]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=base_y,
                marker=dict(color=COLORS_DICT["all"]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        annotations = [
            dict(
                x=1.02,
                y=0.87,
                xref="paper",
                yref="paper",
                text="Wasserstein distance is {:6.3f}".format(wasserstein_dist),
                xanchor="left",
                yanchor="top",
                showarrow=False,
            ),
        ]
        # Overlay both histograms
        fig.update_layout(
            barmode="overlay",
            width=1000,
            height=600,
            annotations=annotations,
            margin=dict(l=20, r=100, t=60, b=20),
            legend=dict(x=1, y=1),
        )
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Model outputs", row=2, col=1)
        fig.layout["yaxis2"].update(showticklabels=False)
        fig.show()

        return fig, wasserstein_dist
