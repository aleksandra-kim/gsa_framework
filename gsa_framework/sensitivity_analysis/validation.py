import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr, wasserstein_distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS_DICT = {
    "all": "#636EFA",
    "influential": "#EF553B",
    "scatter": "#00CC96",
}


class Validation:
    def __init__(
        self, gsa_indices, model, iterations=500, num_influential=10, default_x=None
    ):
        self.gsa_indices = gsa_indices
        self.model = model
        self.num_params = len(model)
        if default_x is None:
            default_x = np.array([0.5] * self.num_params)
        self.iterations = iterations
        self.num_influential = num_influential
        self.default_x = default_x
        self.base_X, self.base_y = self.get_base_X_y()
        self.influential_y = self.get_influential_y(self.base_X)

    def get_base_X_y(self):
        X = np.random.rand(self.iterations, self.num_params)
        y = self.model(X)
        print()
        return X, y

    def get_influential_y(self, X):
        X_ = deepcopy(X)
        non_influential_inds = np.argsort(self.gsa_indices)[::-1][
            self.num_influential :
        ]
        non_influential_inds.sort()
        X_[:, non_influential_inds] = np.tile(
            self.default_x[non_influential_inds], (self.iterations, 1)
        )
        y = self.model(X_)
        return y

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
        self, base_y, influential_y, bin_min=None, bin_max=None, num_bins=50
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
