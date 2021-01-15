import numpy as np
from scipy import stats

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .utils import get_z_alpha_2


def get_confidence_intervals(stability_dict, confidence_level=0.95):
    num_params = list(stability_dict.values())[0].shape[1]
    iter_ = np.array(list(stability_dict.keys()))
    mean_ = np.zeros((len(stability_dict), num_params))
    mins_ = np.zeros((len(stability_dict), num_params))
    maxs_ = np.zeros((len(stability_dict), num_params))
    # use confidence intervals
    z_alpha_2 = get_z_alpha_2(confidence_level)
    for i, S_bootstrap in enumerate(stability_dict.values()):
        mean_[i, :] = np.mean(S_bootstrap, axis=0)
        mins_[i, :] = mean_[i, :] - z_alpha_2 * np.std(S_bootstrap, axis=0)
        maxs_[i, :] = mean_[i, :] + z_alpha_2 * np.std(S_bootstrap, axis=0)
    return iter_, mean_, mins_, maxs_


def get_min_max_bands(stability_dict):
    num_params = list(stability_dict.values())[0].shape[1]
    iter_ = np.array(list(stability_dict.keys()))
    mean_ = np.zeros((len(stability_dict), num_params))
    mins_ = np.zeros((len(stability_dict), num_params))
    maxs_ = np.zeros((len(stability_dict), num_params))
    for i, S_bootstrap in enumerate(stability_dict.values()):
        mean_[i, :] = np.mean(S_bootstrap, axis=0)
        mins_[i, :] = np.min(S_bootstrap, axis=0)
        maxs_[i, :] = np.max(S_bootstrap, axis=0)
    return iter_, mean_, mins_, maxs_


def min_max_bands(mean_, mins_, maxs_):
    # Absolute width min-max bands
    width = maxs_ - mins_
    inds = np.zeros(len(width), dtype=int)
    band_min = np.zeros(len(width))
    band_max = np.zeros(len(width))
    for i in range(len(width)):
        inds[i] = np.where(width[i, :] == np.max(width[i, :]))[0][0]
        band_min[i] = mean_[i, :][inds[i]] - mins_[i, :][inds[i]]
        band_max[i] = -mean_[i, :][inds[i]] + maxs_[i, :][inds[i]]
    return band_min, band_max


def min_max_band(stability_dict):
    iterations = np.array(list(stability_dict.keys()))

    return


def plot_confidence_convergence(sb_dict, sensitivity_index_names=None):
    convergence_iterations = sb_dict["iterations"]
    sb_dict.pop("iterations")
    num_params = len(sb_dict)
    if sensitivity_index_names is None:
        sensitivity_index_names = list(list(sb_dict.values())[0].keys())

    # Plotting
    nrows = len(sensitivity_index_names)
    ncols = 1
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=False,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    opacity = 0.3

    parameters = list(sb_dict.keys())
    colors = {
        parameter: np.random.randint(low=0, high=255, size=3)
        for parameter in parameters
    }
    for row, sensitivity_index_name in enumerate(sensitivity_index_names):
        for parameter, data in sb_dict.items():
            value = data[sensitivity_index_name][:, 0]
            lower = data[sensitivity_index_name][:, 1]
            upper = data[sensitivity_index_name][:, 2]
            fig.add_trace(
                go.Scatter(
                    x=convergence_iterations,
                    y=value,
                    mode="markers+lines",
                    opacity=1,
                    showlegend=False,
                    name="Parameter {}".format(parameter),
                    legendgroup="{}".format(parameter),
                    marker=dict(
                        color="rgba({},{},{},{})".format(
                            colors[parameter][0],
                            colors[parameter][1],
                            colors[parameter][2],
                            1,
                        ),
                    ),
                ),
                row=row + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=convergence_iterations,
                    y=lower,
                    mode="lines",
                    opacity=opacity,
                    showlegend=False,
                    legendgroup="{}".format(parameter),
                    marker=dict(
                        color="rgba({},{},{},{})".format(
                            colors[parameter][0],
                            colors[parameter][1],
                            colors[parameter][2],
                            opacity,
                        ),
                    ),
                    line=dict(width=0),
                ),
                row=row + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=convergence_iterations,
                    y=upper,
                    showlegend=False,
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba({},{},{},{})".format(
                        colors[parameter][0],
                        colors[parameter][1],
                        colors[parameter][2],
                        opacity,
                    ),
                    fill="tonexty",
                    legendgroup="{}".format(parameter),
                ),
                row=row + 1,
                col=1,
            )
        fig.update_yaxes(title_text=sensitivity_index_name, row=row + 1, col=1)

    fig.update_layout(
        width=800,
        height=400 * ncols,
        #     title_text="max conf. interval, and max difference of fscores among all inputs, bootstrap={}".format(num_bootstrap)
    )

    fig.update_xaxes(title_text="iterations")
    # write_pickle(fig, fig_name)
    fig.show()


def get_ci_convergence_per_parameter(
    stability_dict,
    ci_option="student",
    confidence_level=0.95,
    sensitivity_index_names=None,
    parameters=[],
):

    if ci_option == "normal":
        ci_function = ci_normal
    else:
        ci_function = ci_student  # Use student by default
    if len(parameters) == 0:
        num_params = list(list(stability_dict.values())[0].values())[0].shape[1]
        parameters = np.arange(num_params)
    if sensitivity_index_names is None:
        sensitivity_index_names = list(list(stability_dict.values())[0].keys())

    iterations_for_convergence = list(stability_dict.keys())
    SB_parameters = {
        "iterations": iterations_for_convergence,
    }
    for parameter in parameters:
        SB_parameters[parameter] = {
            si_name: np.zeros((0, 3)) for si_name in sensitivity_index_names
        }

    for sensitivity_index_name in sensitivity_index_names:
        for iterations in iterations_for_convergence:
            B_array = stability_dict[iterations][sensitivity_index_name]
            ci_dict = ci_function(B_array, confidence_level)
            for parameter in parameters:
                SB_parameters[parameter][sensitivity_index_name] = np.vstack(
                    [
                        SB_parameters[parameter][sensitivity_index_name],
                        np.array(
                            [
                                ci_dict["value"][parameter],
                                ci_dict["lower"][parameter],
                                ci_dict["upper"][parameter],
                            ]
                        ),
                    ]
                )
    return SB_parameters


def ci_student(B_array, confidence_level=0.95):
    """
    Student t-distribution confidence interval
    B_array : np.array
        Bootstrap array of size num_resamples x num_params
    """
    num_resamples = B_array.shape[0]
    degrees_of_freedom = num_resamples - 1
    t_alpha_2 = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
    means = np.mean(B_array, axis=0)
    interval_width = t_alpha_2 * np.std(B_array, axis=0)
    ci_dict = {
        "value": means,
        "lower": means - interval_width,
        "upper": means + interval_width,
    }
    return ci_dict


def ci_normal(B_array, confidence_level=0.95):
    """Normal confidence interval."""
    z_alpha_2 = get_z_alpha_2(confidence_level)
    means = np.mean(B_array, axis=0)
    interval_width = z_alpha_2 * np.std(B_array, axis=0)
    ci_dict = {
        "value": means,
        "lower": means - interval_width,
        "upper": means + interval_width,
    }
    return ci_dict
