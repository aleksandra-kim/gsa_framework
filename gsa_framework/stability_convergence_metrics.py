import numpy as np

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

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
