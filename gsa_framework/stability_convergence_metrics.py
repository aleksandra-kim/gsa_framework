import numpy as np

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

from .utils import get_z_alpha_2


def get_stability_arrs(stability_dict, confidence_level=0.95):
    num_params = list(stability_dict.values())[0].shape[1]
    iter_ = np.array(list(stability_dict.keys()))
    mean_ = np.zeros((len(stability_dict), num_params))
    mins_ = np.zeros((len(stability_dict), num_params))
    maxs_ = np.zeros((len(stability_dict), num_params))
    if confidence_level is None:
        # use absolute values
        for i, vals in enumerate(stability_dict.values()):
            mean_[i, :] = np.mean(vals, axis=0)
            mins_[i, :] = np.min(vals, axis=0)
            maxs_[i, :] = np.max(vals, axis=0)
    else:
        # use confidence intervals
        z_alpha_2 = get_z_alpha_2(confidence_level)
        for i, vals in enumerate(stability_dict.values()):
            mean_[i, :] = np.mean(vals, axis=0)
            mins_[i, :] = mean_[i, :] - z_alpha_2 * np.std(vals, axis=0)
            maxs_[i, :] = mean_[i, :] + z_alpha_2 * np.std(vals, axis=0)
    return iter_, mean_, mins_, maxs_


def min_max_bands(mean_, mins_abs, maxs_abs):
    # Absolute width min-max bands
    abs_width = maxs_abs - mins_abs
    inds = np.zeros(len(abs_width), dtype=int)
    band_min = np.zeros(len(abs_width))
    band_max = np.zeros(len(abs_width))
    for i in range(len(abs_width)):
        inds[i] = np.where(abs_width[i, :] == np.max(abs_width[i, :]))[0][0]
        band_min[i] = mean_[i, :][inds[i]] - mins_abs[i, :][inds[i]]
        band_max[i] = -mean_[i, :][inds[i]] + maxs_abs[i, :][inds[i]]
    return band_min, band_max


def min_max_band(stability_dict):
    iterations = np.array(list(stability_dict.keys()))

    return
