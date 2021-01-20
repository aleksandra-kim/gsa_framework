import numpy as np
from scipy import stats
from pathlib import Path

from .utils import get_z_alpha_2, read_pickle, write_pickle

CI_THRESHOLD = 0.05


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
        "means": means,
        "width": interval_width,
    }
    return ci_dict


def ci_normal(B_array, confidence_level=0.95):
    """Normal confidence interval."""
    z_alpha_2 = get_z_alpha_2(confidence_level)
    means = np.mean(B_array, axis=0)
    interval_width = z_alpha_2 * np.std(B_array, axis=0)
    ci_dict = {
        "means": means,
        "width": interval_width,
    }
    return ci_dict


def get_ci_max_width(sb_dict):
    """get max confidence intervals among all parameters, aka worst-case scenario"""
    sensitivity_index_names = list(list(sb_dict.values())[0].keys())
    data = {}
    for b_dict in sb_dict.values():
        for sensitivity_index_name in sensitivity_index_names:
            B_array = b_dict[sensitivity_index_name]
            B_mean = np.mean(B_array, axis=0)
            B_std = np.std(B_array, axis=0)
            B_array_normalized = (B_array - B_mean) / B_std
            ci_width = np.max(B_array_normalized, axis=1) - np.min(
                B_array_normalized, axis=1
            )
            data[sensitivity_index_name] = np.vstack(
                [
                    -ci_width / 2,
                    +ci_width / 2,
                ]
            ).T
    return data


class Stability:
    def __init__(self, stability_dicts, write_dir, **kwargs):

        self.stability_dicts = stability_dicts
        self.write_dir = Path(write_dir)
        self.make_dirs()
        self.ci_type = kwargs.get("ci_type", "student")
        self.confidence_level = kwargs.get("confidence_level", 0.95)
        self.sa_stability_dict = self.get_sa_stability_dict(self.stability_dicts)
        self.sa_names = list(self.sa_stability_dict.keys())
        self.confidence_intervals = self.get_confidence_intervals(
            self.sa_stability_dict
        )
        self.confidence_intervals_max = self.get_confidence_intervals_max()
        self.rankings = self.get_rankings(self.sa_stability_dict)

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["figures", "stability"]  # TODO maybe add loggin later on
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_ranking_filepath(self, sa_name):
        filename = "ranking.{}.steps{}.pickle".format(
            sa_name, len(self.sa_stability_dict[sa_name]["iterations"])
        )
        return self.write_dir / "stability" / filename

    def get_sa_stability_dict(self, stability_dicts):
        sa_stability_dict = {}
        for stability_dict in stability_dicts:
            sa_names_current = list(list(stability_dict.values())[0].keys())
            sa_iteration_steps = np.array(list(stability_dict.keys()))
            sa_stability_dict.update(
                {
                    sa_name: {
                        "iterations": sa_iteration_steps,
                        "bootstrap": [],
                    }
                    for sa_name in sa_names_current
                }
            )
            for data in stability_dict.values():
                for sa_name in sa_names_current:
                    sa_stability_dict[sa_name]["bootstrap"].append(data[sa_name])
        return sa_stability_dict

    def get_confidence_intervals(self, sa_stability_dict):
        if self.ci_type == "normal":
            get_ci = ci_normal
        else:
            get_ci = ci_student
        confidence_intervals = {}
        for sa_name, data in sa_stability_dict.items():
            B_list = data["bootstrap"]
            confidence_intervals[sa_name] = []
            for B_array in B_list:
                confidence_intervals[sa_name].append(
                    get_ci(B_array, self.confidence_level)
                )
        return confidence_intervals

    def get_confidence_intervals_max(self):
        confidence_intervals_max = {}
        for sa_name, list_ in self.confidence_intervals.items():
            confidence_intervals_max[sa_name] = {
                "iterations": self.sa_stability_dict[sa_name]["iterations"],
                "width": [],
            }
            for data in list_:
                confidence_intervals_max[sa_name]["width"].append(max(data["width"]))
            confidence_intervals_max[sa_name]["width"] = np.array(
                confidence_intervals_max[sa_name]["width"]
            )
        return confidence_intervals_max

    def get_rankings(self, sa_stability_dict):
        rankings = {}
        for sa_name, data in sa_stability_dict.items():
            B_list = data["bootstrap"]
            rankings[sa_name] = {
                "iterations": self.sa_stability_dict[sa_name]["iterations"],
                "ranks": [],
            }
            for B_array in B_list:
                ranks = np.zeros((0, B_array.shape[1]), dtype=int)
                for array in B_array:
                    ranks = np.vstack([ranks, np.argsort(array)[-1::-1]])
                rankings[sa_name]["ranks"].append(ranks)
        return rankings

    def stat_ranking(self, ranks, sindices):
        num_bootstrap, num_params = ranks.shape
        rho = np.zeros(num_bootstrap * (num_bootstrap - 1) // 2)
        jk = 0
        for j in range(num_bootstrap):
            for k in range(j + 1, num_bootstrap):
                diff = np.abs(ranks[j, :] - ranks[k, :])
                maxs2 = np.max(sindices[[j, k], :], axis=0) ** 2
                rho[jk] = sum(diff * maxs2) / sum(maxs2)
                jk += 1
        return rho

    def stat_ranking_all_steps(self):
        ranking_stats = {}
        for sa_name in self.sa_names:
            filepath_rho = self.create_ranking_filepath(sa_name)
            if filepath_rho.exists():
                rho = read_pickle(filepath_rho)
            else:
                ranks = self.rankings[sa_name]["ranks"]
                sindices = self.sa_stability_dict[sa_name]["bootstrap"]
                num_bootstrap = ranks[0].shape[0]
                rho = np.zeros((0, num_bootstrap * (num_bootstrap - 1) // 2))
                for i in range(len(ranks)):
                    ranks_i = ranks[i]
                    sindices_i = sindices[i]
                    rho = np.vstack([rho, self.stat_ranking(ranks_i, sindices_i)])
                write_pickle(rho, filepath_rho)
            ranking_stats[sa_name] = {
                "iterations": self.sa_stability_dict[sa_name]["iterations"],
                "q95": np.percentile(rho, 95, axis=1),
                "q05": np.percentile(rho, 5, axis=1),
                "mean": np.mean(rho, axis=1),
            }
        return ranking_stats


# def plot_confidence_convergence(sb_dict, sensitivity_index_names=None):
#     sb_dict_ = deepcopy(sb_dict)
#     convergence_iterations = sb_dict_["iterations"]
#     sb_dict_.pop("iterations")
#     if sensitivity_index_names is None:
#         sensitivity_index_names = list(list(sb_dict_.values())[0].keys())
#
#     # Plotting
#     nrows = len(sensitivity_index_names)
#     ncols = 1
#     fig = make_subplots(
#         rows=nrows,
#         cols=ncols,
#         shared_yaxes=False,
#         shared_xaxes=True,
#         vertical_spacing=0.05,
#     )
#     opacity = 0.3
#
#     parameters = list(sb_dict_.keys())
#     colors = {
#         parameter: np.random.randint(low=0, high=255, size=3)
#         for parameter in parameters
#     }
#     for row, sensitivity_index_name in enumerate(sensitivity_index_names):
#         for parameter, data in sb_dict_.items():
#             value = data[sensitivity_index_name][:, 0]
#             lower = data[sensitivity_index_name][:, 1]
#             upper = data[sensitivity_index_name][:, 2]
#             fig.add_trace(
#                 go.Scatter(
#                     x=convergence_iterations,
#                     y=value,
#                     mode="markers+lines",
#                     opacity=1,
#                     showlegend=False,
#                     name="Parameter {}".format(parameter),
#                     legendgroup="{}".format(parameter),
#                     marker=dict(
#                         color="rgba({},{},{},{})".format(
#                             colors[parameter][0],
#                             colors[parameter][1],
#                             colors[parameter][2],
#                             1,
#                         ),
#                     ),
#                 ),
#                 row=row + 1,
#                 col=1,
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=convergence_iterations,
#                     y=lower,
#                     mode="lines",
#                     opacity=opacity,
#                     showlegend=False,
#                     legendgroup="{}".format(parameter),
#                     marker=dict(
#                         color="rgba({},{},{},{})".format(
#                             colors[parameter][0],
#                             colors[parameter][1],
#                             colors[parameter][2],
#                             opacity,
#                         ),
#                     ),
#                     line=dict(width=0),
#                 ),
#                 row=row + 1,
#                 col=1,
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=convergence_iterations,
#                     y=upper,
#                     showlegend=False,
#                     line=dict(width=0),
#                     mode="lines",
#                     fillcolor="rgba({},{},{},{})".format(
#                         colors[parameter][0],
#                         colors[parameter][1],
#                         colors[parameter][2],
#                         opacity,
#                     ),
#                     fill="tonexty",
#                     legendgroup="{}".format(parameter),
#                 ),
#                 row=row + 1,
#                 col=1,
#             )
#         fig.update_yaxes(title_text=sensitivity_index_name, row=row + 1, col=1)
#
#     fig.update_layout(
#         width=800,
#         height=400 * ncols,
#         #     title_text="max conf. interval, and max difference of fscores among all inputs, bootstrap={}".format(num_bootstrap)
#     )
#
#     fig.update_xaxes(title_text="iterations")
#     # write_pickle(fig, fig_name)
#     fig.show()
