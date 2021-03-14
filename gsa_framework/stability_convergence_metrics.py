import numpy as np
from scipy import stats
from pathlib import Path
from scipy.stats import spearmanr
import jenkspy
from copy import deepcopy

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
    interval_width = t_alpha_2 * np.std(B_array, axis=0)
    return interval_width


def ci_normal(B_array, confidence_level=0.95):
    """Normal confidence interval."""
    z_alpha_2 = get_z_alpha_2(confidence_level)
    interval_width = z_alpha_2 * np.std(B_array, axis=0)
    return interval_width


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


def rho1_jk(Rj, Rk):
    """Spearman correlation"""
    rho = spearmanr(Rj, Rk)[0]
    return rho


def rho2_jk(Rj, Rk):
    """Weighted Spearman correlation"""
    numerator = (Rj - Rk) ** 2 * (1 / (Rj + 1) + 1 / (Rk + 1))
    rho = 1 - 2 * np.sum(numerator) / np.max(numerator)
    return rho


def rho3_jk(Rj, Rk):
    """Weighted Spearman correlation"""
    numerator = (Rj - Rk) ** 2 / (Rj + 1) / (Rk + 1)
    rho = 1 - 2 * np.sum(numerator) / np.max(numerator)
    return rho


def rho4_jk(Rj, Rk):
    """Weighted Spearman correlation"""
    numerator = (Rj - Rk) ** 2 / (Rj + 1 + Rk + 1)
    rho = 1 - 2 * np.sum(numerator) / np.max(numerator)
    return rho


def rho5_jk(Rj, Rk):
    """Correlation coefficient computed on Savage scores"""
    M = len(Rj)
    SSarr = 1 / np.arange(1, M + 1)
    SSsum = np.cumsum(SSarr[::-1])[::-1]
    SSj = SSsum[Rj]
    SSk = SSsum[Rk]
    numerator = (SSj - SSk) ** 2
    rho = 1 - np.sum(numerator) / 2 / (M - SSsum[0])
    return rho


def rho6_jk(Rj, Rk, Sj, Sk):
    diff = np.abs(Rj - Rk)
    Sjk = np.vstack([Sj, Sk])
    maxs2 = np.max(Sjk, axis=0) ** 2
    rho = sum(diff * maxs2) / sum(maxs2)
    return rho


def compute_spearmanr(mat, vec):
    """
    Spearmanr between each row of matrix `mat` and vector `vec`. Takes into account the case when some rows in mat
    have just one unique element.
    """
    rho = np.zeros(len(mat))
    rho[:] = np.nan
    skip_inds = np.where(np.array([len(set(r)) for r in mat]) == 1)[0]
    incl_inds = np.setdiff1d(np.arange(len(mat)), skip_inds)
    if len(incl_inds) > 0:
        rho_temp, _ = spearmanr(mat[incl_inds, :].T, vec)
        rho_temp = rho_temp[-1, :-1]
        rho[incl_inds] = rho_temp
    return rho


#######################
### Stability class ###
#######################
class Stability:
    def __init__(self, stability_dicts, write_dir, **kwargs):

        self.stability_dicts = self.remove_nans(stability_dicts)
        self.write_dir = Path(write_dir)
        self.make_dirs()
        self.ci_type = kwargs.get("ci_type", "student")
        self.confidence_level = kwargs.get("confidence_level", 0.95)
        self.num_ranks = kwargs.get("num_ranks", 10)
        self.bootstrap_ranking_tag = kwargs.get("bootstrap_ranking_tag")
        self.q_min = kwargs.get("q_min", 5)
        self.q_max = kwargs.get("q_max", 95)
        self.num_params = list(list(self.stability_dicts[0].values())[0].values())[
            0
        ].shape[1]
        (
            self.sa_names,
            self.iterations,
            self.bootstrap_data,
            self.sa_mean_results,
        ) = self.get_data_from_stability_dicts(self.stability_dicts)
        self.confidence_intervals = self.get_confidence_intervals(
            self.bootstrap_data, self.ci_type
        )
        self.confidence_intervals_max = self.get_confidence_intervals_max(
            self.confidence_intervals
        )
        # self.rankings_convergence = self.get_rankings_convergence_to_last(self.sa_mean_results, num_ranks=self.num_ranks)
        self.bootstrap_rankings = self.get_bootstrap_rankings(
            self.bootstrap_data,
            self.sa_mean_results,
            self.bootstrap_ranking_tag,
            self.num_ranks,
        )
        self.bootstrap_rankings_width_percentiles = (
            self.get_bootstrap_rankings_width_percentiles(
                self.bootstrap_rankings,
                q_min=self.q_min,
                q_max=self.q_max,
            )
        )
        self.stat_medians = self.get_stat_medians(self.bootstrap_data)

    def remove_nans(self, stability_dicts):
        for stability_dict in stability_dicts:
            where = []
            for k, v in stability_dict.items():
                for array in v.values():
                    if np.isnan(array).any():
                        where.append(k)
                        break
            [stability_dict.pop(k) for k in where]
        return stability_dicts

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = ["figures", "stability"]  # TODO maybe add loggin later on
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_bootstrap_rankings_filepath(
        self, num_ranks, tag, sa_name, num_bootstrap
    ):
        filename = "ranking{}.{}.{}.bootstrap{}.steps{}.pickle".format(
            num_ranks, tag, sa_name, num_bootstrap, len(self.iterations[sa_name])
        )
        return self.write_dir / "stability" / filename

    def get_data_from_stability_dicts(self, stability_dicts):
        iterations, bootstrap_data, sa_mean_results = {}, {}, {}
        for stability_dict in stability_dicts:
            sa_names_current = list(list(stability_dict.values())[0].keys())
            sa_iteration_steps = np.array(list(stability_dict.keys()))
            for sa_name in sa_names_current:
                iterations.update({sa_name: sa_iteration_steps})
                bootstrap_data.update({sa_name: []})
            for data in stability_dict.values():
                for sa_name in sa_names_current:
                    bootstrap_data[sa_name].append(data[sa_name])
        sa_names = []
        for sa_name, list_ in bootstrap_data.items():
            if "stat." in sa_name:
                ydim = 1
            else:
                ydim = self.num_params
                sa_names.append(sa_name)
            means = np.zeros((0, ydim))
            means[:] = np.nan
            for data in list_:
                means = np.vstack([means, np.mean(data, axis=0)])
            sa_mean_results[sa_name] = means
        return sa_names, iterations, bootstrap_data, sa_mean_results

    def get_confidence_intervals(self, bootstrap_data, ci_type="student"):
        if ci_type == "normal":
            get_ci = ci_normal
        else:
            get_ci = ci_student
        confidence_intervals = {}
        for sa_name, data in bootstrap_data.items():
            confidence_intervals_arr = np.zeros((0, data[0].shape[1]))
            confidence_intervals_arr[:] = np.nan
            for B_array in data:
                confidence_intervals_arr = np.vstack(
                    [confidence_intervals_arr, get_ci(B_array, self.confidence_level)]
                )
            confidence_intervals[sa_name] = confidence_intervals_arr
        return confidence_intervals

    def get_confidence_intervals_max(self, confidence_intervals):
        confidence_intervals_max = {}
        for sa_name, data in confidence_intervals.items():
            confidence_intervals_max[sa_name] = np.max(data, axis=1)
        return confidence_intervals_max

    def get_bootstrap_rankings(
        self, bootstrap_data, sa_mean_results, tag, num_ranks=10
    ):
        bootstrap_rankings = {}
        for sa_name in self.sa_names:
            num_bootstrap = bootstrap_data[sa_name][0].shape[0]
            filepath_bootstrap_rankings = self.create_bootstrap_rankings_filepath(
                num_ranks,
                tag,
                sa_name,
                num_bootstrap,
            )
            if filepath_bootstrap_rankings.exists():
                bootstrap_rankings_arr = read_pickle(filepath_bootstrap_rankings)
            else:
                bootstrap_rankings_arr = np.zeros((0, num_bootstrap))
                bootstrap_rankings_arr[:] = np.nan
                if sa_name == "total_gain":
                    means = self.bootstrap_data[sa_name][-1][0, :]
                else:
                    means = sa_mean_results[sa_name][-1, :]
                breaks = jenkspy.jenks_breaks(means, nb_class=num_ranks)
                mean_ranking = self.get_one_clustered_ranking(means, num_ranks, breaks)
                for i in range(len(self.iterations[sa_name])):
                    bootstrap_data_sa = bootstrap_data[sa_name][i]
                    rankings = np.zeros((0, self.num_params))
                    for data in bootstrap_data_sa:
                        rankings = np.vstack(
                            [
                                rankings,
                                self.get_one_clustered_ranking(data, num_ranks, breaks),
                            ]
                        )
                    rho = compute_spearmanr(rankings, mean_ranking)
                    bootstrap_rankings_arr = np.vstack([bootstrap_rankings_arr, rho])
                write_pickle(bootstrap_rankings_arr, filepath_bootstrap_rankings)
            bootstrap_rankings[sa_name] = bootstrap_rankings_arr
        return bootstrap_rankings

    def get_bootstrap_rankings_width_percentiles(
        self, bootstrap_rankings, q_min=5, q_max=95
    ):
        bootstrap_rankings_width_percentiles = {}
        for sa_name in self.sa_names:
            data = bootstrap_rankings[sa_name]
            min_ = np.percentile(data, q_min, axis=1)
            max_ = np.percentile(data, q_max, axis=1)
            median = np.percentile(data, 50, axis=1)
            mean = np.mean(data, axis=1)
            confidence_interval = ci_student(data.T)
            bootstrap_rankings_width_percentiles[sa_name] = {
                "q_max": max_,
                "q_min": min_,
                "median": median,
                "mean": mean,
                "confidence_interval": confidence_interval,
            }
        return bootstrap_rankings_width_percentiles

    def get_rankings_convergence_to_last(self, sa_mean_results, num_ranks=10):
        ranking_convergence = {}
        for sa_name in self.sa_names:
            means = sa_mean_results[sa_name]
            breaks = jenkspy.jenks_breaks(means[-1, :], nb_class=num_ranks)
            rankings = np.zeros((0, self.num_params))
            for means_ in means:
                rankings = np.vstack(
                    [
                        rankings,
                        self.get_one_clustered_ranking(means_, num_ranks, breaks),
                    ]
                )
            rho = compute_spearmanr(rankings[:-1, :], rankings[-1, :])
            ranking_convergence[sa_name] = rho
        return ranking_convergence

    def get_one_clustered_ranking(self, array, num_ranks, breaks=None):
        if breaks is None:
            breaks = jenkspy.jenks_breaks(array, nb_class=num_ranks)
        breaks = deepcopy(breaks)
        breaks[0] = array.min()
        breaks[-1] = array.max()
        clustered_ranking = np.zeros(len(array))
        clustered_ranking[:] = np.nan
        where_dict = {}
        for b in range(num_ranks):
            where = np.where(np.logical_and(array >= breaks[b], array < breaks[b + 1]))[
                0
            ]
            if b == num_ranks - 1:
                where = np.where(
                    np.logical_and(array >= breaks[b], array <= breaks[b + 1])
                )[0]
            clustered_ranking[where] = num_ranks - b
            where_dict[b] = where
        return clustered_ranking

    def get_stat_medians(self, bootstrap_data):
        stat_medians = {}
        for k, v in bootstrap_data.items():
            if "stat." in k:
                medians = []
                for data in v:
                    medians.append(np.percentile(data, 50))
                stat_medians[k] = np.array(medians)
        return stat_medians

    # def get_bootstrap_rankings(self, bootstrap_data, sa_mean_results, tag, num_ranks=10):
    #     bootstrap_rankings = {}
    #     for sa_name in self.sa_names:
    #         num_bootstrap  = bootstrap_data[sa_name][0].shape[0]
    #         filepath_bootstrap_rankings = self.create_bootstrap_rankings_filepath(
    #             num_ranks, tag, sa_name, num_bootstrap,
    #         )
    #         if filepath_bootstrap_rankings.exists():
    #             bootstrap_rankings_arr = read_pickle(filepath_bootstrap_rankings)
    #         else:
    #             bootstrap_rankings_arr = np.zeros((0, num_bootstrap))
    #             bootstrap_rankings_arr[:] = np.nan
    #             for i in range(len(self.iterations[sa_name])):
    #                 means = sa_mean_results[sa_name][i]
    #                 breaks = jenkspy.jenks_breaks(means, nb_class=num_ranks)
    #                 mean_ranking = self.get_one_clustered_ranking(means, num_ranks, breaks)
    #                 bootstrap_data_sa = bootstrap_data[sa_name][i]
    #                 rankings = np.zeros((0, self.num_params))
    #                 for data in bootstrap_data_sa:
    #                     rankings = np.vstack(
    #                         [
    #                             rankings,
    #                             self.get_one_clustered_ranking(data, num_ranks, breaks)
    #                         ]
    #                     )
    #                 rho = compute_spearmanr(rankings, mean_ranking)
    #                 bootstrap_rankings_arr  =  np.vstack([bootstrap_rankings_arr, rho])
    #             write_pickle(bootstrap_rankings_arr, filepath_bootstrap_rankings)
    #         bootstrap_rankings[sa_name] = bootstrap_rankings_arr
    #     return bootstrap_rankings
