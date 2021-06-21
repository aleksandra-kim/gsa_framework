import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gsa_framework.utils import read_pickle
from dev.utils_paper_plotting import *
from gsa_framework.models import Morris4


def get_files_dict_sorted(files):
    def get_file(filename):
        temp = [f for f in files if filename in f.name]
        if len(temp) != 1:
            print(temp)
        assert len(temp) == 1
        return temp[0]

    files_dict = {
        "corr": get_file("corr"),
        "salt": get_file("salt"),
        "delt": get_file("delt"),
        "xgbo": get_file("xgbo"),
    }
    return files_dict


if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )
    # write_dir_fig = path_base / "paper_figures"
    write_dir_fig = path_base / "lea_figures"
    nums_params = [1000, 5000, 10000]
    num_ranks = 4
    morris_model_names = {
        1000: r"$\underline{\text{Morris model, 1'000 inputs}}$",
        5000: r"$\underline{\text{Morris model, 5'000 inputs}}$",
        10000: r"$\underline{\text{Morris model, 10'000 inputs}}$",
    }
    all_gsa_names = [v["name"] for v in sa_plot.values()]

    filepath_gsa_dict, filepath_stability_dict = {}, {}
    for num_params in nums_params:
        write_dir = path_base / "{}_morris4".format(num_params)
        write_dir_arr = write_dir / "arrays"
        # GSA results
        files = [
            x
            for x in write_dir_arr.iterdir()
            if x.is_file() and "S." in x.name and "stability" not in x.name
        ]
        files_dict = get_files_dict_sorted(files)
        filepath_gsa_dict.update(
            {
                num_params: {
                    "corr": (files_dict["corr"], "spearman"),
                    "salt": (files_dict["salt"], "Total order"),
                    "delt": (files_dict["delt"], "delta"),
                    "xgbo": (files_dict["xgbo"], "total_gain"),
                },
            }
        )
        # Stability dictionaries
        files_stability = [
            x
            for x in write_dir_arr.iterdir()
            if x.is_file() and "S." in x.name and "stability" in x.name
        ]
        files_stability_dict = get_files_dict_sorted(files_stability)
        filepath_stability_dict.update(
            {
                num_params: {
                    "corr": (files_stability_dict["corr"], "spearman"),
                    "salt": (files_stability_dict["salt"], "Total order"),
                    "delt": (files_stability_dict["delt"], "delta"),
                    "xgbo": (files_stability_dict["xgbo"], "total_gain"),
                },
            }
        )
    # Read GSA files
    S_dict_all = {}
    S_arr_all = {}
    for num_params in nums_params:
        S_arr = np.zeros((0, num_params))
        S_dict = {}
        filepath_gsa_dict_ = filepath_gsa_dict[num_params]
        for k, v in filepath_gsa_dict_.items():
            S_dict[k] = read_pickle(v[0])[v[1]]
            S_arr = np.vstack([S_arr, S_dict[k]])
        S_dict_all[num_params] = S_dict
        S_arr_all[num_params] = S_arr

    # Read stability files
    # st_classes = {}
    # for num_params in nums_params:
    #     stability_dicts = []
    #     write_dir = path_base / "{}_morris4".format(num_params)
    #     for k, v in filepath_stability_dict[num_params].items():
    #         stability_dict = read_pickle(v[0])
    #         stability_dicts.append(stability_dict)
    #     st_classes[num_params] = Stability(
    #         stability_dicts,
    #         write_dir,
    #         num_ranks=num_ranks,
    #         bootstrap_ranking_tag="paper1",
    #     )

    fig_format = ["pdf"]
    opacity = 0.6
    sa_names = {
        "spearman": "corr",
        "total": "salt",
        "delta": "delt",
        "total_gain": "xgbo",
    }

    morris_models = {}
    for num_params in nums_params:
        num_influential = num_params // 100
        model = Morris4(num_params=num_params, num_influential=num_influential)
        morris_models[num_params] = model

    ### 1. Scalability of sensitivity_analysis, results from all GSA
    ###################################################
    # region
    #
    # option = "zoomed_in"
    # if option=="zoomed_in":
    #     shared_yaxes=False
    # else:
    #     shared_yaxes=True
    #
    # fig = make_subplots(
    #     rows=4,
    #     cols=3,
    #     shared_xaxes=True,
    #     shared_yaxes=shared_yaxes,
    #     vertical_spacing=0.12,
    #     horizontal_spacing=0.05,
    #     row_heights=[0.25,0.25,0.25,0.25],
    #     subplot_titles=[
    #         "", all_gsa_names[0], "",
    #         "", all_gsa_names[1], "",
    #         "", all_gsa_names[2], "",
    #         "", all_gsa_names[3], "",
    #     ]
    # )
    # col=1
    # ipos = 0
    # showlegend, showlegend2 = True, True
    # for num_params in nums_params:
    #     if option == "zoomed_in":
    #         num_params_plot = num_params // 10
    #         marker_size = 2
    #     else:
    #         num_params_plot = num_params
    #         marker_size = 1
    #
    #     S_dict = S_dict_all[num_params]
    #     row = 1
    #     for k,v in S_dict.items():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=np.arange(num_params)[:num_params_plot],
    #                 y=v[:num_params_plot],
    #                 mode="markers",
    #                 opacity=1,
    #                 marker=dict(size=2, color=color_blue_rgb),
    #                 showlegend=showlegend,
    #                 name=r"$\text{Estimates of sensitivity indices}$",
    #             ),
    #             row=row,
    #             col=col,
    #         )
    #         showlegend = False
    #         if k=='salt':
    #             model = morris_models[num_params]
    #             analytical_total = model.get_sensitivity_indices()['Total order']
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=np.arange(num_params)[:num_params_plot],
    #                     y=analytical_total[:num_params_plot],
    #                     mode='markers',
    #                     marker=dict(size=marker_size,color=color_orange_rgb,symbol="diamond-wide"),
    #                     showlegend=showlegend2,
    #                     name=r"$\text{Analytical sensitivity indices}$",
    #                 ),
    #             row = row,
    #             col = col,
    #             )
    #             showlegend2=False
    #         if col==1:
    #             fig.update_yaxes(title_text=sa_plot[k]['notation'], row=row, col=col)
    #         if row==1:
    #             fig.add_annotation(
    #                 x=1/6*(col+ipos),
    #                 y=1.13,  # annotation point
    #                 xref="paper",
    #                 yref="paper",
    #                 text=morris_model_names[num_params],
    #                 showarrow=False,
    #                 xanchor="center",
    #                 font=dict(
    #                     size=16,
    #                 )
    #             )
    #             ipos += 1
    #         row += 1
    #     fig.update_xaxes(title_text=r"$\text{Model inputs}$", row=row-1, col=col)
    #     col += 1
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_gray_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_layout(
    #     width=1100, height=600,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     margin=dict(l=0, r=0, t=70, b=10),
    #     legend=dict(
    #         x=0.5,
    #         y=-0.12,
    #         xanchor='center',
    #         font_size=14,
    #         orientation='h',
    #         itemsizing='constant',
    #     )
    # )
    # fig.show()
    # if option == "zoomed_in":
    #     save_fig(fig, "morris_all_gsa_results_zoomed_in", fig_format, write_dir_fig)
    # else:
    #     save_fig(fig, "morris_all_gsa_results", fig_format, write_dir_fig)

    # endregion

    ### 2. Convergence and stability of confidence intervals
    ########################################################
    # region
    #
    # option = "zoomed_in"
    # if option == "zoomed_in":
    #     shared_xaxes = False
    #     start_iterations = 1
    # else:
    #     shared_xaxes = True
    #     start_iterations = 0
    #
    # fig = make_subplots(
    #     rows=4,
    #     cols=3,
    #     shared_xaxes=shared_xaxes,
    #     shared_yaxes=False,
    #     vertical_spacing=0.12,
    #     horizontal_spacing=0.05,
    #     subplot_titles=[
    #         "", all_gsa_names[0], "",
    #         "", all_gsa_names[1], "",
    #         "", all_gsa_names[2], "",
    #         "", all_gsa_names[3], "",
    #     ]
    # )
    #
    # col = 1
    # ipos = 0
    # showlegend = True
    # for num_params in nums_params:
    #     st = st_classes[num_params]
    #     thetas = np.linspace(0.01, 0.95, 100)
    #     all_iterations = st.iterations['spearman']
    #     analytical_spearman_ci = []
    #     for iterations in all_iterations:
    #         list_ = []
    #         for theta in thetas:
    #             list_.append(get_corrcoef_interval_width(theta, iterations=iterations)['spearman'])
    #         analytical_spearman_ci.append(max(list_))
    #     analytical_spearman_ci = np.array(analytical_spearman_ci)
    #
    #     row = 1
    #     for sa_name in sa_names.keys():
    #         if sa_name == "total" and option == "zoomed_in":
    #             start_iterations_ = 6
    #         elif sa_name == "total" and option == "nzoomed_in":
    #             start_iterations_ = 1
    #         else:
    #             start_iterations_ = start_iterations
    #         x = st.iterations[sa_name][start_iterations_:]
    #         y = np.zeros(len(x))
    #         if sa_name == 'spearman':
    #             color = color_orange_tuple
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=x,
    #                     y=analytical_spearman_ci[start_iterations_:]/2,
    #                     mode="lines",
    #                     opacity=1,
    #                     showlegend=showlegend,
    #                     marker=dict(
    #                         color="rgba({},{},{},{})".format(
    #                             color[0],
    #                             color[1],
    #                             color[2],
    #                             1,
    #                         ),
    #                     ),
    #                     name=r"$\text{Analytical confidence intervals}$",
    #                     line=dict(dash='dot'),
    #                 ),
    #                 row=row,
    #                 col=col,
    #             )
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=x,
    #                     y=-analytical_spearman_ci[start_iterations_:]/2,
    #                     mode="lines",
    #                     opacity=1,
    #                     showlegend=False,
    #                     marker=dict(
    #                         color="rgba({},{},{},{})".format(
    #                             color[0],
    #                             color[1],
    #                             color[2],
    #                             1,
    #                         ),
    #                     ),
    #                     line=dict(dash='dot'),
    #                 ),
    #                 row=row,
    #                 col=col,
    #             )
    #         width = st.confidence_intervals_max[sa_name][start_iterations_:]
    #         lower = y - width / 2
    #         upper = y + width / 2
    #         color = color_blue_tuple
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x,
    #                 y=y,
    #                 mode="lines",
    #                 opacity=1,
    #                 showlegend=showlegend,
    #                 marker=dict(
    #                     color="rgba({},{},{},{})".format(
    #                         color[0],
    #                         color[1],
    #                         color[2],
    #                         1,
    #                     ),
    #                 ),
    #                 name=r"$\text{Bootstrap confidence intervals}$",
    #             ),
    #             row=row,
    #             col=col,
    #         )
    #         showlegend = False
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x,
    #                 y=lower,
    #                 mode="lines",
    #                 opacity=opacity,
    #                 showlegend=False,
    #                 marker=dict(
    #                     color="rgba({},{},{},{})".format(
    #                         color[0],
    #                         color[1],
    #                         color[2],
    #                         opacity,
    #                     ),
    #                 ),
    #                 line=dict(width=0),
    #             ),
    #             row=row,
    #             col=col,
    #         )
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x,
    #                 y=upper,
    #                 showlegend=False,
    #                 line=dict(width=0),
    #                 mode="lines",
    #                 fillcolor="rgba({},{},{},{})".format(
    #                     color[0],
    #                     color[1],
    #                     color[2],
    #                     opacity,
    #                 ),
    #                 fill="tonexty",
    #             ),
    #             row=row,
    #             col=col,
    #         )
    #
    #         if col == 1:
    #             fig.update_yaxes(title_text=sa_plot[sa_names[sa_name]]['stat_indices'], row=row, col=col)
    #         if row==1:
    #             fig.add_annotation(
    #                 x=1/6*(col+ipos),
    #                 y=1.13,  # annotation point
    #                 xref="paper",
    #                 yref="paper",
    #                 text=morris_model_names[num_params],
    #                 showarrow=False,
    #                 xanchor="center",
    #                 font=dict(
    #                     size=16,
    #                 )
    #             )
    #             ipos += 1
    #         row+=1
    #     fig.update_xaxes(title_text=r"$\text{Iterations}$", row=row - 1, col=col)
    #     col+=1
    # fig.update_layout(
    #     legend=dict(
    #         orientation='h',
    #         x=0.5,
    #         y=-0.12,
    #         xanchor='center',
    #         font_size=14,
    #     ),
    #     width=1100, height=600,
    #     margin=dict(l=0, r=0, t=70, b=10),
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    # )
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    #
    # # fig.show()
    # if option == "zoomed_in":
    #     save_fig(fig, "morris_stat_indices_zoomed_in", fig_format, write_dir_fig)
    # else:
    #     save_fig(fig, "morris_stat_indices", fig_format, write_dir_fig)
    #
    # endregion

    # region
    # from scipy.stats import spearmanr
    # def compute_spearmanr(mat, vec):
    #     """
    #     Spearmanr between each row of matrix `mat` and vector `vec`. Takes into account the case when some rows in mat
    #     have just one unique element.
    #     """
    #     rho = np.zeros(len(mat))
    #     rho[:] = np.nan
    #     skip_inds = np.where(np.array([len(set(r)) for r in mat]) == 1)[0]
    #     incl_inds = np.setdiff1d(np.arange(len(mat)), skip_inds)
    #     if len(incl_inds) > 0:
    #         rho_temp, _ = spearmanr(mat[incl_inds, :].T, vec)
    #         rho_temp = rho_temp[-1, :-1]
    #         rho[incl_inds] = rho_temp
    #     return rho
    #
    #
    # num_ranks = 4
    #
    # pathS = path_base / "10000_morris4" / "arrays" / "S.xgboostGsa_Lr0.2G0Mcw600Md2RegL0RegA0Ne1500Ss0.2Cbt0.2_.randomSampling.40000.3407.pickle"
    # S = read_pickle(pathS)
    # import jenkspy
    # st = st_classes[10000]
    # # means = st.sa_mean_results["total_gain"][-1, :]
    # R1_ind = 10
    # # means = st.bootstrap_data["total_gain"][-1][R1_ind,:]
    # means = S['total_gain']
    # breaks = jenkspy.jenks_breaks(means, nb_class=num_ranks)
    # R1 = st.get_one_clustered_ranking(means, num_ranks, breaks)
    # R2 = st.get_one_clustered_ranking(st.bootstrap_data["total_gain"][-1][0,:], num_ranks, breaks)
    # R = np.vstack([R1,R2])
    # rho = compute_spearmanr(R, R1)
    # fig = go.Figure()
    # freq, bins = np.histogram(means, 20)
    # fig.add_trace(
    #     go.Bar(
    #         x=bins,
    #         y=freq
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=breaks,
    #         y=np.zeros(len(breaks)),
    #         mode = 'markers',
    #         marker_color='red',
    #     )
    # )
    # fig.show()
    # endregion

    ### 3. Convergence of ranking
    #############################
    # region
    #
    # option = "nzoomed_in"
    # if option == "zoomed_in":
    #     shared_xaxes = False
    #     start_iterations = 1
    # else:
    #     shared_xaxes = True
    #     start_iterations = 0
    #
    # fig = make_subplots(
    #     rows=4,
    #     cols=3,
    #     shared_xaxes=shared_xaxes,
    #     shared_yaxes=False,
    #     vertical_spacing=0.12,
    #     horizontal_spacing=0.12,
    #     subplot_titles=[
    #         "", all_gsa_names[0], "",
    #         "", all_gsa_names[1], "",
    #         "", all_gsa_names[2], "",
    #         "", all_gsa_names[3], "",
    #     ],
    #     specs=[
    #         [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": True},  {"secondary_y": True},  {"secondary_y": True}],
    #     ],
    # )
    # plot_robustness_ranking = True
    #
    # ipos = 0
    # iposes = [0.1165, 0.466, 0.8155]
    # showlegend, showlegend2 = True, True
    # for col, num_params in enumerate(nums_params):
    #     col += 1
    #     st = st_classes[num_params]
    #     row = 1
    #     for sa_name in sa_names.keys():
    #         # y = st.bootstrap_rankings_width_percentiles[sa_name]['median'][:-1]
    #         # lower = st.bootstrap_rankings_width_percentiles[sa_name]['min'][:-1]
    #         # upper = st.bootstrap_rankings_width_percentiles[sa_name]['max'][:-1]
    #         if sa_name == 'total_gain':
    #             y=st.stat_medians["stat.r2"][:-1]
    #             width=st.confidence_intervals_max['stat.r2'][:-1]
    #             lower=y-width/2
    #             upper=y+width/2
    #             color = color_orange_tuple
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=st.iterations[sa_name][:-1],
    #                     y=y,
    #                     mode="lines",
    #                     marker=dict(color=color_orange_rgb),
    #                     showlegend=showlegend2,
    #                     name=r"$\text{Metrics within GSA sensitivity_analysis}$",
    #                 ),
    #                 col=col, row=row,
    #                 secondary_y=True,
    #             )
    #             showlegend2=False
    #             if plot_robustness_ranking:
    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=st.iterations[sa_name][:-1],
    #                         y=lower,
    #                         mode="lines",
    #                         opacity=opacity,
    #                         showlegend=False,
    #                         marker=dict(
    #                             color="rgba({},{},{},{})".format(
    #                                 color[0],
    #                                 color[1],
    #                                 color[2],
    #                                 opacity,
    #                             ),
    #                         ),
    #                         line=dict(width=0),
    #                     ),
    #                     row=row,
    #                     col=col,
    #                     secondary_y=True,
    #                 )
    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=st.iterations[sa_name][:-1],
    #                         y=upper,
    #                         showlegend=False,
    #                         line=dict(width=0),
    #                         mode="lines",
    #                         fillcolor="rgba({},{},{},{})".format(
    #                             color[0],
    #                             color[1],
    #                             color[2],
    #                             opacity,
    #                         ),
    #                         fill="tonexty",
    #                     ),
    #                     row=row,
    #                     col=col,
    #                     secondary_y=True,
    #                 )
    #             fig.update_yaxes(
    #                 title_text=r'$r^2$',
    #                 row=row, col=col, secondary_y=True,
    #                 color=color_orange_rgb,
    #                 title_standoff=5,
    #             )
    #             if option == "zoomed_in":
    #                 if num_params == 1000:
    #                     fig.update_yaxes(range=[-0.2, 1.1], row=row, col=col)
    #                 elif num_params == 5000:
    #                     fig.update_yaxes(range=[0.2, 1.1], row=row, col=col)
    #                 elif num_params == 10000:
    #                     fig.update_yaxes(range=[-0.85, 1.1], row=row, col=col)
    #
    #         y = st.bootstrap_rankings_width_percentiles[sa_name]["mean"][:-1]
    #         cf_width = st.bootstrap_rankings_width_percentiles[sa_name]["confidence_interval"][:-1]
    #         lower = y - cf_width / 2
    #         upper = y + cf_width / 2
    #         color = color_blue_tuple
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=st.iterations[sa_name][:-1],
    #                 y=y,
    #                 mode="lines",
    #                 marker=dict(color=color_blue_rgb),
    #                 showlegend=showlegend,
    #                 name=r"$\text{Convergence of ranking}$",
    #             ),
    #             col=col, row=row,
    #         )
    #         showlegend=False
    #         if plot_robustness_ranking:
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=st.iterations[sa_name][:-1],
    #                     y=lower,
    #                     mode="lines",
    #                     opacity=opacity,
    #                     showlegend=False,
    #                     marker=dict(
    #                         color="rgba({},{},{},{})".format(
    #                             color[0],
    #                             color[1],
    #                             color[2],
    #                             opacity,
    #                         ),
    #                     ),
    #                     line=dict(width=0),
    #                 ),
    #                 row=row,
    #                 col=col,
    #                 secondary_y=False,
    #             )
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=st.iterations[sa_name][:-1],
    #                     y=upper,
    #                     showlegend=False,
    #                     line=dict(width=0),
    #                     mode="lines",
    #                     fillcolor="rgba({},{},{},{})".format(
    #                         color[0],
    #                         color[1],
    #                         color[2],
    #                         opacity,
    #                     ),
    #                     fill="tonexty",
    #                 ),
    #                 row=row,
    #                 col=col,
    #                 secondary_y=False,
    #             )
    #
    #         if col == 1:
    #             fig.update_yaxes(
    #                 title_text=sa_plot[sa_names[sa_name]]['stat_ranking'],
    #                 row=row, col=1,
    #                 secondary_y=False,
    #             )
    #             if option == "nzoomed_in":
    #                 fig.update_yaxes(range=[0, 1.1],)
    #
    #             # fig.update_xaxes(
    #             #     range=[
    #             #         min(st.iterations['spearman']),
    #             #         max(st.iterations['total'])
    #             #     ],
    #             #     row=row,
    #             #     col=col,
    #             # )
    #             # fig.add_annotation(
    #             #     x=0.5,
    #             #     y=(1 - 0.16) / 3 * (4 - row) + 0.16 + 0.02,  # annotation point
    #             #     xref="paper",
    #             #     yref="paper",
    #             #     text=all_gsa_names[row - 1],
    #             #     showarrow=False,
    #             #     xanchor="center",
    #             #     yanchor='bottom',
    #             #     font=dict(
    #             #         size=16,
    #             #     )
    #             # )
    #         if row==1:
    #             fig.add_annotation(
    #                 x=iposes[col-1],
    #                 y=1.13,  # annotation point
    #                 xref="paper",
    #                 yref="paper",
    #                 text=morris_model_names[num_params],
    #                 showarrow=False,
    #                 xanchor="center",
    #                 font=dict(
    #                     size=16,
    #                 )
    #             )
    #             ipos += 1
    #         row += 1
    #     fig.update_xaxes(title_text=r"$\text{Iterations}$", row=row - 1, col=col)
    #
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex, )
    # fig.update_layout(
    #     width=1000, height=600,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     margin=dict(l=0, r=0, t=70, b=10),
    #     legend=dict(
    #         x=0.5,
    #         y=-0.12,
    #         xanchor='center',
    #         font_size=14,
    #         orientation='h',
    #         traceorder="normal",
    #     )
    # )
    # fig.show()
    # if plot_robustness_ranking:
    #     save_fig(fig, "morris_stat_ranking_{}_robust_{}".format(num_ranks, option), fig_format, write_dir_fig)
    # else:
    #     save_fig(fig, "morris_stat_ranking_{}_{}".format(num_ranks, option), fig_format, write_dir_fig)
    #
    #
    # endregion

    ### 3. Scalability of GSA results visually
    ##########################################
    # region

    data_dict = {
        "spearman": {
            "sampling": [0.06, 1.6, 5.19],
            "indices": [4.6, 16.73, 56.99],
            "memory": [31, 762, 3 * 1024],
        },
        "total": {
            "sampling": [4.88, 138.01, 735.34],
            "indices": [0.01, 0.04, 0.05],
            "memory": [1, 3.8, 7.6],
        },
        "delta": {
            "sampling": [0.27, 9.44, 88.56],
            "indices": [5.61, 114.06, 425.37],
            "memory": [61, 1.5 * 1024, 6 * 1024],
        },
        "total_gain": {
            "sampling": [0.06, 1.6, 5.19],
            "indices": [5.51, 169.24, 1115.42],
            "memory": [31, 762, 3 * 1024],
        },
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.22,
        vertical_spacing=0.12,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        subplot_titles=[v["name"] for v in sa_plot.values()],
    )
    row = 1
    col = 1
    showlegend = True
    for sa_name in sa_names.keys():
        data = data_dict[sa_name]
        fig.add_trace(
            go.Scatter(
                x=nums_params,
                y=data["sampling"],
                mode="lines+markers",
                marker=dict(color=color_blue_rgb, symbol="circle", size=5),
                showlegend=showlegend,
                name=r"$\text{Sampling time}$",
                # line=dict(dash='dot'),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=nums_params,
                y=data["indices"],
                mode="lines+markers",
                marker=dict(color=color_orange_rgb, symbol="circle", size=5),
                showlegend=showlegend,
                name=r"$\text{GSA time}$",
                # line=dict(dash='dot'),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=nums_params,
                y=data["memory"],
                mode="lines+markers",
                marker=dict(color=color_pink_rgb, symbol="x", size=7),
                showlegend=showlegend,
                name=r"$\text{Space}$",
                line=dict(dash="dash"),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )
        showlegend = False
        col += 1
        if col == 3:
            col = 1
            row = 2

    # fig.update_yaxes(type="log")
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_gray_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_black_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )
    fig.update_xaxes(title_text=r"$\text{Number of inputs}$", row=2, title_standoff=5)
    fig.update_yaxes(
        title_text=r"$\text{Time, [s]}$",
        col=1,
        title_standoff=5,
        secondary_y=False,
    )
    fig.update_yaxes(color=color_pink_rgb, secondary_y=True)
    fig.update_yaxes(
        title_text=r"$\text{Memory, [MB]}$",
        col=2,
        title_standoff=5,
        secondary_y=True,
    )
    fig.update_layout(
        width=650,
        height=500,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            x=0.5,
            y=-0.12,
            xanchor="center",
            font_size=14,
            orientation="h",
        ),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    save_fig(fig, "morris_scalability", fig_format, write_dir_fig)
    fig.show()

    # endregion
