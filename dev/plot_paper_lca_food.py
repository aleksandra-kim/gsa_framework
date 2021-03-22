import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from gsa_framework.utils import read_hdf5_array, read_pickle
from gsa_framework.convergence_robustness_validation.robustness import Robustness
from dev.utils_paper_plotting import *


normalize = lambda val: (val - min(val)) / (max(val) - min(val))
color_all = color_blue_rgb
color_inf = color_orange_rgb
color_sca = color_blue_orange_av_rgb


if __name__ == "__main__":

    #     path_base = Path('/data/user/kim_a/paper_gsa/')
    path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/")
    num_params = 10000
    iterations = 2000
    num_ranks = 10
    write_dir = path_base / "lca_model_food_10000"
    write_dir_arr = write_dir / "arrays"
    write_dir_fig = path_base / "paper_figures"
    fig_format = ["pdf"]

    all_gsa_names = [v["name"] for v in sa_plot.values()]

    # GSA results
    filepath_gsa_corr = (
        write_dir_arr / "S.correlationsGsa.randomSampling.40000.92374523.pickle"
    )
    filepath_gsa_salt = (
        write_dir_arr / "S.saltelliGsa.saltelliSampling.390078.None.pickle"
    )
    filepath_gsa_delt = (
        write_dir_arr / "S.deltaGsaNr0.latinSampling.80000.92374523.pickle"
    )
    filepath_gsa_gain = (
        write_dir_arr
        / "S.xgboostGsa_Lr0.15G0Mcw300Md4RegL0RegA0Ne600Ss0.3Cbt0.2_.randomSampling.40000.92374523.pickle"
    )
    filepath_gsa_dict = {
        "corr": (filepath_gsa_corr, "spearman"),
        "salt": (filepath_gsa_salt, "Total order"),
        "delt": (filepath_gsa_delt, "delta"),
        "xgbo": (filepath_gsa_gain, "total_gain"),
    }

    # Stability dicts
    filepath_stability_corr = (
        write_dir_arr
        / "stability.S.correlationsGsa.randomSampling.40000Step800.60.92374523.pickle"
    )
    filepath_stability_salt = (
        write_dir_arr
        / "stability.S.saltelliGsa.saltelliSampling.390078Step10002.60.None.pickle"
    )
    filepath_stability_delt = (
        write_dir_arr
        / "stability.S.deltaGsaNr0.latinSampling.80000Step1600.60.92374523.pickle"
    )
    filepath_stability_gain = (
        write_dir_arr
        / "stability.S.xgboostGsa_Lr0.15G0Mcw300Md4RegL0RegA0Ne600Ss0.3Cbt0.2_.randomSampling.40000Step800.60.92374523.pickle"
    )
    filepath_stability_dict = {
        "corr": (filepath_stability_corr, "spearman"),
        "salt": (filepath_stability_salt, "Total order"),
        "delt": (filepath_stability_delt, "delta"),
        "xgbo": (filepath_stability_gain, "total_gain"),
    }

    # Validation results
    filepath_val_all = write_dir_arr / "validation.Y.all.2000.23467.hdf5"
    filepath_val_corr = (
        write_dir_arr / "validation.Y.60inf.2000.23467.SpearmanIndex.hdf5"
    )
    filepath_val_salt = write_dir_arr / "validation.Y.60inf.2000.23467.TotalIndex.hdf5"
    filepath_val_delt = (
        write_dir_arr / "validation.Y.60inf.2000.23467.DeltaIndexNr0.hdf5"
    )
    filepath_val_gain = write_dir_arr / "validation.Y.60inf.2000.23467.TotalGain.hdf5"
    filepath_val_dict = {
        "all": filepath_val_all,
        "corr": filepath_val_corr,
        "salt": filepath_val_salt,
        "delt": filepath_val_delt,
        "xgbo": filepath_val_gain,
    }

    Y_dict, S_dict = {}, {}
    Y_arr, S_arr = np.zeros((0, iterations)), np.zeros((0, num_params))
    stability_dicts = []
    for k in filepath_val_dict.keys():
        Y_dict[k] = read_hdf5_array(filepath_val_dict[k]).flatten()
        if k != "all":
            Y_arr = np.vstack([Y_arr, Y_dict[k]])
            S_dict[k] = read_pickle(filepath_gsa_dict[k][0])[filepath_gsa_dict[k][1]]
            S_arr = np.vstack([S_arr, S_dict[k]])
            stability_dict = read_pickle(filepath_stability_dict[k][0])
            stability_dicts.append(stability_dict)
    bootstrap_ranking_tag = "paper1"
    st = Robustness(
        stability_dicts,
        write_dir,
        num_ranks=num_ranks,
        bootstrap_ranking_tag=bootstrap_ranking_tag,
    )

    # ### 1. Plot  GSA  results
    # ######################
    # region
    # fig = make_subplots(
    #     rows=4,
    #     cols=1,
    #     shared_xaxes=True,
    #     vertical_spacing=0.12,
    #     subplot_titles=all_gsa_names,
    #     # column_widths=[0.7, 0.3]
    # )
    # row = 1
    # for k,v in S_dict.items():
    #     fig.add_trace(
    #         go.Scatter(
    #             x=np.arange(num_params),
    #             y=v,
    #             mode="markers",
    #             opacity=1,
    #             showlegend=False,
    #             marker=dict(size=3, color=color_blue_rgb),
    #         ),
    #         row=row,
    #         col=1,
    #     )
    #     fig.update_yaxes(title_text=sa_plot[k]['notation'], row=row, col=1)
    #     row += 1
    # fig.update_xaxes(title_text=r"$\text{Model inputs}$", row=row-1, col=1)
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_gray_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex,)
    # fig.update_layout(
    #     width=700, height=600,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     margin=dict(l=0, r=0, t=30, b=0),
    # )
    # # fig.show()
    # save_fig(fig, "lca_all_gsa_results", fig_format, write_dir_fig)

    # endregion

    # ## 2. Plot stability and convergence
    # #################################
    # region

    # # Analytical spearman confidence intervals
    # from gsa_framework.sensitivity_methods.correlations import get_corrcoef_interval_width
    # thetas = np.linspace(0.01, 0.95, 100)
    # all_iterations = st.iterations['spearman']
    # analytical_spearman_ci = []
    # for iterations in all_iterations:
    #     list_ = []
    #     for theta in thetas:
    #         list_.append(get_corrcoef_interval_width(theta, iterations=iterations)['spearman'])
    #     analytical_spearman_ci.append(max(list_))
    # analytical_spearman_ci = np.array(analytical_spearman_ci)
    #
    # opacity = 0.6
    # sa_names = {
    #     "spearman": "corr",
    #     "total": "salt",
    #     "delta": "delt",
    #     "total_gain": "xgbo"
    # }
    #
    # fig = make_subplots(
    #     rows=4,
    #     cols=2,
    #     shared_xaxes=False,
    #     shared_yaxes=False,
    #     vertical_spacing=0.12,
    #     horizontal_spacing=0.1,
    # )
    # num_influential = 20
    # num_non_influential = 60
    #
    # showlegend = True
    #
    # for col in [1,2]:
    #     if col==1:
    #         option = "nzoomed_in"
    #     else:
    #         option = "zoomed_in"
    #     row = 1
    #     for sa_name in sa_names.keys():
    #         # where_sorted = [np.argsort(el)[-1::-1] for el in st.sa_mean_results[sa_name]]
    #         # where_non_inf = [w[num_non_influential:] for w in where_sorted]
    #         # # where_non_inf = [where_sorted[-1][num_non_influential:]]*len(where_sorted)
    #         # where_inf = [w[:num_influential] for w in where_sorted]
    #         # # where_inf = [where_sorted[-1][:num_influential]]*len(where_sorted)
    #         # sa_min_inf = [min(el[where_inf[i]]) for i,el in enumerate(st.sa_mean_results[sa_name])]
    #         # sa_max_non_inf = [max(el[where_non_inf[i]]) for i,el in enumerate(st.sa_mean_results[sa_name])]
    #         # confidence_intervals_non_inf = np.array([st.confidence_intervals[sa_name][i][w] for i,w in enumerate(where_non_inf)])
    #         # confidence_intervals_inf = np.array([st.confidence_intervals[sa_name][i][w] for i, w in enumerate(where_inf)])
    #         # stat_screening = np.max(confidence_intervals_non_inf, axis=1)
    #         # stat_indices = np.max(confidence_intervals_inf, axis=1)
    #         if option == "zoomed_in":
    #             if sa_name == "total":
    #                 start_iterations_ = 6
    #             else:
    #                 start_iterations_ = 1
    #         else:
    #             if sa_name == "total":
    #                 start_iterations_ = 1
    #             else:
    #                 start_iterations_ = 0
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
    #                     line=dict(dash='dot'),
    #                     name=r"$\text{Analytical confidence intervals}$"
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
    #         # k = 0
    #         # for k in range(2):
    #         #     if k == 0:
    #         #         color = color_purple_tuple
    #         #         y=sa_min_inf
    #         #         width = stat_indices
    #         #     else:
    #         #         color = color_blue_tuple
    #         #         y=sa_max_non_inf
    #         #         width = stat_screening
    #         width = st.confidence_intervals_max[sa_name][start_iterations_:]
    #         color = color_blue_tuple
    #         lower = y-width / 2
    #         upper = y+width / 2
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
    #         if col==1:
    #             fig.update_yaxes(title_text=sa_plot[sa_names[sa_name]]['stat_indices'], row=row, col=1)
    #             fig.update_xaxes(
    #                 range=[
    #                     min(st.iterations['spearman']),
    #                     max(st.iterations['total'])
    #                 ],
    #                 row=row,
    #                 col=col,
    #             )
    #             fig.add_annotation(
    #                 x=0.5,
    #                 y=(1-0.16)/3*(4-row)+0.16 + 0.02,  # annotation point
    #                 xref="paper",
    #                 yref="paper",
    #                 text=all_gsa_names[row-1],
    #                 showarrow=False,
    #                 xanchor="center",
    #                 yanchor='bottom',
    #                 font=dict(
    #                     size=16,
    #                 )
    #             )
    #
    #         row+=1
    #     fig.update_xaxes(title_text=r"$\text{Iterations}$", row=row-1, col=col)
    #
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_layout(
    #     width=900, height=600,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     legend=dict(
    #         orientation = 'h',
    #         x=0.5,
    #         y=-0.12,
    #         xanchor='center',
    #         font_size=14,
    #     ),
    #     margin=dict(l=0, r=0, t=30, b=0),
    # )
    # fig.show()
    # save_fig(fig, "lca_stat_indices", fig_format, write_dir_fig)

    # endregion

    # ## 3. Validation of GSA  results
    # #################################
    # region

    # # 4. Spearman and Wasserstein for the table
    # rho, _ = spearmanr(Y_arr.T, Y_dict['all'])
    # wdist = []
    # for arr in Y_arr:
    #     wdist.append(wasserstein_distance(arr, Y_dict['all']))
    #
    # lca_scores_axis_title = r"$\text{LCA scores, [kg CO}_2\text{-eq}]$"
    # all_inputs_text = r"$\text{All inputs vary}$"
    # inf_inputs_text =  r"$\text{Only influential inputs vary}$"
    #
    # fig = make_subplots(
    #     rows=1,
    #     cols=2,
    #     horizontal_spacing=0.16,
    # )
    #
    # num_bins = 60
    # opacity = 0.7
    # start,end = 0,50
    # Y1 = Y_dict['all']
    # Y2 = Y_arr[2,:]
    #
    # # Validation correlation
    # x = np.arange(start, end)
    # fig.add_trace(
    #     go.Scatter(
    #         x=x,
    #         y=Y1[start:end],
    #         mode="lines+markers",
    #         marker=dict(color=color_all),
    #         showlegend=False,
    #     ),
    #     row=1, col=1,
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=x,
    #         y=Y2[start:end],
    #         mode="lines+markers",
    #         marker=dict(color=color_inf),
    #         showlegend=False,
    #     ),
    #     row=1, col=1,
    # )
    # fig.update_yaxes(title_text=lca_scores_axis_title, row=1, col=1, )
    # fig.update_xaxes(title_text=r"$\text{Subset of 50/2000 datapoints}$", row=1, col=1, )
    #
    #
    # # Validation histogram
    # bin_min = min(np.hstack([Y1, Y2]))
    # bin_max = max(np.hstack([Y1, Y2]))
    # bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    # freq1, bins1 = np.histogram(Y1, bins=bins_)
    # freq2, bins2 = np.histogram(Y2, bins=bins_)
    #
    # fig.add_trace(
    #     go.Bar(
    #         x=bins1,
    #         y=freq1,
    #         name=all_inputs_text,
    #         opacity=opacity,
    #         marker=dict(color=color_all),
    #         showlegend=True,
    #     ),
    #     row=1, col=2,
    # )
    # fig.add_trace(
    #     go.Bar(
    #         x=bins2,
    #         y=freq2,
    #         name=inf_inputs_text,
    #         opacity=opacity,
    #         marker=dict(color=color_inf),
    #         showlegend=True,
    #     ),
    #     row=1, col=2,
    # )
    # fig.update_layout(barmode="overlay")
    # fig.update_yaxes(title_text=r"$\text{Frequency}$", row=1, col=2)
    # fig.update_xaxes(title_text=lca_scores_axis_title, row=1, col=2)
    #
    # # Both
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_gray_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex, )
    # fig.update_layout(
    #     width=700, height=250,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     legend=dict(
    #         x=0.5,
    #         y=-0.32,
    #         orientation='h',
    #         xanchor='center',
    #         font=dict(size=14),
    #     ),
    #     margin=dict(l=0, r=0, t=0, b=0),
    # )
    # save_fig(fig, "lca_validation", fig_format, write_dir_fig)

    # endregion

    # ### 4. Convergence of ranking
    # #############################
    # region
    #
    # plot_robustness_ranking = True
    # sa_names = {
    #     "spearman": "corr",
    #     "total": "salt",
    #     "delta": "delt",
    #     "total_gain": "xgbo"
    # }
    # fig = make_subplots(
    #     rows=4,
    #     cols=2,
    #     shared_xaxes=False,
    #     shared_yaxes=False,
    #     vertical_spacing=0.12,
    #     horizontal_spacing=0.12,
    #     specs=[
    #         [{"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": False}, {"secondary_y": False}],
    #         [{"secondary_y": True}, {"secondary_y": True}],
    #     ],
    # )
    #
    # color = color_blue_tuple
    # opacity = 0.6
    # showlegend, showlegend2 = True, True
    # for col in [1, 2]:
    #     if col==1:
    #         option = "nzoomed_in"
    #     else:
    #         option = "zoomed_in"
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
    #                 fig.update_yaxes(range=[0.2, 1.1], row=row, col=col)
    #         y = st.bootstrap_rankings_width_percentiles[sa_name]["mean"][:-1]
    #         cf_width = st.bootstrap_rankings_width_percentiles[sa_name]["confidence_interval"][:-1]
    #         lower = y - cf_width/2
    #         upper = y + cf_width/2
    #         color=color_blue_tuple
    #         fig.add_trace(
    #             go.Scatter(
    #                 x = st.iterations[sa_name][:-1],
    #                 y = y,
    #                 mode="lines",
    #                 marker = dict(color=color_blue_rgb),
    #                 showlegend=showlegend,
    #                 name=r"$\text{Convergence of ranking}$",
    #             ),
    #             row=row,
    #             col=col,
    #             secondary_y=False,
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
    #                 row=row, col=1, secondary_y=False,
    #             )
    #             fig.update_xaxes(
    #                 range=[
    #                     min(st.iterations['spearman']),
    #                     max(st.iterations['total'])
    #                 ],
    #                 row=row,
    #                 col=col,
    #             )
    #             fig.add_annotation(
    #                 x=0.5,
    #                 y=(1-0.16)/3*(4-row)+0.16 + 0.02,  # annotation point
    #                 xref="paper",
    #                 yref="paper",
    #                 text=all_gsa_names[row-1],
    #                 showarrow=False,
    #                 xanchor="center",
    #                 yanchor='bottom',
    #                 font=dict(
    #                     size=16,
    #                 )
    #             )
    #         if option == "nzoomed_in":
    #             fig.update_yaxes(range=[0, 1.1], row=row,col=col)
    #         row += 1
    #     fig.update_xaxes(title_text=r"$\text{Iterations}$", row=row - 1, col=col)
    #
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
    #                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
    #                  showline=True, linewidth=1, linecolor=color_gray_hex,
    #                  title_standoff=5,)
    # fig.update_layout(
    #     width=900, height=600,
    #     paper_bgcolor='rgba(255,255,255,1)',
    #     plot_bgcolor='rgba(255,255,255,1)',
    #     margin=dict(l=0, r=0, t=30, b=0),
    #     legend=dict(
    #         x=0.5,
    #         y=-0.12,
    #         xanchor='center',
    #         font_size=14,
    #         orientation='h',
    #         traceorder="normal",
    #     )
    # )
    # if plot_robustness_ranking:
    #     save_fig(fig, "lca_stat_ranking_{}_robust".format(num_ranks), fig_format, write_dir_fig)
    # else:
    #     save_fig(fig, "lca_stat_ranking_{}".format(num_ranks), fig_format, write_dir_fig)
    # fig.show()
    # endregion

    ### 5. Distances in high dimensions
    ################################
    # region

    from sklearn.metrics.pairwise import (
        euclidean_distances,
        manhattan_distances,
        cosine_distances,
    )
    from scipy.spatial import distance

    get_diff = lambda arr: np.max(arr) / np.min(arr) - 1
    step = 50
    dims = np.arange(step, 2000 + step, step)
    N = 1000
    euclidean, manhattan, cosine, minkowski, correlation, chebyshev, mahalanobis = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for dim in dims:
        x = np.random.rand(1, dim)
        y = np.random.rand(N, dim)
        euclidean.append(get_diff(euclidean_distances(x, y)))
        manhattan.append(get_diff(manhattan_distances(x, y)))
        cosine.append(get_diff(cosine_distances(x, y)))

        minkowski_dist = np.zeros(N)
        correlation_dist = np.zeros(N)
        chebyshev_dist = np.zeros(N)
        mahalanobis_dist = np.zeros(N)
        for i, y_ in enumerate(y):
            minkowski_dist[i] = distance.minkowski(x, y_, p=3)
            correlation_dist[i] = distance.correlation(x, y_)
            chebyshev_dist[i] = distance.chebyshev(x, y_)
            # mahalanobis_dist[i] = distance.mahalanobis(x, y_)
        minkowski.append(get_diff(minkowski_dist))
        correlation.append(get_diff(correlation_dist))
        chebyshev.append(get_diff(chebyshev_dist))
        # mahalanobis.append(get_diff(mahalanobis_dist))

    dist_dict = {
        r"$\text{Euclidean}$": euclidean,
        r"$\text{Minkowski, or p-norm}$": minkowski,
        r"$\text{Manhattan}$": manhattan,
        r"$\text{Correlation}$": correlation,
        r"$\text{Cosine}$": cosine,
        r"$\text{Chebyshev}$": chebyshev,
        # r"$\text{Mahalanobis}$": mahalanobis,
    }
    colors = [
        color_blue_rgb,
        color_purple_rgb,
        color_orange_rgb,
        color_green_rgb,
        color_pink_rgb,
        color_yellow_rgb,
    ]

    fig = go.Figure()
    i = 0
    for k, v in dist_dict.items():
        fig.add_trace(
            go.Scatter(
                x=dims, y=v, name=k, showlegend=True, mode="lines", line_color=colors[i]
            )
        )
        i += 1
    y_name = r"$\frac{dist^k_{\max} - dist^k_{\min}}{dist^k_{\min}}$"
    fig.update_xaxes(title_text=r"$\text{Number of dimensions, }k$")
    fig.update_yaxes(title_text=y_name, title_font_size=18)
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
    fig.update_layout(
        width=400,
        height=250,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        # font=dict(
        #     size=16,
        # ),
        legend=dict(
            x=0.53,
            y=0.97,
            xanchor="left",
            yanchor="top",
            font=dict(size=13),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    save_fig(fig, "distances_in_high_dim", fig_format, write_dir_fig)
    # endregion

    ### 6. Agreement between results of GSA sensitivity_analysis
    ###############################################
    # region

    # num_ranks = 10
    # flag_normalize = False
    # if flag_normalize:
    #     S_corr = normalize(np.abs(S_arr[0]))
    #     S_salt = normalize(S_arr[1])
    #     S_delt = normalize(S_arr[2])
    #     S_xgbo = normalize(S_arr[3])
    # else:
    #     S_corr = np.abs(S_arr[0])
    #     S_salt = S_arr[1]
    #     S_delt = S_arr[2]
    #     S_xgbo = S_arr[3]
    # breaks = jenkspy.jenks_breaks(S_corr, nb_class=num_ranks)
    # rankings_corr = st.get_one_clustered_ranking(S_corr, num_ranks, breaks=None)
    # rankings_salt = st.get_one_clustered_ranking(S_salt, num_ranks, breaks=None)
    # rankings_delt = st.get_one_clustered_ranking(S_delt, num_ranks, breaks=None)
    # rankings_xgbo = st.get_one_clustered_ranking(S_xgbo, num_ranks, breaks=None)
    # rankings = np.vstack([
    #     rankings_corr,
    #     rankings_salt,
    #     rankings_delt,
    #     rankings_xgbo,
    # ])
    # rho, _ = spearmanr(rankings.T)
    # r = np.corrcoef(rankings)
    #
    # def my_corrcoef(ranking, S_arr):
    #     ranks = list(set(ranking))
    #     corrp, corrs, where = {}, {}, {}
    #     for rank in ranks:
    #         where[rank] = np.where(ranking==rank)[0]
    #         corrp[rank] = np.corrcoef(S_arr.T[where[rank]].T)
    #         corrs[rank], _ = spearmanr(S_arr.T[where[rank]])
    #     return corrs, corrp, where
    #
    # ss, pp, where  = my_corrcoef(rankings_salt, S_arr)
    #
    # def get_groups(rankings):
    #     where_list = []
    #     for ranking in rankings:
    #         ranks = set(ranking)
    #         where = {}
    #         for rank in ranks:
    #             where[rank] = np.where(ranking==rank)[0]
    #         where_list.append(where)
    #     return where_list
    #
    # def get_inf_non(S_arr):
    #     inds = []
    #     for S in S_arr:
    #         inds_sorted = np.argsort(S)[-1::-1]
    #         inds.append(inds_sorted)
    #     return inds
    #
    # inds = get_inf_non(S_arr)
    #
    # all_i60 = []
    # num1 = 100
    # num2 = 100
    # mat = np.zeros((4,4))
    # for i in range(4):
    #     ind_i = inds[i]
    #     i60 = set(ind_i[:num1])
    #     i_60 = set(ind_i[num2:])
    #     for j in range(i+1,4):
    #         ind_j = inds[j]
    #         j60 = set(ind_j[:num1])
    #         j_60 = set(ind_j[num2:])
    #         mat[i,j] = len(i60.intersection(j60))/num1
    #         mat[j,i] = len(i_60.intersection(j_60))/(num_params-num2)
    #
    # where_list = get_groups(rankings)
    #
    # fig =  make_subplots(rows=4, cols=1)
    # row = 1
    # for data in [S_corr, S_salt,  S_delt, S_xgbo]:
    #     breaks = jenkspy.jenks_breaks(data, nb_class=num_ranks)
    #     fig.add_trace(
    #         go.Histogram(
    #             x=data,
    #             showlegend=False,
    #         ),
    #         row=row,
    #         col=1,
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=breaks,
    #             y=np.zeros(len(breaks)),
    #             mode="markers",
    #             marker=dict(color="red"),
    #             showlegend=False,
    #         ),
    #         row=row,
    #         col=1,
    #     )
    #     row += 1
    # fig.show()

    # rho,_ = spearmanr(rankings.T)
    # rho = rho[:,-1][:-1]
    #
    # def correct_ties(array):
    #     array_ = deepcopy(array)
    #     cf = {}
    #     for val in list(set(array)):
    #         where = np.where(array == val)[0]
    #         m = len(where)
    #         array_[where] = val / m
    #         cf[val] = m * (m ** 2 - 1) / 12
    #     return array_, cf
    #
    #
    # def compute_rho1_complete_ties(array1, array2):
    #     M = len(array1)
    #     array1_, cf1 = correct_ties(array1)
    #     array2_, cf2 = correct_ties(array2)
    #     cf1_ = sum(list(cf1.values()))
    #     cf2_ = sum(list(cf2.values()))
    #     d = 6 * (np.sum((array1_ - array2_) ** 2) + cf1_ + cf2_)
    #     rho = 1 - d / M / (M ** 2 - 1)
    #     return rho
    #
    # def compute_rho1_complete(array1, array2):
    #     M = len(array1)
    #     rho = 1 - np.sum(6*(array1-array2)**2) / M / (M**2-1)
    #     return rho
    #
    # def compute_rho1_ties(array1, array2, where_dict):
    #     array1_, cf1 = correct_ties(array1)
    #     array2_, cf2 = correct_ties(array2)
    #     num_ranks = len(where_dict)
    #     M = len(array1)
    #     F = np.zeros(num_ranks)
    #     for rank, where in where_dict.items():
    #         F[rank-1] = 6 * (np.sum((array1_[where]-array2_[where])**2) + cf1.get(rank,0) + cf2.get(rank,0))
    #     return F / M / (M**2-1)
    # F = np.zeros((0,num_ranks))
    # F[:] = np.nan
    # for i in range(len(steps)):
    #     F = np.vstack(
    #         [
    #             F,
    #             compute_rho1_ties(rankings[i], rankings[-1], where_dict)
    #         ]
    #     )
    #
    # ncols = len(steps)
    # fig = make_subplots(
    #     rows=1,
    #     cols=1,
    #     shared_xaxes=False,
    #     # specs=[ [{"colspan": ncols}] + [None]*(ncols-1), [{}]*ncols ]
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=iterations[:-1],
    #         y=rho,
    #     ),
    #     row=1, col=1,
    # )
    # for i in range(ncols):
    #     iteration = iterations[i]
    #     fig.add_trace(
    #         go.Bar(
    #             x=F[i,:],
    #             y=np.arange(num_ranks)+1,
    #             showlegend=False,
    #             orientation='h',
    #         ),
    #         row=2, col=i+1,
    #     )
    #     fig.update_yaxes(autorange="reversed", row=2, col=i+1,)
    #
    # fig.update_xaxes(range=[-0.1, -(-F.max()//0.1 * 0.1)], row=2)
    # fig.show()

    # endregion
