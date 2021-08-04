import numpy as np
from pathlib import Path
import jenkspy

from gsa_framework.models.test_functions import Morris4
from gsa_framework.utils import read_pickle, write_pickle
from gsa_framework.convergence_robustness_validation import Robustness
from gsa_framework.convergence_robustness_validation.robustness import (
    compute_rho_choice,
)
from dev.utils_paper_plotting import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


path_base = Path("/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/")
num_params = 1000
write_dir = path_base / "{}_morris4".format(num_params)
write_dir_fig = path_base / "paper_figures_review1"
write_dir_arr = write_dir / "arrays"

# Read correlations GSA results
filepath_S = write_dir_arr / "S.correlationsGsa.randomSampling.4000.3407.pickle"
S = read_pickle(filepath_S)
sa_name = "spearman"

# Read stability dicts
files_stability = [
    x
    for x in write_dir_arr.iterdir()
    if x.is_file() and "S." in x.name and "stability" in x.name
]
files_stability_dict = get_files_dict_sorted(files_stability)
filepath_stability_dict = {
    "corr": (files_stability_dict["corr"], "spearman"),
    "salt": (files_stability_dict["salt"], "Total order"),
    "delt": (files_stability_dict["delt"], "delta"),
    "xgbo": (files_stability_dict["xgbo"], "total_gain"),
}
stability_dicts = []
for k, v in filepath_stability_dict.items():
    stability_dict = read_pickle(v[0])
    stability_dicts.append(stability_dict)
num_ranks = 4
rho_types_sarrazin = ["rho1", "rho5", "rho6"]
rho_types = ["rho1", "rho5", "rho6", "spearmanr"]
st_classes = {}
for rho_type in rho_types:
    st = Robustness(
        stability_dicts,
        write_dir,
        num_ranks=num_ranks,
        bootstrap_ranking_tag="paper1_review1",
        ci_type="student",
        rho_choice=rho_type,
    )
    st_classes[rho_type] = st

fig_format = ["pdf"]
opacity = 0.85

num_influential = num_params // 100
model = Morris4(num_params=num_params, num_influential=num_influential)
morris_model_name = r"$\underline{\text{Morris model, 1'000 inputs}}$"

sa_names = {
    "spearman": "corr",
    "total": "salt",
    "delta": "delt",
    "total_gain": "xgbo",
}
rho_type_names = {
    "rho1": r"$\underline{\text{Unweighted Spearman}}$",
    "rho5": r"$\underline{\text{Coef. based on Savage scores}}$",
    "rho6": r"$\underline{\text{Weighted rank [Sarrazin, 2016]}}$",
    "spearmanr": r"$\underline{\text{Spearman & tied ranks}}$",
}
rho_type_names_no_underline = {
    "rho1": r"$\text{Contribution }F_j \text{ of } j\text{-th input to unweighted Spearman } \rho_1$",
    "rho5": r"$\text{Contribution }F_j \text{ of } j\text{-th input to coef. } \rho_5 \text{ based on Savage scores}$",
    "rho6": r"$\text{Contribution }F_j \text{ of } j\text{-th input to weighted rank } \rho_6 {  [Sarrazin, 2016]}$",
}
rho_yaxes = {
    "rho1": r"$\rho_1$",
    "rho5": r"$\rho_5$",
    "rho6": r"$\rho_6$",
    "spearmanr": r"$\rho$",
}

# def get_savage_score(R):
#     M = len(R)
#     SS = np.empty(M)
#     for i in range(M):
#         m = R[i]
#         SS[i] = np.sum(1 / np.arange(m, M))
#     return SS
#
# def compute_rho(Sj,Sk,Rj,Rk,M,rho='rho1'):
#     if rho != 'rho6':
#         if rho=='rho1':
#             Fi = 6 * (Rj - Rk) ** 2 / M / (M ** 2 - 1)
#         elif rho=='rho2':
#             Ftemp = (Rj - Rk) ** 2 * (1 / Rj + 1 / Rk)
#             Fi = 2 * Ftemp / np.max(Ftemp)
#         elif rho=='rho3':
#             Ftemp = (Rj - Rk) ** 2 * (1 / Rj / Rk)
#             Fi = 2 * Ftemp / np.sum(Ftemp)
#         elif rho=='rho4':
#             Ftemp = (Rj - Rk) ** 2 * (1 / (Rj + Rk))
#             Fi = 2 * Ftemp / np.sum(Ftemp)
#         elif rho=='rho5':
#             SSj = get_savage_score(Rj)
#             SSk = get_savage_score(Rk)
#             SS1 = np.sum(1 / np.arange(1, M+1))
#             Fi = (SSj-SSk)**2 / 2 / (M-SS1)
#         r = 1 - np.sum(Fi)
#     else:
#         maxS = np.max(np.vstack([Sj,Sk]), axis=0)
#         Fi = np.abs(Rj-Rk) * maxS / sum(maxS)
#         r = np.sum(Fi)
#     return r, Fi

step = -1
j = 10
k = 111
Sj = np.abs(st.bootstrap_data[sa_name][step][j, :])
Sk = np.abs(st.bootstrap_data[sa_name][step][k, :])
Rj = np.argsort(Sj)[::-1] + 1
Rk = np.argsort(Sk)[::-1] + 1

breaks = jenkspy.jenks_breaks(Sj, nb_class=num_ranks)
Rj_clustered = st.get_one_clustered_ranking(Sj, num_ranks, breaks)
Rk_clustered = st.get_one_clustered_ranking(Sk, num_ranks, breaks)

rho_dict = {}
for rho_type in rho_types_sarrazin:
    r, Fi = compute_rho_choice([Rj], Rk, [Sj], Sk, rho_type)
    rho_dict[rho_type] = {"rho": r[0], "Fi": Fi}

rho_types_clustered = rho_types_sarrazin
rho_dict_clustered = {}
for rho_type in rho_types_clustered:
    r, Fi = compute_rho_choice([Rj_clustered], Rk_clustered, [Sj], Sk, rho_type)
    rho_dict_clustered[rho_type] = {"rho": r[0], "Fi": Fi}

# subplot_titles = ["",] + ["{:s} = {:5.4f}".format(k,v['rho']) for k,v in rho_dict.items()]

#####################################################
### Figure with clustered and NON-clustered ranks ###
#####################################################

# region

subplot_titles = [
    "",
    "",
    r"$\rho_1 = {:4.3f}$".format(rho_dict["rho1"]["rho"]),
    r"$\rho_1 = {:4.3f}$".format(rho_dict_clustered["rho1"]["rho"]),
    r"$\rho_5 = {:4.3f}$".format(rho_dict["rho5"]["rho"]),
    r"$\rho_5 = {:4.3f}$".format(rho_dict_clustered["rho5"]["rho"]),
    r"$\rho_6 = {:4.3f}$".format(rho_dict["rho6"]["rho"]),
    r"$\rho_6 = {:4.3f}$".format(rho_dict_clustered["rho6"]["rho"]),
]

nrows = len(rho_types_sarrazin) + 1
num_params_plot = 50
x = np.arange(num_params_plot)
# fig = make_subplots(
#     rows=nrows, cols=2,
#     shared_xaxes=True,
#     vertical_spacing=0.2,
#     horizontal_spacing=0.12,
#     subplot_titles=subplot_titles,
# )
# # subplot_titles = [
# #     r"$\underline{\text{Rankings } R_m \text{ and } R_n}$",
# #     r"$\underline{\text{Rankings } R_m \text{ and } R_n \text{ clustered into 4 ranks}}$",
# # ]
# column_titles = [
#     r"$\underline{\text{(i) All model inputs have distinct ranks}}$",
#     r"$\underline{\text{(ii) Model inputs are clustered into 4 ranks}}$",
# ]
# jpos = 0
# jposes = np.array([1.01, 0.7, 0.4, 0.1]) + 0.05
# iposes = [0.22, 0.78]
# showlegend = True
# for col in [1,2]:
#     if col==1:
#         rho_dict_plot = rho_dict
#         Rj_plot, Rk_plot = Rj, Rk
#     elif col==2:
#         rho_dict_plot = rho_dict_clustered
#         Rj_plot, Rk_plot = Rj_clustered, Rk_clustered
#     if min(Rj_plot) == 0 and min(Rk_plot) == 0:
#         Rj_plot += 1
#         Rk_plot += 1
#     fig.add_trace(
#         go.Bar(
#             x=x,
#             y=Rj_plot[:num_params_plot],
#             marker=dict(color=color_blue_rgb),
#             name=r"$\text{Ranking } R^m$",
#             showlegend=showlegend,
#             opacity=opacity,
#         ),
#         row=1,
#         col=col,
#     )
#     fig.add_trace(
#         go.Bar(
#             x=x,
#             y=Rk_plot[:num_params_plot],
#             marker=dict(color=color_orange_rgb),
#             name=r"$\text{Ranking } R^n$",
#             showlegend=showlegend,
#             opacity=opacity,
#         ),
#         row=1,
#         col=col,
#     )
#     showlegend = False
#     for i, rho_type in enumerate(rho_types_sarrazin):
#         r = rho_dict_plot[rho_type]['rho']
#         Fi = rho_dict_plot[rho_type]['Fi']
#         fig.add_trace(
#             go.Bar(
#                 x=x,
#                 y=Fi[:num_params_plot],
#                 marker=dict(color=color_purple_rgb),
#                 showlegend=False,
#                 opacity=opacity,
#             ),
#             row=i+2,
#             col=col,
#         )
#         if col==1:
#             fig.add_annotation(
#                 x=0.5,
#                 y=jposes[jpos+1],  # annotation point
#                 xref="paper",
#                 yref="paper",
#                 text=rho_type_names_no_underline[rho_type],
#                 showarrow=False,
#                 xanchor="center",
#                 yanchor="bottom",
#                 font=dict(
#                     size=16,
#                 )
#             )
#             jpos += 1
#     fig.add_annotation(
#         x=iposes[col-1],
#         y=1.15,  # annotation point
#         xref="paper",
#         yref="paper",
#         text=column_titles[col-1],
#         showarrow=False,
#         xanchor="center",
#         yanchor="bottom",
#         font=dict(
#             size=16,
#         )
#     )
# fig.add_annotation(
#     x=0.5,
#     y=jposes[0],  # annotation point
#     xref="paper",
#     yref="paper",
#     text=r"$\text{Model inputs ranks obtained from two rankings}$",
#     showarrow=False,
#     xanchor="center",
#     yanchor="bottom",
#     font=dict(
#         size=16,
#     )
# )
#
# fig.update_xaxes(
#     title_text=r'$\text{Model inputs}$',
#     row=nrows,
# )
# fig.update_yaxes(
#     title_text=r'$F_j$',
# )
# fig.update_yaxes(
#     title_text=r'$\text{Ranks}$',
#     row=1,
# )
#
# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
#                  zeroline=False, zerolinewidth=1, zerolinecolor=color_black_hex,
#                  showline=True, linewidth=1, linecolor=color_gray_hex)
# fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_gray_hex,
#                  zeroline=True, zerolinewidth=1, zerolinecolor=color_black_hex,
#                  showline=True, linewidth=1, linecolor=color_gray_hex, )
# fig.update_layout(
#     width=1000, height=600,
#     paper_bgcolor='rgba(255,255,255,1)',
#     plot_bgcolor='rgba(255,255,255,1)',
#     margin=dict(l=0, r=0, t=90, b=0),
# )
# fig.update_layout(
#     barmode='group',
#     bargap=0.45, # gap between bars of adjacent location coordinates.
#     bargroupgap=0.06,
#     legend=dict(
#         x=0.5,
#         y=0.99,
#         xanchor='center',
#         yanchor='bottom',
#         font_size=14,
#         orientation='h',
#         traceorder="normal",
#         bgcolor='rgba(0,0,0,0)',
#     )
# )
#
# save_fig(fig, "rankings_comparison", fig_format, write_dir_fig)
# fig.show()

# endregion


###################################
### Figure with converging rhos ###
###################################
opacity = 0.6

plot_robustness_ranking = True
all_gsa_names = [v["name"] for v in sa_plot.values()]
nrows = 4
fig = make_subplots(
    rows=nrows,
    cols=len(rho_types),
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=0.17,
    horizontal_spacing=0.09,
    # subplot_titles=[
    #     "", all_gsa_names[0], "", "",
    #     "", all_gsa_names[1], "", "",
    #     "", all_gsa_names[2], "", "",
    #     "", all_gsa_names[3], "", "",
    # ],
)

ipos = 0
# iposes = [0.137, 0.5, 0.863]
iposes = [0.09125, 0.36375, 0.63625, 0.90875]
jposes = np.array([1.0, 0.7075, 0.415, 0.1225]) + 0.01
showlegend = True
for col, rho_type in enumerate(rho_types):
    col += 1
    st = st_classes[rho_type]
    if rho_type == "rho1" or rho_type == "rho5" or rho_type == "spearmanr":
        agreement_value = 1
    elif rho_type == "rho6":
        agreement_value = 0
    for row, sa_name in enumerate(sa_names.keys()):
        row += 1
        y = st.bootstrap_rankings_width_percentiles[sa_name]["mean"][:-1]
        cf_width = st.bootstrap_rankings_width_percentiles[sa_name][
            "confidence_interval"
        ][:-1]
        lower = y - cf_width / 2
        upper = y + cf_width / 2
        color = color_blue_tuple
        fig.add_trace(
            go.Scatter(
                x=st.iterations[sa_name][:-1],
                y=y,
                mode="lines",
                marker=dict(color=color_blue_rgb),
                showlegend=showlegend,
                name=r"$\text{Convergence of ranking}$",
            ),
            col=col,
            row=row,
        )
        fig.add_trace(
            go.Scatter(
                x=st.iterations[sa_name][[0, -2]],
                y=agreement_value * np.ones(2),
                mode="lines",
                line=dict(
                    color=color_orange_rgb,
                    dash="dash",
                ),
                showlegend=showlegend,
                name=r"$\text{Value of perfect agreement for given statistic}$",
            ),
            col=col,
            row=row,
        )
        showlegend = False
        if plot_robustness_ranking:
            fig.add_trace(
                go.Scatter(
                    x=st.iterations[sa_name][:-1],
                    y=lower,
                    mode="lines",
                    opacity=opacity,
                    showlegend=False,
                    marker=dict(
                        color="rgba({},{},{},{})".format(
                            color[0],
                            color[1],
                            color[2],
                            opacity,
                        ),
                    ),
                    line=dict(width=0),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=st.iterations[sa_name][:-1],
                    y=upper,
                    showlegend=False,
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba({},{},{},{})".format(
                        color[0],
                        color[1],
                        color[2],
                        opacity,
                    ),
                    fill="tonexty",
                ),
                row=row,
                col=col,
            )
        if row == 1:
            fig.add_annotation(
                x=iposes[col - 1],
                y=1.15,  # annotation point
                xref="paper",
                yref="paper",
                text=rho_type_names[rho_type],
                showarrow=False,
                xanchor="center",
                font=dict(
                    size=16,
                ),
            )
            ipos += 1
        if sa_name == "spearman" or sa_name == "total_gain":
            if num_params == 1000:
                tickvals = [1000, 2000, 3000]
                ticktext = ["1'000", "2'000", "3'000"]
            if num_params == 5000:
                tickvals = [5000, 10000, 15000]
                ticktext = ["5'000", "10'000", "15'000"]
            if num_params == 10000:
                tickvals = [10000, 20000, 30000]
                ticktext = ["10'000", "20'000", "30'000"]
        if sa_name == "total":
            tickangle = 30
            if num_params == 1000:
                tickvals = [20000, 40000, 60000, 80000]
                ticktext = ["20'000", "40'000", "60'000", "80'000"]
            if num_params == 5000:
                tickvals = [100000, 200000, 300000, 400000]
                ticktext = ["100'000", "200'000", "300'000", "400'000"]
            if num_params == 10000:
                tickvals = [200000, 400000, 600000, 800000]
                ticktext = ["200'000", "400'000", "600'000", "800'000"]
        if sa_name == "delta":
            if num_params == 1000:
                tickvals = [2000, 4000, 6000]
                ticktext = ["2'000", "4'000", "6'000"]
            if num_params == 5000:
                tickvals = [10000, 20000, 30000]
                ticktext = ["10'000", "20'000", "30'000"]
            if num_params == 10000:
                tickvals = [20000, 40000, 60000]
                ticktext = ["20'000", "40'000", "60'000"]
        fig.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=rho_yaxes[rho_type],
            row=row,
            col=col,
            title_standoff=5,
        )
    if rho_type == "rho1":
        range_ = [0.99998, 1.000005]
    elif rho_type == "rho5":
        range_ = [0.84, 1.01]
    elif rho_type == "rho6":
        range_ = [-0.1, 2.1]
    elif rho_type == "spearmanr":
        range_ = [0, 1.01]
    fig.update_yaxes(range=range_, col=col)
    fig.update_xaxes(title_text=r"$\text{Iterations}$", row=row, col=col)
for j in range(nrows):
    fig.add_annotation(
        x=0.5,
        y=jposes[j],  # annotation point
        xref="paper",
        yref="paper",
        text=all_gsa_names[j],
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(
            size=16,
        ),
    )

fig.update_xaxes(
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
    width=1000,
    height=600,
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
    margin=dict(l=0, r=0, t=70, b=10),
    legend=dict(
        x=0.5,
        y=-0.15,
        xanchor="center",
        font_size=14,
        orientation="h",
        traceorder="normal",
    ),
)

save_fig(
    fig, "stat_ranking_{}_si_experiments".format(num_ranks), fig_format, write_dir_fig
)

# fig.show()
