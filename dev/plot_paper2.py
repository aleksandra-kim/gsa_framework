import numpy as np
from pathlib import Path
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
from scipy.stats import spearmanr

import bw2data as bd
import bw2calc as bc

from gsa_framework.models.life_cycle_assessment import LCAModel
from gsa_framework.utils import read_hdf5_array, read_pickle
from gsa_framework.visualization.plotting import plot_histogram_Y
from dev.utils_paper_plotting import *


show_figure1 = False
save_figure1 = False

show_figure2 = True
save_figure2 = False

show_figure3 = False
save_figure3 = False

show_figure4 = False
save_figure4 = False

show_figure5 = False
save_figure5 = False

show_figure6 = False
save_figure6 = False


if __name__ == "__main__":

    path_base = Path(
        "/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/"
    )
    # path_base = Path('/data/user/kim_a/protocol_gsa')

    bd.projects.set_current("GSA for protocol")
    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA(demand, uncertain_method)
    lca.lci()
    lca.lcia()

    write_dir = path_base / "protocol_gsa"
    write_dir_arr = write_dir / "arrays"
    write_dir_fig = write_dir / "figures"
    write_dir_sct = write_dir / "supply_chain"

    fig_format = ["pdf", "png"]
    num_bins = 60

    color_all = color_blue_rgb
    color_inf = color_orange_rgb
    color_sca = color_purple_rgb
    opacity = 0.65
    lca_scores_axis_title = r"$\text{LCA scores, [kg CO}_2\text{-eq}]$"
    lca_scores_axis_title_short = r"$\text{LCA scores}$"
    all_inputs_text = r"$\text{All inputs vary}$"
    inf_inputs_text = r"$\text{Only influential vary}$"
    inf_inputs_long_text = r"$\text{Only influential inputs vary}$"

    ########################################################
    ### FIGURE 1: Uncertainty distribution of LCA scores ###
    ########################################################
    filepath_Y = write_dir_arr / "validation.Y.all.2000.100023423.hdf5"
    Y = read_hdf5_array(filepath_Y).flatten()
    bin_min = min(Y)
    bin_max = max(Y)

    if show_figure1 or save_figure1:
        default_Y = lca.score

        fig = go.Figure()

        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq, bins = np.histogram(Y, bins=bins_)

        fig.add_trace(
            go.Scatter(
                x=bins,
                y=freq,
                name=all_inputs_text,
                opacity=opacity,
                line=dict(color=color_all, width=1, shape="hvh"),
                showlegend=True,
                fill="tozeroy",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(Y)],
                y=[0, 0],
                mode="markers",
                name=r"$\text{distribution mean}$",
                opacity=opacity,
                marker=dict(
                    color="black",
                    size=20,
                    symbol="x",
                ),
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(Y) - np.std(Y), np.mean(Y) - np.std(Y)],
                y=[0, 130],
                mode="lines",
                name=r"$\text{mean } \pm \text{ standard deviation}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dash",
                    color="black",
                    width=1,
                ),
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(Y) + np.std(Y), np.mean(Y) + np.std(Y)],
                y=[0, 130],
                mode="lines",
                name=r"$\pm \text{ standard deviation}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dash",
                    color="black",
                    width=1,
                ),
                showlegend=False,
            ),
        )

        if default_Y is not None:
            trace_name_default = r"$\text{Static LCA score}$"
            color_default_Y = "red"
            fig.add_trace(
                go.Scatter(
                    x=[default_Y],
                    y=[0],
                    mode="markers",
                    name=trace_name_default,
                    opacity=opacity,
                    marker=dict(
                        color=color_default_Y,
                        size=20,
                        symbol="x",
                    ),
                    showlegend=True,
                ),
            )

        fig.update_yaxes(title_text=r"$\text{Frequency}$", range=[-10, 130])
        fig.update_xaxes(
            title_text=lca_scores_axis_title,
        )

        # Both
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
            width=600,
            height=250,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                x=0.85,
                y=0.95,
                orientation="v",
                xanchor="center",
                font=dict(size=14),
                # bgcolor=color_lightgray_hex,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if show_figure1:
            fig.show()
        if save_figure1:
            save_fig(fig, "lca_scores_uncertainty", fig_format, write_dir_fig)

    ########################################################
    ### FIGURE 2: Uncertainty distribution of LCA scores ###
    ########################################################

    cutoffs = [1e-2, 1e-3, 1e-4]
    overlap_params = [100, 200, 400, 800, 1600]
    # --> Our results
    filepath_S = write_dir_arr / "S.correlationsGsa.randomSampling.80000.4000238.pickle"
    spearman = read_pickle(filepath_S)["spearman"]
    S_sorted = np.argsort(np.abs(spearman))[::-1]
    filepath = write_dir_arr / "parameter_choice_rm_lowinf.pickle"
    parameter_choice_rm_lowinf = read_pickle(filepath)
    filepath_model = write_dir_arr / "model.pickle"
    model = read_pickle(filepath_model)
    len_tech = model.uncertain_exchange_lengths["tech"]
    len_bio = model.uncertain_exchange_lengths["bio"]
    len_cf = model.uncertain_exchange_lengths["cf"]

    parameter_choice_dict_lsa = {}
    for num_params_ranking in overlap_params:
        # --> Our results
        parameter_choice_lsa = parameter_choice_rm_lowinf[S_sorted[:num_params_ranking]]
        parameter_choice_lsa_tech = parameter_choice_lsa[
            parameter_choice_lsa < len_tech
        ]
        parameter_choice_lsa_bio = (
            parameter_choice_lsa[
                np.logical_and(
                    parameter_choice_lsa >= len_tech,
                    parameter_choice_lsa < len_tech + len_bio,
                )
            ]
            - len_tech
        )
        parameter_choice_lsa_cf = (
            parameter_choice_lsa[parameter_choice_lsa >= len_tech + len_bio]
            - len_tech
            - len_bio
        )
        parameter_choice_dict_lsa[num_params_ranking] = {
            "tech": parameter_choice_lsa_tech,
            "bio": parameter_choice_lsa_bio,
            "cf": parameter_choice_lsa_cf,
        }

    parameter_choice_dict_sct = {}
    for cutoff in cutoffs:
        cutoff_str = "%.2E" % Decimal(cutoff)
        parameter_choice_dict_sct[cutoff] = {}
        for num_params_ranking in overlap_params:
            # --> SCT
            filename = "cutoff{}.params{}.pickle".format(cutoff_str, num_params_ranking)
            filepath = write_dir_sct / filename
            parameter_choice_dict_sct[cutoff][num_params_ranking] = read_pickle(
                filepath
            )["ranking"]["parameter_choice_dict"]

    overlaps = {}
    for cutoff in cutoffs:
        cutoff_str = "%.2E" % Decimal(cutoff)
        overlaps[cutoff] = {}
        for num_params_ranking in overlap_params:
            tech_lsa = parameter_choice_dict_lsa[num_params_ranking]["tech"]
            bio_lsa = parameter_choice_dict_lsa[num_params_ranking]["bio"]
            cf_lsa = parameter_choice_dict_lsa[num_params_ranking]["cf"]
            tech_sct = parameter_choice_dict_sct[cutoff][num_params_ranking]["tech"]
            bio_sct = parameter_choice_dict_sct[cutoff][num_params_ranking]["bio"]
            cf_sct = parameter_choice_dict_sct[cutoff][num_params_ranking]["cf"]
            overlaps[cutoff][num_params_ranking] = {
                "tech": {
                    "overlap": len(np.intersect1d(tech_lsa, tech_sct)),
                    "len_lsa": len(tech_lsa),
                    "len_sct": len(tech_sct),
                },
                "bio": {
                    "overlap": len(np.intersect1d(bio_lsa, bio_sct)),
                    "len_lsa": len(bio_lsa),
                    "len_sct": len(bio_sct),
                },
                "cf": {
                    "overlap": len(np.intersect1d(cf_lsa, cf_sct)),
                    "len_lsa": len(cf_lsa),
                    "len_sct": len(cf_sct),
                },
            }

    # --> Validation files
    val_dict = {
        "sct": {
            100: read_hdf5_array(
                write_dir_arr / "validation.Y.100inf.2000.100023423.sct.hdf5"
            ).flatten(),
            200: read_hdf5_array(
                write_dir_arr / "validation.Y.200inf.2000.100023423.sct.hdf5"
            ).flatten(),
            400: read_hdf5_array(
                write_dir_arr / "validation.Y.400inf.2000.100023423.sct.hdf5"
            ).flatten(),
            800: read_hdf5_array(
                write_dir_arr / "validation.Y.800inf.2000.100023423.sct.hdf5"
            ).flatten(),
            1600: read_hdf5_array(
                write_dir_arr / "validation.Y.1600inf.2000.100023423.sct.hdf5"
            ).flatten(),
        },
        "lsa": {
            100: read_hdf5_array(
                write_dir_arr / "validation.Y.100inf.2000.100023423.localSA.hdf5"
            ).flatten(),
            200: read_hdf5_array(
                write_dir_arr / "validation.Y.200inf.2000.100023423.localSA.hdf5"
            ).flatten(),
            400: read_hdf5_array(
                write_dir_arr / "validation.Y.400inf.2000.100023423.localSA.hdf5"
            ).flatten(),
            800: read_hdf5_array(
                write_dir_arr / "validation.Y.800inf.2000.100023423.localSA.hdf5"
            ).flatten(),
            1600: read_hdf5_array(
                write_dir_arr / "validation.Y.1600inf.2000.100023423.localSA.hdf5"
            ).flatten(),
        },
    }

    Yall = read_hdf5_array(
        write_dir_arr / "validation.Y.all.2000.100023423.hdf5"
    ).flatten()
    bin_min = min(Yall)
    bin_max = max(Yall)

    title_text_str_all = []
    for screening_name, data_dict in val_dict.items():
        for num_params_ranking, Yinf in data_dict.items():
            rho, _ = spearmanr(Yall, Yinf)
            title_text_str = r"$\rho = {:4.3f}$".format(rho)
            title_text_str_all.append(title_text_str)

    nrows = len(overlap_params)
    ncols = len(cutoffs) + 2
    cutoffs_str = [r"$\underline{\tau = %.2E}$" % Decimal(cutoff) for cutoff in cutoffs]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=[
            [],
            [],
            [],
            title_text_str_all[0],
            title_text_str_all[5],
            [],
            [],
            [],
            title_text_str_all[1],
            title_text_str_all[6],
            [],
            [],
            [],
            title_text_str_all[2],
            title_text_str_all[7],
            [],
            [],
            [],
            title_text_str_all[3],
            title_text_str_all[8],
            [],
            [],
            [],
            title_text_str_all[4],
            title_text_str_all[9],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.11,
    )

    color_sct = "white"
    color_lsa = color_darkgray_hex
    marker_pattern_shape = "/"
    bar_spacing = 1
    fontsize = 16

    ### First half
    #################

    if show_figure2 or save_figure2:
        showlegend = True
        xpos = [0.052, 0.276, 0.5, 0.724, 0.948]
        ipos = 0
        # y_sct_all = {row+1: {col+1: [] for col in range(ncols-2)} for row in range(nrows) }
        # y_lsa_all = {row + 1: {col + 1: [] for col in range(ncols - 2)} for row in range(nrows)}
        for col, cutoff in enumerate(cutoffs):
            for row, num_params_ranking in enumerate(overlap_params):
                data = overlaps[cutoff][num_params_ranking]
                height_cf = data["cf"]["overlap"]
                width = [
                    data["tech"]["overlap"] / height_cf,
                    data["bio"]["overlap"] / height_cf,
                    1,
                ]
                x = [
                    width[0] / 2,
                    width[0] + bar_spacing + width[1] / 2,
                    width[0] + width[1] + 2 * bar_spacing + width[2] / 2,
                ]
                y_overlap = [height_cf, height_cf, height_cf]
                y_sct = [
                    data["tech"]["len_sct"] / width[0] - height_cf,
                    data["bio"]["len_sct"] / width[1] - height_cf,
                    data["cf"]["len_sct"] / width[2] - height_cf,
                ]
                y_lsa = [
                    data["tech"]["len_lsa"] / width[0] - height_cf,
                    data["bio"]["len_lsa"] / width[1] - height_cf,
                    data["cf"]["len_lsa"] / width[2] - height_cf,
                ]
                # y_sct_all[row+1][col+1] = np.ceil(max(y_sct))
                # y_lsa_all[row + 1][col + 1] = np.ceil(max(y_lsa))
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y_overlap,
                        width=width,
                        marker=dict(
                            color=color_lsa,
                            pattern=dict(shape=marker_pattern_shape),
                        ),
                        name=r"$\text{# of overlapping inputs}$",
                        legendrank=2,
                        showlegend=showlegend,
                    ),
                    col=col + 1,
                    row=row + 1,
                )
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y_sct,
                        width=width,
                        marker=dict(
                            color=color_sct,
                        ),
                        name=r"$\text{# of inputs when screening with contributions}$",
                        legendrank=1,
                        showlegend=showlegend,
                    ),
                    col=col + 1,
                    row=row + 1,
                )
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=-np.array(y_lsa),
                        width=width,
                        marker=dict(
                            color=color_lsa,
                        ),
                        name=r"$\text{# of inputs when screening with local SA}$",
                        legendrank=3,
                        showlegend=showlegend,
                    ),
                    col=col + 1,
                    row=row + 1,
                )
                showlegend = False
                title_text_str = r"$\bigcap" + r" = {} + {} + {} = {}$".format(
                    data["tech"]["overlap"],
                    data["bio"]["overlap"],
                    data["cf"]["overlap"],
                    data["tech"]["overlap"]
                    + data["bio"]["overlap"]
                    + data["cf"]["overlap"],
                )
                text_sct = (
                    r"$\text{SCT = "
                    + "{}".format(
                        data["tech"]["overlap"],
                    )
                    + "$}"
                )
                fig.update_xaxes(title_text=title_text_str, row=row + 1, col=col + 1)

                if row == 0:
                    fig.add_annotation(
                        x=xpos[col],
                        y=1.08,  # annotation point
                        xref="paper",
                        yref="paper",
                        text=cutoffs_str[col],
                        showarrow=False,
                        xanchor="center",
                        font=dict(
                            size=fontsize,
                        ),
                    )
                    ipos += 1

        fig.update_layout(
            barmode="relative",
            width=200 * nrows,
            height=160 * ncols,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            margin=dict(l=0, r=30, t=80, b=0),
            legend=dict(
                x=0.5,
                y=-0.10,
                xanchor="center",
                font_size=14,
                orientation="h",
                traceorder="normal",
                itemsizing="constant",
                # bgcolor=color_lightgray_hex,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
        )
        fig.update_traces(
            marker=dict(
                line_color=color_darkgray_hex,
                line_width=1.6,
                pattern_fillmode="replace",
            )
        )
        fig.update_xaxes(
            showticklabels=False,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
        )
        fig.update_yaxes(
            visible=False,
            showticklabels=False,
        )
        for row in range(nrows):
            if overlap_params[row] == 1600:
                params_str = "1'600"
            else:
                params_str = str(overlap_params[row])
            title_text = (
                r"$\underline{k_{\text{inf}} = " + r"{}".format(params_str) + r"}$"
            )
            fig.update_yaxes(
                title_text=title_text,
                col=1,
                row=row + 1,
                visible=True,
                title_standoff=50,
                title_font_size=fontsize,
            )
            # for col in [1,2,3]:
            #     fig.update_yaxes(
            #         range=[-max(list(y_lsa_all[row+1].values())), max(list(y_sct_all[row+1].values()))],
            #         row=row+1,
            #         col=col
            #     )

        ### Second half
        ##################
        Ymin = min(Yall)
        Ymax = max(Yall)
        col = len(cutoffs) + 1
        scatter_str_screening = [r"$\text{Screening based on}$"] * 2
        scatter_str = [
            r"$\underline{\text{contributions}}$",
            r"$\underline{\text{local SA}}$",
        ]
        showlegend = True
        for screening_name, data_dict in val_dict.items():
            row = 1
            for num_params_ranking, Yinf in data_dict.items():
                fig.add_trace(
                    go.Scatter(
                        x=Yall,
                        y=Yinf,
                        # name=trace_name3,
                        mode="markers",
                        marker=dict(
                            color=color_blue_orange_av_rgb,
                            line=dict(
                                width=1,
                                color=color_purple_rgb,
                            ),
                            opacity=0.25,
                        ),
                        name=r"$\text{Scatter plot between } Y_{\text{all}} \text{ and } Y_{\text{inf}}$",
                        legendrank=4,
                        showlegend=showlegend,
                    ),
                    row=row,
                    col=col,
                )
                showlegend = False
                if row == len(overlap_params):
                    fig.update_xaxes(
                        title_text=r"$Y_\text{all}$",
                        title_standoff=10,
                        row=row,
                        col=col,
                    )
                fig.update_xaxes(
                    visible=True,
                    showticklabels=True,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=color_gray_hex,
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor=color_black_hex,
                    showline=True,
                    linewidth=1,
                    linecolor=color_gray_hex,
                    # color=color_darkgray_hex,
                    row=row,
                    col=col,
                    range=[Ymin, Ymax],
                )
                if col == len(cutoffs) + 2:
                    fig.update_yaxes(
                        title_text=r"$Y_\text{inf}$",
                        title_standoff=10,
                        row=row,
                        col=col,
                    )
                if col >= len(cutoffs) + 1:
                    fig.update_yaxes(
                        visible=True,
                        showticklabels=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor=color_gray_hex,
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor=color_black_hex,
                        showline=True,
                        linewidth=1,
                        linecolor=color_gray_hex,
                        # color=color_darkgray_hex,
                        range=[Ymin, Ymax],
                        side="right",
                        row=row,
                        col=col,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[Ymin, Ymax],
                        y=[Ymin, Ymax],
                        mode="lines",
                        marker=dict(
                            color="black",
                            # line=dict(
                            #     width=0.01,
                            #     color="black",
                            # ),
                        ),
                        showlegend=False,
                        opacity=0.4,
                    ),
                    row=row,
                    col=col,
                )
                if row == 1:
                    fig.add_annotation(
                        x=xpos[col - 1],
                        y=1.11,  # annotation point
                        xref="paper",
                        yref="paper",
                        text=scatter_str_screening[col - 4],
                        showarrow=False,
                        xanchor="center",
                        font=dict(
                            size=fontsize,
                        ),
                    )
                    fig.add_annotation(
                        x=xpos[col - 1],
                        y=1.08,  # annotation point
                        xref="paper",
                        yref="paper",
                        text=scatter_str[col - 4],
                        showarrow=False,
                        xanchor="center",
                        font=dict(
                            size=fontsize,
                        ),
                    )
                    ipos += 1
                row += 1
            col += 1

    if show_figure2:
        fig.show()
    if save_figure2:
        save_fig(fig, "sct_lsa_overlap", fig_format, write_dir_fig)

    ###############################################
    ### FIGURE 3: Validation for 1 to 20 params ###
    ###############################################

    write_dir_val_ranking = write_dir_arr / "validation_ranking"

    num_ranked_max = 20
    num_ranked_arr = np.arange(1, num_ranked_max + 1)

    Ysct, spearman_sct = [], []
    for r in num_ranked_arr:
        filename_sct = "validation.Y.{}inf.2000.100023423.TotalRanked.sct.hdf5".format(
            r
        )
        Ycurrent = read_hdf5_array(write_dir_val_ranking / filename_sct).flatten()
        Ysct.append(Ycurrent)
        s, _ = spearmanr(Ycurrent, Yall)
        spearman_sct.append(s)
    spearman_sct = np.array(spearman_sct)
    Ysct = np.array(Ysct)

    Ylsa, spearman_lsa = [], []
    for r in num_ranked_arr:
        filename_lsa = (
            "validation.Y.{}inf.2000.100023423.TotalRanked.localSA.hdf5".format(r)
        )
        Ycurrent = read_hdf5_array(write_dir_val_ranking / filename_lsa).flatten()
        Ylsa.append(Ycurrent)
        s, _ = spearmanr(Ycurrent, Yall)
        spearman_lsa.append(s)
    spearman_lsa = np.array(spearman_lsa)
    Ylsa = np.array(Ylsa)

    num_params_ranking = 20
    option = "sct"
    if option == "localSA":
        Yinf = Ylsa[num_params_ranking - 1, :]
    elif option == "sct":
        Yinf = Ysct[num_params_ranking - 1, :]

    def add_histogram(fig, Yall, Yinf, row, col, showlegend=False):

        bin_min = min(Yall)
        bin_max = max(Yall)
        num_bins = 60
        opacity = 0.65

        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq_all, bins_all = np.histogram(Yall, bins=bins_)
        freq_inf, bins_inf = np.histogram(Yinf, bins=bins_)

        fig.add_trace(
            go.Scatter(
                x=bins_all,
                y=freq_all,
                name=all_inputs_text,
                opacity=opacity,
                line=dict(color=color_all, width=1, shape="hvh"),
                legendrank=6,
                legendgroup="histograms",
                legendgrouptitle=dict(text="Histograms"),
                showlegend=showlegend,
                fill="tozeroy",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=bins_inf,
                y=freq_inf,
                name=inf_inputs_long_text,
                opacity=opacity,
                line=dict(color=color_inf, width=1, shape="hvh"),
                legendrank=7,
                legendgroup="histograms",
                showlegend=showlegend,
                fill="tozeroy",
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
            title_text=r"$\text{Frequency}$",
            range=[0, 180],
            title_standoff=2,
            row=row,
            col=col,
            side="right",
        )
        fig.update_xaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
            title_text=lca_scores_axis_title_short,
            range=[bin_min, bin_max],
            title_standoff=2,
            tickmode="array",
            tickvals=[200, 300],
            ticktext=[200, 300],
            row=row,
            col=col,
        )

    def add_scatter(fig, Yall, Yinf, row, col, showlegend=False):
        r, _ = spearmanr(Yall, Yinf)

        Ymin = min(Yall)
        Ymax = max(Yall)

        fig.add_trace(
            go.Scatter(
                x=Yall,
                y=Yinf,
                mode="markers",
                marker=dict(
                    color=color_blue_orange_av_rgb,
                    line=dict(
                        width=1,
                        color=color_purple_rgb,
                    ),
                    opacity=0.25,
                ),
                name=r"$\text{Scatter plot between } Y_{\text{all}} \text{ and } Y_{\text{inf}}$",
                legendgroup="scatter",
                legendgrouptitle=dict(text="Scatter plots"),
                showlegend=showlegend,
                legendrank=5,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=[Ymin, Ymax],
                y=[Ymin, Ymax],
                mode="lines",
                marker=dict(
                    color="black",
                ),
                showlegend=False,
                opacity=0.4,
                legendgroup="scatter",
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
            title_text=r"$Y_\text{all}$",
            range=[Ymin, Ymax],
            title_font_color=color_all,
            tickcolor=color_all,
            tickfont_color=color_all,
            title_standoff=2,
            tickmode="array",
            tickvals=[200, 300],
            ticktext=[200, 300],
            row=row,
            col=col,
        )

        fig.update_yaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
            title_text=r"$Y_\text{inf}$",
            range=[Ymin, Ymax],
            title_font_color=color_inf,
            tickcolor=color_inf,
            tickfont_color=color_inf,
            title_standoff=2,
            tickmode="array",
            tickvals=[200, 300],
            ticktext=[200, 300],
            side="right",
            row=row,
            col=col,
        )

    color_corr_diff = color_pink_rgb
    if show_figure3 or save_figure3:
        nrows = 5
        ncols = 6
        subplot_titles = [
            r"$\rho={:4.3}$".format(s)
            for s in [
                spearman_lsa[19],
                spearman_sct[19],
                spearman_lsa[1],
                spearman_sct[9],
                spearman_lsa[9],
            ]
        ]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[
                [
                    {
                        "rowspan": 2,
                        "colspan": 3,
                        "secondary_y": True,
                    },
                    None,
                    None,
                    {},
                    {},
                    {},
                ],
                [None, None, None, {}, {}, {}],
                [{}, {}, {}, {}, {}, {}],
                [{}, {}, {}, {}, {}, {}],
                [{}, {}, {}, {}, {}, {}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.09,
            subplot_titles=[
                None,
                None,
                subplot_titles[0],
                None,
                None,
                subplot_titles[1],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                subplot_titles[2],
                subplot_titles[3],
                subplot_titles[4],
            ],
            column_widths=[0.19, 0.19, 0.19, -0.1, 0.19, 0.19],
            row_heights=[0.24, 0.24, -0.1, 0.24, 0.24],
        )
        fig.add_trace(
            go.Scatter(
                x=num_ranked_arr,
                y=spearman_sct,
                mode="markers+lines",
                marker=dict(
                    color=color_darkgray_hex,
                    symbol="x",
                ),
                line=dict(
                    dash="dot",
                ),
                name=r"$\text{Ranking when screening with contributions}$",
                legendrank=2,
                legendgroup="main",
                legendgrouptitle=dict(text="Spearman correlation plot"),
                showlegend=True,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=num_ranked_arr[1:],
                y=spearman_sct[1:] - spearman_sct[:-1],
                mode="markers+lines",
                marker=dict(
                    color=color_corr_diff,
                    symbol="x",
                ),
                line=dict(
                    dash="dot",
                ),
                name=r"$\text{Increase in } \rho \text{ when screening with contributions}$",
                legendrank=4,
                legendgroup="main",
                showlegend=True,
            ),
            secondary_y=True,
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=num_ranked_arr,
                y=spearman_lsa,
                mode="markers+lines",
                marker=dict(
                    color=color_darkgray_hex,
                    symbol="circle",
                ),
                name=r"$\text{Ranking when screening with local SA}$",
                legendrank=1,
                legendgroup="main",
                showlegend=True,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=num_ranked_arr[1:],
                y=spearman_lsa[1:] - spearman_lsa[:-1],
                mode="markers+lines",
                marker=dict(
                    color=color_corr_diff,
                    symbol="circle",
                ),
                name=r"$\text{Increase in } \rho \text{ when screening with local SA}$",
                legendrank=3,
                legendgroup="main",
                showlegend=True,
            ),
            secondary_y=True,
            row=1,
            col=1,
        )

        fig.update_xaxes(
            title_text=r"$\text{Number of influential inputs}$",
            range=[-1, 21],
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text=r"$\text{Spearman correlation } \rho$",
            range=[-0.1, 1.1],
            secondary_y=False,
            row=1,
            col=1,
        )

        fig.update_yaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_corr_diff,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=color_corr_diff,
            showline=True,
            linewidth=1,
            linecolor=color_corr_diff,
            title_text=r"$\text{Relative increase in } \rho$",
            range=[-0.05, 0.2],
            secondary_y=True,
            title_font_color=color_corr_diff,
            tickcolor=color_corr_diff,
            tickfont_color=color_corr_diff,
            title_standoff=2,
            row=1,
            col=1,
        )

        ### Histograms
        ##############
        num_params_ranking = 2
        Yinf = Ysct[num_params_ranking - 1, :]
        add_histogram(fig, Yall, Yinf, 5, 1, showlegend=True)

        num_params_ranking = 10
        Yinf = Ysct[num_params_ranking - 1, :]
        add_histogram(fig, Yall, Yinf, 5, 2)

        num_params_ranking = 10
        Yinf = Ylsa[num_params_ranking - 1, :]
        add_histogram(fig, Yall, Yinf, 5, 3)

        num_params_ranking = 20
        Yinf = Ylsa[num_params_ranking - 1, :]
        add_histogram(fig, Yall, Yinf, 1, 6)

        num_params_ranking = 20
        Yinf = Ysct[num_params_ranking - 1, :]
        add_histogram(fig, Yall, Yinf, 2, 6)

        ### Scatter plots
        #################
        num_params_ranking = 2
        Yinf = Ysct[num_params_ranking - 1, :]
        add_scatter(fig, Yall, Yinf, 4, 1, showlegend=True)

        num_params_ranking = 10
        Yinf = Ysct[num_params_ranking - 1, :]
        add_scatter(fig, Yall, Yinf, 4, 2)

        num_params_ranking = 10
        Yinf = Ylsa[num_params_ranking - 1, :]
        add_scatter(fig, Yall, Yinf, 4, 3)

        num_params_ranking = 20
        Yinf = Ylsa[num_params_ranking - 1, :]
        add_scatter(fig, Yall, Yinf, 1, 5)

        num_params_ranking = 20
        Yinf = Ysct[num_params_ranking - 1, :]
        add_scatter(fig, Yall, Yinf, 2, 5)

        fig.update_xaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
        )
        fig.update_yaxes(
            visible=True,
            showticklabels=True,
            showgrid=True,
            gridwidth=1,
            gridcolor=color_gray_hex,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=color_black_hex,
            showline=True,
            linewidth=1,
            linecolor=color_gray_hex,
            secondary_y=False,
        )

        fig.update_layout(
            width=180 * (ncols - 1),
            height=150 * (nrows - 1),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(
                x=0.8,
                y=0.0,
                xanchor="center",
                font_size=14,
                orientation="v",
                traceorder="grouped",
                itemsizing="constant",
                # bgcolor=color_lightgray_hex,
                # opacity=opacity,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
        )

        line1 = (
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=0.156,
                y0=0.65,
                x1=0.156,
                y1=0.21,
                line=dict(color="Purple", width=3),
            ),
        )

    if show_figure3:
        fig.show()
    if save_figure3:
        save_fig(fig, "gsa_ranking", fig_format, write_dir_fig)

    ########################################################
    ### FIGURE 1: Uncertainty distribution of LCA scores ###
    ########################################################
    color_narrow = color_purple_rgb

    filepath_Y = write_dir_arr / "validation.Y.all.2000.100023423.hdf5"
    Y = read_hdf5_array(filepath_Y).flatten()
    bin_min = min(Y)
    bin_max = max(Y)

    filepath_Y_narrow = (
        write_dir_arr / "validation.Y.narrow.2000.100023423.20Reduced_scale2.exp2.hdf5"
    )
    Y_narrow = read_hdf5_array(filepath_Y_narrow).flatten()

    if show_figure4 or save_figure4:
        # default_Y = lca.score

        fig = go.Figure()

        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq, bins = np.histogram(Y, bins=bins_)
        freqn, binn = np.histogram(Y_narrow, bins=bins_)

        fig.add_trace(
            go.Scatter(
                x=bins,
                y=freq,
                name=r"$\text{All inputs vary with original uncertainties}$",
                opacity=opacity,
                line=dict(color=color_all, width=1, shape="hvh"),
                showlegend=True,
                fill="tozeroy",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=binn,
                y=freqn,
                name=r"$\text{All inputs vary with 20 reduced uncertainties}$",
                opacity=opacity,
                line=dict(color=color_narrow, width=1, shape="hvh"),
                showlegend=True,
                fill="tozeroy",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(Y) - np.std(Y), np.mean(Y) - np.std(Y)],
                y=[0, 210],
                mode="lines",
                name=r"$\text{mean } \pm \text{ standard deviation, original uncertainty}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dash",
                    color="black",
                    width=1,
                ),
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    np.mean(Y_narrow) - np.std(Y_narrow),
                    np.mean(Y_narrow) - np.std(Y_narrow),
                ],
                y=[0, 210],
                mode="lines",
                name=r"$\text{mean } \pm \text{ standard deviation, reduced uncertainty}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dot",
                    color=color_narrow,
                    width=2,
                ),
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[np.mean(Y) + np.std(Y), np.mean(Y) + np.std(Y)],
                y=[0, 210],
                mode="lines",
                name=r"$\text{mean } \pm \text{ standard deviation}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dash",
                    color="black",
                    width=1,
                ),
                showlegend=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    np.mean(Y_narrow) + np.std(Y_narrow),
                    np.mean(Y_narrow) + np.std(Y_narrow),
                ],
                y=[0, 210],
                mode="lines",
                name=r"$\text{mean } \pm \text{ standard deviation}$",
                opacity=opacity,
                # marker=dict(
                #     color="black",
                #     size=20,
                #     symbol="x",
                # ),
                line=dict(
                    dash="dot",
                    color=color_narrow,
                    width=2,
                ),
                showlegend=False,
            ),
        )

        fig.update_yaxes(title_text=r"$\text{Frequency}$", range=[-10, 210])
        fig.update_xaxes(
            title_text=lca_scores_axis_title,
        )

        # Both
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
            height=320,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                x=0.5,
                y=-1.0,
                orientation="v",
                xanchor="center",
                font=dict(size=14),
                # bgcolor=color_lightgray_hex,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if show_figure4:
            fig.show()
        if save_figure4:
            save_fig(fig, "lca_scores_uncertainty_narrow", fig_format, write_dir_fig)

    from gsa_framework.sensitivity_methods.saltelli_sobol import sobol_indices
    from gsa_framework.convergence_robustness_validation import Robustness

    option = "localSA"
    num_params_ranking = 200

    filepath_Y = write_dir_arr / "Y.saltelliSampling.319968.None.{}.hdf5".format(option)
    S_salt = sobol_indices(filepath_Y, num_params_ranking)

    fn_stability = "stability.S.saltelliGsa.saltelliSampling.319968Step6262.1000.None.{}.pickle".format(
        option
    )
    fp_stability = write_dir_arr / fn_stability
    S_dict_stability = read_pickle(fp_stability)
    stability_dicts = [S_dict_stability]
    st = Robustness(
        stability_dicts,
        write_dir,
    )

    total = S_salt["Total order"]
    total_argsort = np.argsort(total)[::-1]

    if show_figure5 or save_figure5:

        fig = go.Figure()

        x = np.arange(1, 201)
        y = total[total_argsort]
        width = st.confidence_intervals["total"][-1, total_argsort]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Total order",
                marker=dict(
                    color=color_darkgray_hex,
                    size=5,
                ),
                showlegend=False,
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates
                    symmetric=False,
                    array=width / 2,
                    arrayminus=width / 2,
                    visible=True,
                    color=color_darkgray_hex,
                    thickness=1.5,
                ),
            ),
        )

        fig.update_xaxes(title=r"$\text{Model inputs}$")
        fig.update_yaxes(title=r"$\text{Sobol total index}$")

        # Both
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
            range=[-0.03, 0.51],
        )
        fig.update_layout(
            width=400,
            height=650,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                x=0.5,
                y=-1.0,
                orientation="v",
                xanchor="center",
                font=dict(size=14),
                # bgcolor=color_lightgray_hex,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if show_figure5:
            fig.show()
        if save_figure5:
            save_fig(fig, "total200_{}".format(option), fig_format, write_dir_fig)

    inputs_colors = {
        0: color_darkgray_tuple,
        1: color_pink_tuple,
        2: color_blue_tuple,
        3: color_purple_tuple,
        4: color_gray_tuple,
        5: color_black_tuple,
        6: color_green_tuple,
        7: color_yellow_tuple,
        8: color_blue_orange_av_tuple,
        9: color_orange_tuple,
    }

    if show_figure6 or save_figure6:

        fig = go.Figure()
        inputs = np.arange(10)

        option = "sct"

        filepath_Y = write_dir_arr / "Y.saltelliSampling.319968.None.{}.hdf5".format(
            option
        )
        S_salt = sobol_indices(filepath_Y, num_params_ranking)

        fn_stability = "stability.S.saltelliGsa.saltelliSampling.319968Step6262.1000.None.{}.pickle".format(
            option
        )
        fp_stability = write_dir_arr / fn_stability
        S_dict_stability = read_pickle(fp_stability)
        stability_dicts = [S_dict_stability]
        st = Robustness(
            stability_dicts,
            write_dir,
        )

        total = S_salt["Total order"]
        total_argsort = np.argsort(total)[::-1]

        for input in inputs:

            color = inputs_colors[input]

            y = st.sa_mean_results["total"][:, total_argsort][:, input]
            x = st.iterations["total"]
            width = st.confidence_intervals["total"][:, total_argsort][:, input]
            lower = y - width / 2
            upper = y + width / 2

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    opacity=1,
                    showlegend=False,
                    legendrank=input + 1,
                    marker=dict(
                        color="rgba({},{},{},{})".format(
                            color[0],
                            color[1],
                            color[2],
                            1,
                        ),
                    ),
                    name=r"$\text{Rank " + "{}".format(input + 1) + "}$",
                ),
            )
            showlegend = False
            fig.add_trace(
                go.Scatter(
                    x=x,
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
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
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
            )

        fig.update_xaxes(title=r"$\text{Iterations}$")
        fig.update_xaxes(
            tickvals=np.arange(50000, 320000, 50000),
            ticktext=["50'000", "100'000", "150'000", "200'000", "250'000", "300'000"],
        )
        fig.update_yaxes(title=r"$\text{Sobol total order index}$")

        # Both
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
            range=[-0.03, 0.58],
        )
        fig.update_layout(
            # width=300,
            # height=400,
            width=350,
            height=300,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                x=0.5,
                y=1.3,
                orientation="h",
                xanchor="center",
                font=dict(size=14),
                # bgcolor=color_lightgray_hex,
                bordercolor=color_darkgray_hex,
                borderwidth=1,
                traceorder="normal",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if show_figure6:
            fig.show()
        if save_figure6:
            save_fig(fig, "convergence20_{}".format(option), fig_format, write_dir_fig)
        # if save_figure6: save_fig(fig, "convergence20_legend", fig_format, write_dir_fig)
