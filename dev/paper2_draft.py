import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc
import stats_arrays as sa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gsa_framework.models.life_cycle_assessment import LCAModelBase
from gsa_framework.utils import read_hdf5_array, write_hdf5_array
from dev.utils_paper_plotting import *

iterations = 2000
seed = 349239
# path_base = Path('/data/user/kim_a')
path_base = Path("/Users/akim/PycharmProjects/gsa-framework-master/dev/write_files/")
write_dir = path_base / "protocol_gsa"
write_dir_fig = write_dir / "figures"

fig_format = ["pdf", "png"]

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

uncertain_tech_params = lca.tech_params[lca.tech_params["uncertainty_type"] > 1]
uncertain_bio_params = lca.bio_params[lca.bio_params["uncertainty_type"] > 1]
uncertain_cf_params = lca.cf_params[lca.cf_params["uncertainty_type"] > 1]

where_utech_params_lognormal = np.where(
    uncertain_tech_params["uncertainty_type"] == sa.LognormalUncertainty.id
)[0]

where_ubio_params_lognormal = np.where(
    uncertain_bio_params["uncertainty_type"] == sa.LognormalUncertainty.id
)[0]

where_ucf_params_lognormal = np.where(
    uncertain_cf_params["uncertainty_type"] == sa.LognormalUncertainty.id
)[0]

n_use_lognormal_list = [-1, 100000, 10000, 1000]

Y_dict = {}
subplot_titles = []
for n_use_lognormal in n_use_lognormal_list:
    where_utech_params_lognormal_partial = where_utech_params_lognormal[
        n_use_lognormal:
    ]  # only n_use params vary

    len_tech = len((uncertain_tech_params))
    len_bio = len((uncertain_bio_params))
    len_cf = len((uncertain_cf_params))

    uncertain_params_selected_where_dict = {
        "tech": np.setdiff1d(np.arange(len_tech), where_utech_params_lognormal_partial),
        "bio": np.setdiff1d(np.arange(len_bio), where_ubio_params_lognormal),
        "cf": np.setdiff1d(np.arange(len_cf), where_ucf_params_lognormal),
    }

    if n_use_lognormal == -1:
        uncertain_params_selected_where_dict["bio"] = np.arange(len_bio)

    uncertain_params = {
        "tech": uncertain_tech_params[uncertain_params_selected_where_dict["tech"]],
        "bio": uncertain_bio_params[uncertain_params_selected_where_dict["bio"]],
        "cf": uncertain_cf_params[uncertain_params_selected_where_dict["cf"]],
    }

    num_params = (
        len(uncertain_params["tech"])
        + len(uncertain_params["bio"])
        + len(uncertain_params["cf"])
    )

    filepath_Y = (
        write_dir
        / "arrays"
        / "si.Y.{}inf.{}.{}.lognormal{}.hdf5".format(
            num_params,
            iterations,
            seed,
            n_use_lognormal,
        )
    )
    print(filepath_Y.name)
    if filepath_Y.exists():
        Y = read_hdf5_array(filepath_Y).flatten()
    else:
        model = LCAModelBase(
            demand,
            uncertain_method,
            uncertain_params,
            # uncertain_params_selected_where_dict,
        )
        np.random.seed(seed)
        X = np.random.rand(iterations, num_params)
        Xr = model.rescale(X)
        Y = model(Xr)
        write_hdf5_array(Y, filepath_Y)
    Y_dict[n_use_lognormal] = Y
    if n_use_lognormal == -1:
        subplot_titles.append(r"$\text{All 408'741 inputs vary}$")
    else:
        subplot_titles.append(
            r"$\text{" + "{}".format(n_use_lognormal // 1000) + "'000 inputs vary}$"
        )

# colors_dict = {
#     1000: color_orange_rgb,
#     10000: color_green_rgb,
#     100000: color_darkgray_rgb,
#     -1: "red",
# }

show_figure = False
save_figure = True

bin_min = min(Y_dict[-1])
bin_max = max(Y_dict[-1])
num_bins = 60
opacity = 0.65

lca_scores_axis_title = r"$\text{LCIA scores, [kg CO}_2\text{-eq}]$"
showlegend = True


if show_figure or save_figure:
    default_Y = lca.score
    nrows = len(Y_dict)
    # fig = make_subplots(
    #     rows=nrows,
    #     cols=1,
    #     shared_xaxes=True,
    #     subplot_titles=subplot_titles,
    #     vertical_spacing=0.12,
    # )
    fig = go.Figure()

    # row = 1
    # for n_use_lognormal, Y in Y_dict.items():

    n_use_lognormal = -1
    Y = Y_dict[n_use_lognormal]

    bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    freq, bins = np.histogram(Y, bins=bins_)
    if n_use_lognormal == -1:
        color = color_blue_rgb
    else:
        color = color_darkgray_hex

    fig.add_trace(
        go.Scatter(
            x=bins,
            y=freq,
            name=n_use_lognormal,
            opacity=opacity,
            line=dict(color=color, width=1, shape="hvh"),
            showlegend=False,
            fill="tozeroy",
        ),
        # row=row,
        # col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[np.mean(Y)],
            y=[0],
            mode="markers",
            name=r"$\text{Distribution mean}$",
            opacity=opacity,
            marker=dict(
                color="black",
                size=20,
                symbol="x",
            ),
            showlegend=showlegend,
        ),
        # row=row,
        # col=1,
    )

    if default_Y is not None:
        trace_name_default = r"$\text{Deterministic LCIA score}$"
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
                showlegend=showlegend,
            ),
            # row=row,
            # col=1,
        )
    # showlegend = False
    # row += 1

    fig.update_yaxes(title_text=r"$\text{Frequency}$", range=[-10, 140])
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
    if showlegend:
        height = 300
        width = 600
    else:
        height = 180
        width = 400

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            x=0.5,
            y=-0.4,
            orientation="h",
            xanchor="center",
            font=dict(size=14),
            # bgcolor=color_lightgray_hex,
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if show_figure:
        fig.show()
    if save_figure:
        if showlegend:
            save_fig(fig, "lca_scores_uncertainty_si_legend", fig_format, write_dir_fig)
        else:
            save_fig(
                fig,
                "lca_scores_uncertainty_si_{}".format(n_use_lognormal),
                fig_format,
                write_dir_fig,
            )


print()
