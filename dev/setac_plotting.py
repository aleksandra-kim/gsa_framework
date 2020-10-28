import plotly.graph_objects as go
import brightway2 as bw
from pathlib import Path
from copy import deepcopy
from stats_arrays import MCRandomNumberGenerator
import pickle
from plotly.subplots import make_subplots
import numpy as np

# TODO choose these parameters
save_fig = True
plot_narrow = False
scaling_factor = 8
num_params = 12
rows = 3
model_seed = 3333
iterations = 2000
act_plotting_dict = {
    "cheese production, soft, from cow milk": "cheese production, soft",
    "operation, housing system, pig, fully-slatted floor": "operation, housing, pig",
    "liquid manure storage and processing facility construction": "liquid manure facility construction",
    "electricity voltage transformation from high to medium voltage": "voltage transformation, high to medium",
    "market for housing system, pig, fully-slatted floor, per pig place": "market for housing, pig",
    "market for operation, housing system, pig, fully-slatted floor, per pig place": "market for operation, housing, pig",
    "market for housing system, cattle, tied, per animal unit": "market for housing, cattle",
    "soybean meal and crude oil production": "soybean meal, crude oil production",
    "operation, housing system, cattle, tied": "operation, housing, cattle",
}

COLORS_DICT = {
    "all": "#636EFA",
    "influential": "#EF553B",
    "scatter": "#00CC96",
}

path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa")
path_setac = path_base / "setac_gsa"
path_merlin = path_setac / "merlin"
path_model_dir = path_setac / "regression" / "{}_model".format(model_seed)
filepath_row_acts_names = path_model_dir / "row_acts_names.pickle"
filepath_col_acts_names = path_model_dir / "col_acts_names.pickle"
filepath_tech_params = path_model_dir / "tech_params.pickle"
with open(filepath_row_acts_names, "rb") as f:
    row_acts_names = pickle.load(f)
with open(filepath_col_acts_names, "rb") as f:
    col_acts_names = pickle.load(f)
with open(filepath_tech_params, "rb") as f:
    tech_params = pickle.load(f)

tech_params_narrow = deepcopy(tech_params)  # TODO understand this!
tech_params_narrow["scale"] = tech_params_narrow["scale"] / scaling_factor
mc = MCRandomNumberGenerator(tech_params, maximum_iterations=iterations)
mc_narrow = MCRandomNumberGenerator(tech_params_narrow, maximum_iterations=iterations)
X = np.array([list(next(mc)) for _ in range(iterations)])
X_narrow = np.array([list(next(mc_narrow)) for _ in range(iterations)])

units = []
act_in_names_ = []
act_out_names_ = []
name_ind = 0
unit_ind = 1
for k, param in enumerate(tech_params):
    units.append(row_acts_names[k][unit_ind])
    act_in_names_.append(row_acts_names[k][name_ind])
    act_out_names_.append(col_acts_names[k][name_ind])

act_in_names = [act_plotting_dict.get(act, act) for act in act_in_names_]
act_out_names = [act_plotting_dict.get(act, act) for act in act_out_names_]

cols = num_params // rows
num_bins = 60

fig = make_subplots(
    rows=rows,
    cols=cols,
    shared_yaxes=True,
    vertical_spacing=0.22,
    horizontal_spacing=0.06,
)

k = 0
opacity_ = 0.65
for i in range(1, rows + 1):
    for j in range(1, cols + 1):

        x, x_narrow = X[:, k], X_narrow[:, k]
        bin_min, bin_max = min(np.hstack([x, x_narrow])), max(np.hstack([x, x_narrow]))
        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq, bins = np.histogram(x, bins=bins_)
        freq_narrow, bins_narrow = np.histogram(x_narrow, bins=bins_)

        fig.add_trace(
            go.Bar(
                x=bins,
                y=freq,
                marker=dict(
                    color=COLORS_DICT["all"],
                    opacity=opacity_,
                ),
                showlegend=False,
            ),
            row=i,
            col=j,
        )
        if plot_narrow:
            fig.add_trace(
                go.Bar(
                    x=bins_narrow,
                    y=freq_narrow,
                    marker=dict(
                        color=COLORS_DICT["influential"],
                        opacity=opacity_,
                    ),
                    showlegend=False,
                ),
                row=i,
                col=j,
            )
        fig.update_xaxes(
            title_text=units[k],
            row=i,
            col=j,
        )
        k += 1

annotations = []
k = 0
for i in range(rows):
    for j in range(cols):
        if k == 0:
            xpos = fig.layout["xaxis"]["domain"][0]
            ypos = fig.layout["yaxis"]["domain"][1] + 0.09
        else:
            xpos = fig.layout["xaxis{}".format(k + 1)]["domain"][0]
            ypos = fig.layout["yaxis{}".format(k + 1)]["domain"][1] + 0.09
        annotation = dict(
            x=xpos,
            y=ypos,
            xref="paper",
            yref="paper",
            text="{:5} {}".format("FROM", act_in_names[k]),
            xanchor="left",
            yanchor="top",
            showarrow=False,
        )
        annotations.append(annotation)

        if k == 0:
            xpos = fig.layout["xaxis"]["domain"][0]
            ypos = fig.layout["yaxis"]["domain"][1] + 0.06
        else:
            xpos = fig.layout["xaxis{}".format(k + 1)]["domain"][0]
            ypos = fig.layout["yaxis{}".format(k + 1)]["domain"][1] + 0.06
        annotation = dict(
            x=xpos,
            y=ypos,
            xref="paper",
            yref="paper",
            text="{:7} {}".format("TO", act_out_names[k]),
            xanchor="left",
            yanchor="top",
            showarrow=False,
        )
        annotations.append(annotation)

        k += 1

fig.update_layout(
    annotations=annotations,
    width=1400,
    height=600,
    barmode="overlay",
    margin=dict(l=40, r=40, t=60, b=40),  # TODO change margins
)

# k = 0
# for i in range(rows):
#     for j in range(cols):
#         fig.update_xaxes(title_text=units[k], row=i, col=j, )
#         k += 1
fig.update_yaxes(
    title_text="Frequency",
    row=1,
    col=1,
)
fig.update_yaxes(
    title_text="Frequency",
    row=2,
    col=1,
)
fig.update_yaxes(
    title_text="Frequency",
    row=3,
    col=1,
)

# fig.show()

if save_fig:
    if plot_narrow:
        filename = "parameters_histograms_narrowed_{}_distr.pdf".format(scaling_factor)
    else:
        filename = "parameters_histograms_standard_{}_distr.pdf".format(scaling_factor)
    filepath = path_setac / "figures" / filename
    fig.write_image(str(filepath))
