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
scaling_factor = 3
num_params = 20
rows = 5
model_seed = 3333
iterations = 2000
act_plotting_dict = {
    "cheese production, soft, from cow milk": "cheese production, soft",
    "operation, housing system, pig, fully-slatted floor": "operation, housing, pig",
    "liquid manure storage and processing facility construction": "liquid manure facility constr.",
    "electricity voltage transformation from high to medium voltage": "voltage transf., high to med.",
    "market for housing system, pig, fully-slatted floor, per pig place": "market for housing, pig",
    "market for operation, housing system, pig, fully-slatted floor, per pig place": "market for oper., hous.",
    "market for housing system, cattle, tied, per animal unit": "market for housing, cattle",
    "soybean meal and crude oil production": "soybean meal, crude oil prod.",
    "operation, housing system, cattle, tied": "operation, housing, cattle",
    "housing system construction, pig, fully-slatted floor": "housing constr., pig",
    "electricity voltage transformation from medium to low voltage": "volt. transf. from med. to low",
    "housing system construction, cattle, tied": "housing constr., cattle",
    "market group for transport, freight, lorry, unspecified": "market for transport, freight, lorry",
    "market for liquid manure storage and processing facility": "market for manure stor. & process.",
    "market group for transport, freight, light commercial vehicle": "market for transport, freight, light veh.",
    "market for nitric acid, without water, in 50% solution state": "market for nitric acid, w/o water, 50%",
    "market for land tenure, arable land, measured as carbon net primary productivity, perennial crop": "market for land tenure, arable",
    "market group for electricity, low voltage": "market for electr., low volt.",
    "nutrient supply from calcium nitrate": "nutr. supply from calc. nitr.",
    "market for electricity, high voltage": "market for electr., high volt.",
    "maize grain, feed production": "maize grain, feed prod.",
    "soybean, feed production": "soybean, feed prod.",
    "milk production, from cow": "milk prod., from cow",
}

COLORS_DICT = {
    "parameters_standard": "#00CC96",
    "parameters_narrow": "#FFA15A",
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
mc = MCRandomNumberGenerator(tech_params, maximum_iterations=iterations, seed=89)
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
    vertical_spacing=0.25,
    horizontal_spacing=0.06,
)

k = 0
opacity_ = 0.75
for i in range(1, rows + 1):
    freq_row = np.zeros((1,))
    for j in range(1, cols + 1):

        x, x_narrow = X[:, k], X_narrow[:, k]
        bin_min, bin_max = min(np.hstack([x, x_narrow])), max(np.hstack([x, x_narrow]))
        bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
        freq, bins = np.histogram(x, bins=bins_)
        freq_narrow, bins_narrow = np.histogram(x_narrow, bins=bins_)
        freq_row = np.hstack([freq_row, freq_narrow])

        fig.add_trace(
            go.Bar(
                x=bins,
                y=freq,
                marker=dict(
                    color=COLORS_DICT["parameters_standard"],
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
                        color=COLORS_DICT["parameters_narrow"],
                        opacity=opacity_,
                    ),
                    showlegend=False,
                ),
                row=i,
                col=j,
            )
        fig.update_xaxes(
            title_text=units[k],
            range=[bin_min, bin_max],
            row=i,
            col=j,
        )
        k += 1
    for j in range(1, cols + 1):
        fig.update_yaxes(
            range=[min(freq_row), max(freq_row)],
            row=i,
            col=j,
        )

annotations = []
k = 0
for i in range(rows):
    for j in range(cols):
        if k == 0:
            xpos = fig.layout["xaxis"]["domain"][0] + 0.02
            ypos = fig.layout["yaxis"]["domain"][1] + 0.09
        else:
            xpos = fig.layout["xaxis{}".format(k + 1)]["domain"][0] + 0.02
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
            # xpos = fig.layout["xaxis"]["domain"][0] + 0.02
            ypos = fig.layout["yaxis"]["domain"][1] + 0.06
        else:
            # xpos = fig.layout["xaxis{}".format(k + 1)]["domain"][0] + 0.02
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
        ann_number = dict(
            x=xpos - 0.02,
            y=ypos + 0.02,
            xref="paper",
            yref="paper",
            text=str(k + 1),
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font_size=14,
        )
        annotations.append(ann_number)

        k += 1

fig.update_layout(
    annotations=annotations,
    width=1100,
    height=470,
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

fig.show()

if save_fig:
    if plot_narrow:
        filename = "parameters_histograms_narrowed_div{}_distr.pdf".format(
            scaling_factor
        )
    else:
        filename = "parameters_histograms_standard_div{}_distr.pdf".format(
            scaling_factor
        )
    filepath = path_setac / "figures" / filename
    fig.write_image(str(filepath))
