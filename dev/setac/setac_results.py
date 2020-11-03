from pathlib import Path
import pickle
from gsa_framework.utils_setac_lca import get_xgboost_params
import plotly.graph_objects as go
import numpy as np

num_params = 20
model_seed = 3333

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

path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/paper_gsa")
path_setac = path_base / "setac_gsa"
path_merlin = path_setac / "merlin"
path_model_dir = path_setac / "regression" / "{}_model".format(model_seed)
filepath_row_acts_names = path_model_dir / "row_acts_names.pickle"
filepath_col_acts_names = path_model_dir / "col_acts_names.pickle"
filepath_tech_params = path_model_dir / "tech_params.pickle"
filepath_params_yes_0 = path_merlin / "params_yes_0.pickle"
with open(filepath_row_acts_names, "rb") as f:
    row_acts_names = pickle.load(f)
with open(filepath_col_acts_names, "rb") as f:
    col_acts_names = pickle.load(f)
with open(filepath_tech_params, "rb") as f:
    tech_params = pickle.load(f)
with open(filepath_params_yes_0, "rb") as f:
    params_yes_0 = pickle.load(f)

model, params_yes_xgboost, importance_dict = get_xgboost_params(
    path_model_dir, params_yes_0
)

act_in_names_ = []
act_out_names_ = []
name_ind = 0
unit_ind = 1
loc_ind = 2
location_in, location_out = [], []
for k, param in enumerate(tech_params):
    act_in_names_.append(row_acts_names[k][name_ind])
    location_in.append(row_acts_names[k][loc_ind])
    act_out_names_.append(col_acts_names[k][name_ind])
    location_out.append(col_acts_names[k][loc_ind])
act_in_names = [act_plotting_dict.get(act, act) for act in act_in_names_]
act_out_names = [act_plotting_dict.get(act, act) for act in act_out_names_]

importances = np.array(list(importance_dict.values()))

annotations = []
product_names_red = [0]
product_names_green = [1, 2, 5, 7, 9, 19]
product_names_blue = [6, 17]
product_names_purple = [3, 8, 10, 12, 13, 14, 15, 16]
product_names_orange = [4, 11, 18]

for i in range(num_params):
    # Set color depending in database
    color = "black"
    if i in product_names_red:
        color = "#EF553B"
    elif i in product_names_green:
        color = "#1C8356"
    elif i in product_names_blue:
        color = "#1F77B4"
    elif i in product_names_purple:
        color = "#782AB6"
    # elif i in product_names_orange:
    #     color = "#F58518"

    ann_input = dict(
        x=0.08,
        y=i - 0.15,
        xref="x",
        yref="y",
        text="FROM  {}, {}".format(act_in_names[i], location_in[i]),
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font_size=11,
        font_color=color,
    )
    ann_output = dict(
        x=0.08,
        y=i + 0.15,
        xref="x",
        yref="y",
        text="TO      {}, {}".format(act_out_names[i], location_out[i]),
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font_size=11,
        font_color=color,
    )
    ann_text = dict(
        x=0.04,
        y=i,
        xref="x",
        yref="y",
        text=i + 1,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font_size=14,
        font_color=color,
    )
    annotations.append(ann_input)
    annotations.append(ann_output)
    annotations.append(ann_text)

annotations.append(
    dict(
        x=0.42,
        y=-1.8,
        xref="x",
        yref="y",
        text="Technosphere exchanges",
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font_size=16,
    )
)
annotations.append(
    dict(
        x=-1 / 2,
        y=-1.8,
        xref="x",
        yref="y",
        text="Sensitivity indices",
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font_size=16,
    )
)
xtickvals = -np.linspace(0, 1, 6, endpoint=True)
xticktext = [0, 0.2, 0.4, 0.6, 0.8, 1]
for j in range(xtickvals.shape[0]):
    annotations.append(
        dict(
            x=xtickvals[j],
            y=-1,
            xref="x",
            yref="y",
            text=xticktext[j],
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font_size=14,
        )
    )


fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=-importances[:num_params],
        y=np.arange(num_params),
        orientation="h",
        opacity=0.65,
        showlegend=False,
        marker=dict(color="#A95C9A"),
    ),
)
fig.add_trace(
    go.Scatter(
        x=[-1.01, 1.01],
        y=[-0.6, -0.6],
        line=dict(color="white"),
        showlegend=False,
    ),
)

fig.update_layout(
    width=700,
    height=num_params * 28 + 26,
    annotations=annotations,
    bargap=0.25,
    xaxis=dict(
        showticklabels=False,
    ),
    yaxis=dict(
        autorange="reversed",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.show()

fig.write_image("influential_exchanges.pdf")
