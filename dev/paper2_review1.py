import bw2data as bd
import pandas as pd
from consumption_model_ch.utils import get_habe_filepath
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from dev.utils_paper_plotting import *

bd.projects.set_current("GSA for archetypes")
co_name = "swiss consumption 1.0"
co = bd.Database(co_name)
year_habe = co.metadata["year_habe"]
dir_habe = co.metadata["dir_habe"]

# 2. Extract total demand from HABE
path_beschrei = get_habe_filepath(dir_habe, year_habe, "Datenbeschreibung")
path_ausgaben = get_habe_filepath(dir_habe, year_habe, "Ausgaben")
path_mengen = get_habe_filepath(dir_habe, year_habe, "Mengen")

# change codes to be consistent with consumption database and Andi's codes
ausgaben = pd.read_csv(path_ausgaben, sep="\t")
mengen = pd.read_csv(path_mengen, sep="\t")
ausgaben.columns = [col.lower() for col in ausgaben.columns]
mengen.columns = [col.lower() for col in mengen.columns]
codes_co_db = sorted([act["code"] for act in co])
columns_a = ausgaben.columns.values
columns_m = [columns_a[0]]
for code_a in columns_a[1:]:
    code_m = code_a.replace("a", "m")
    if code_m in codes_co_db:
        columns_m.append(code_m)
    else:
        columns_m.append(code_a)
ausgaben.columns = columns_m

bread = ausgaben["m511103"].values / 12
milk = ausgaben["m511401"].values / 12
syrups = ausgaben["m512203"].values / 12
garlic = ausgaben["m511709"].values / 12

products = [[bread, milk], [syrups, garlic]]
units = [
    [r"$\text{Consumed amount, [kilogram]}$", r"$\text{Consumed amount, [litre]}$"],
    [r"$\text{Consumed amount, [litre]}$", r"$\text{Consumed amount, [kilogram]}$"],
]
subplot_titles = [
    r"$\text{Bread}$",
    r"$\text{Milk}$",
    r"$\text{Flavored syrups}$",
    r"$\text{Garlic}$",
]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=subplot_titles,
    horizontal_spacing=0.15,
)

num_bins = 80
for row, data in enumerate(products):
    for col, x in enumerate(data):
        bins_ = np.linspace(min(x), max(x), num_bins, endpoint=True)
        freq, bins = np.histogram(x, bins=bins_)
        fig.add_trace(
            go.Bar(
                x=bins, y=freq, showlegend=False, marker=dict(color=color_purple_rgb)
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.update_xaxes(
            title_text=units[row][col],
            row=row + 1,
            col=col + 1,
        )
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
    width=680,
    height=400,
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
    margin=dict(l=40, r=40, t=30, b=10),
)
fig.update_yaxes(title_text=r"$\text{Frequency}$")

# fig.show()

save_fig(fig, "hh_variability", ["pdf"], Path("write_files"))

print()
