from pathlib import Path
from gsa_framework.utils import *
from gsa_framework.stability_convergence_metrics import Stability
from gsa_framework.plotting import plot_max_min_band_many, plot_ranking_convergence_many
from gsa_framework.sensitivity_analysis.saltelli_sobol import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jenkspy


# 1. Choose which stability dictionaries to include
path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files")

# 1. Models
# models = ["1000_morris4", "5000_morris4", "10000_morris4", "10000_lca"]
models = ["1000_morris4"]
data_dicts, ranking_stats = {}, {}
ranking_stats_rho = {}
for model in models:
    if "morris" in model:
        write_dir = path_base / model
    elif "lca" in model:
        write_dir = path_base / "lca_model_food_10000"
    write_arr = write_dir / "arrays"
    files = [x for x in write_arr.iterdir() if x.is_file() and "stability." in x.name]
    files = sorted(files)
    stability_dicts = []
    for file in files:
        stability_dict = read_pickle(file)
        if "correlations" in file.name:
            stability_dict_abs = {}
            for k, v in stability_dict.items():
                stability_dict_abs[k] = {"spearman": np.abs(v["spearman"])}
            stability_dict = stability_dict_abs
        stability_dicts.append(stability_dict)
    st = Stability(stability_dicts, write_dir, num_ranks=16)
    # data_dicts[model] = st.confidence_intervals_max
    ranking_stats_rho[model] = {}
    for i in range(6):
        rho = "rho{}".format(i + 1)
        ranking_stats_rho[model].update(
            {rho: st.stat_ranking(rho_name=rho, which_ranking="clustered")}
        )

# fig1 = max_min_band_many(data_dicts)
fig2 = plot_ranking_convergence_many(ranking_stats_rho)


def numerator_rho1(Rj, Rk):
    M = len(Rj)
    numerator = 3 * (Rj - Rk) ** 2 / M / (M ** 2 - 1)
    return numerator


def numerator_rho2(Rj, Rk):
    expr = (Rj - Rk) ** 2 * (1 / (Rj + 1) + 1 / (Rk + 1))
    numerator = expr / np.max(expr)
    return numerator


def numerator_rho3(Rj, Rk):
    expr = (Rj - Rk) ** 2 / (Rj + 1) / (Rk + 1)
    numerator = expr / np.max(expr)
    return numerator


def numerator_rho4(Rj, Rk):
    expr = (Rj - Rk) ** 2 / (Rj + 1 + Rk + 1)
    numerator = expr / np.max(expr)
    return numerator


def numerator_rho5(Rj, Rk):
    M = len(Rj)
    SSarr = 1 / np.arange(1, M + 1)
    SSsum = np.cumsum(SSarr[::-1])[::-1]
    SSj = SSsum[Rj]
    SSk = SSsum[Rk]
    expr = (SSj - SSk) ** 2
    numerator = expr / 4 / (M - SSsum[0])
    return numerator


def numerator_rho6(Rj, Rk, Sj, Sk):
    diff = np.abs(Rj - Rk)
    Sjk = np.vstack([Sj, Sk])
    maxs2 = np.max(Sjk, axis=0) ** 2
    numerator = (diff * maxs2) / sum(maxs2)
    return numerator


num_params = 1000
write_dir = path_base / "{}_morris4".format(num_params) / "arrays"
if num_params == 1000:
    fp_stab = (
        write_dir
        / "stability.S.correlationsGsa.randomSampling.4000Step80.60.3407.pickle"
    )
    stab = np.abs(read_pickle(fp_stab)[3920]["spearman"])
    # fp_stab = write_dir / "stability.S.saltelliGsa.saltelliSampling.99198Step1002.60.None.pickle"
    # stab = read_pickle(fp_stab)[98196]['total']
Sj = stab[0, :]
Sk = stab[2, :]
Rj = np.argsort(Sj)[-1::-1]
Rk = np.argsort(Sk)[-1::-1]

num_bins = 100
Sj_num_bins, Sk_num_bins = num_bins, num_bins
Sj_bins_linspace = np.linspace(min(Sj), max(Sj), Sj_num_bins, endpoint=True)
Sj_freq, Sj_bins = np.histogram(Sj, bins=Sj_bins_linspace)
Sk_bins_linspace = np.linspace(min(Sk), max(Sk), Sk_num_bins, endpoint=True)
Sk_freq, Sk_bins = np.histogram(Sk, bins=Sk_bins_linspace)
num_ranks = 8

num1 = numerator_rho1(Rj, Rk)
num2 = numerator_rho2(Rj, Rk)
num3 = numerator_rho3(Rj, Rk)
num4 = numerator_rho4(Rj, Rk)
num5 = numerator_rho5(Rj, Rk)
num6 = numerator_rho6(Rj, Rk, Sj, Sk)

x = np.arange(num_params)


data = [
    {"Sj": (x, Sj), "Sk": (x, Sk)},
    {"Sj_hist": (Sj_bins, Sj_freq), "Sk_hist": (Sk_bins, Sk_freq)},
    {"Rj": (x, Rj), "Rk": (x, Rk)},
    {"num1": (x, num1), "num2": (x, num2)},
    {"num3": (x, num3), "num4": (x, num4)},
    {"num5": (x, num5), "num6": (x, num6)},
]
fig = make_subplots(
    rows=len(data),
    cols=len(data[0]),
    shared_xaxes=False,
    vertical_spacing=0.1,
)

use_params = 100
for i, dict_ in enumerate(data):
    j = 0
    for name, tuple_ in dict_.items():
        fig.add_trace(
            go.Bar(
                x=tuple_[0][:use_params],
                y=tuple_[1][:use_params],
                showlegend=False,
            ),
            row=i + 1,
            col=j + 1,
        )
        if "hist" in name:
            xname, yname = name, "frequency"
            if "Sj" in name:
                breaks = jenkspy.jenks_breaks(Sj, nb_class=num_ranks)
            elif "Sk" in name:
                breaks = jenkspy.jenks_breaks(Sk, nb_class=num_ranks)
            fig.add_trace(
                go.Scatter(
                    x=breaks,
                    y=np.zeros(len(breaks)),
                    mode="markers",
                    showlegend=False,
                    marker=dict(symbol="x", color="red"),
                ),
                row=i + 1,
                col=j + 1,
            )
        else:
            xname, yname = "parameter", name
        fig.update_xaxes(title=xname, row=i + 1, col=j + 1)
        fig.update_yaxes(title=yname, row=i + 1, col=j + 1)
        j += 1

fig.show()
