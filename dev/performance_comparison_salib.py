import numpy as np
from SALib.sample.saltelli import sample as salib_saltelli_sample
from SALib.sample.latin import sample as salib_latin_sample
from SALib.analyze.delta import analyze as salib_delta
from gsa_framework.sampling.get_samples import saltelli_samples, latin_hypercube_samples
from gsa_framework.sensitivity_methods.correlations import (
    corrcoef_parallel_stability_spearman,
)
from gsa_framework.sensitivity_methods.delta import (
    delta_indices_parallel_stability,
)
from pathlib import Path
from gsa_framework.utils import write_pickle, read_pickle
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dev.utils_paper_plotting import *
from scipy.stats import spearmanr


if __name__ == "__main__":
    calc_second_order = False
    seed = 55
    skip_values = 1000

    path_base = Path("write_files/paper_figures/salib_gsafr")
    write_dir_fig = Path("write_files/paper_figures")
    fig_format = ["pdf"]

    color_dict = {"salib": color_blue_rgb, "gsa_framework": color_orange_rgb}

    ### A. Saltelli Sampling
    ########################
    input_step = 100
    input_last = 1000
    inputs = np.arange(input_step, input_last + 1, input_step)
    saltelli_lcm = np.lcm.reduce(np.arange(1, len(inputs) + 1))
    iterations_const = saltelli_lcm * input_step

    inputs_const = 1000
    # saltelli_Mstep = saltelli_lcm // (inputs_const // input_step) // 10
    # saltelli_Mlast = saltelli_lcm // (inputs_const // input_step)
    saltelli_Mstep = 2
    saltelli_Mlast = 20
    saltelli_M = np.arange(saltelli_Mstep, saltelli_Mlast + 1, saltelli_Mstep)

    # 1. wrt number of inputs
    # -> salib
    filename = "saltelli.sampling.salib.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        saltelli_sampling_salib_inputs = read_pickle(filepath)
    else:
        saltelli_sampling_salib_inputs = {}
        for ind, k in enumerate(inputs):
            problem = {"num_vars": k, "bounds": [[0, 1] * k]}
            t1 = time.time()
            x = salib_saltelli_sample(
                problem, saltelli_lcm // (ind + 1), calc_second_order, seed, skip_values
            )
            t2 = time.time()
            saltelli_sampling_salib_inputs[k] = t2 - t1
        write_pickle(saltelli_sampling_salib_inputs, filepath)
    # -> gsa_framework
    filename = "saltelli.sampling.gsafr.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        saltelli_sampling_gsafr_inputs = read_pickle(filepath)
    else:
        saltelli_sampling_gsafr_inputs = {}
        for ind, k in enumerate(inputs):
            saltelli_iterations = (k + 2) * saltelli_lcm // (ind + 1)
            t1 = time.time()
            x = saltelli_samples(saltelli_iterations, k, skip_values)
            t2 = time.time()
            saltelli_sampling_gsafr_inputs[k] = t2 - t1
        write_pickle(saltelli_sampling_gsafr_inputs, filepath)

    # 2. wrt number of iterations
    # -> salib
    filename = "saltelli.sampling.salib.M{}.Mstep{}.k{}.pickle".format(
        saltelli_Mlast, saltelli_Mstep, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        saltelli_sampling_salib_iterations = read_pickle(filepath)
    else:
        saltelli_sampling_salib_iterations = {}
        problem = {"num_vars": inputs_const, "bounds": [[0, 1] * inputs_const]}
        for M in saltelli_M:
            N = int(M * (inputs_const + 2))
            t1 = time.time()
            x = salib_saltelli_sample(problem, M, calc_second_order, seed, skip_values)
            t2 = time.time()
            print(x.shape)
            saltelli_sampling_salib_iterations[N] = t2 - t1
        write_pickle(saltelli_sampling_salib_iterations, filepath)
    # -> gsa_framework
    filename = "saltelli.sampling.gsafr.M{}.Mstep{}.k{}.pickle".format(
        saltelli_Mlast, saltelli_Mstep, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        saltelli_sampling_gsafr_iterations = read_pickle(filepath)
    else:
        saltelli_sampling_gsafr_iterations = {}
        for M in saltelli_M:
            N = int(M * (inputs_const + 2))
            t1 = time.time()
            x = saltelli_samples(N, inputs_const, skip_values)
            t2 = time.time()
            print(x.shape)
            saltelli_sampling_gsafr_iterations[N] = t2 - t1
        write_pickle(saltelli_sampling_gsafr_iterations, filepath)

    ### B. Latin Hypercube sampling
    ###############################
    input_step = 100
    input_last = 1000
    inputs = np.arange(input_step, input_last + 1, input_step)
    iterations_const = 20000

    inputs_const = 1000
    iterations_last = iterations_const
    iterations_step = iterations_last // 10
    iterations = np.arange(iterations_step, iterations_last + 1, iterations_step)

    # 1. wrt number of inputs
    # -> salib
    filename = "latin.sampling.salib.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        latin_sampling_salib_inputs = read_pickle(filepath)
    else:
        latin_sampling_salib_inputs = {}
        for k in inputs:
            problem = {"num_vars": k, "bounds": [[0, 1] * k]}
            t1 = time.time()
            x = salib_latin_sample(problem, iterations_const, seed)
            t2 = time.time()
            print(x.shape)
            latin_sampling_salib_inputs[k] = t2 - t1
        write_pickle(latin_sampling_salib_inputs, filepath)
    # -> gsa_framework
    filename = "latin.sampling.gsafr.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        latin_sampling_gsafr_inputs = read_pickle(filepath)
    else:
        latin_sampling_gsafr_inputs = {}
        for k in inputs:
            t1 = time.time()
            x = latin_hypercube_samples(iterations_const, k, seed)
            t2 = time.time()
            print(x.shape)
            latin_sampling_gsafr_inputs[k] = t2 - t1
        write_pickle(latin_sampling_gsafr_inputs, filepath)

    # 2. wrt number of iterations
    # -> salib
    filename = "latin.sampling.salib.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        latin_sampling_salib_iterations = read_pickle(filepath)
    else:
        latin_sampling_salib_iterations = {}
        problem = {"num_vars": inputs_const, "bounds": [[0, 1] * inputs_const]}
        for N in iterations:
            t1 = time.time()
            x = salib_latin_sample(problem, N, seed)
            t2 = time.time()
            print(x.shape)
            latin_sampling_salib_iterations[N] = t2 - t1
        write_pickle(latin_sampling_salib_iterations, filepath)
    # -> gsa_framework
    filename = "latin.sampling.gsafr.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        latin_sampling_gsafr_iterations = read_pickle(filepath)
    else:
        latin_sampling_gsafr_iterations = {}
        for N in iterations:
            t1 = time.time()
            x = latin_hypercube_samples(N, inputs_const, seed)
            t2 = time.time()
            print(x.shape)
            latin_sampling_gsafr_iterations[N] = t2 - t1
        write_pickle(latin_sampling_gsafr_iterations, filepath)

    ### C. GSA: correlation coefficients
    ####################################
    input_step = 100
    input_last = 1000
    inputs = np.arange(input_step, input_last + 1, input_step)
    iterations_const = 20000

    inputs_const = 1000
    iterations_last = iterations_const
    iterations_step = iterations_last // 10
    iterations = np.arange(iterations_step, iterations_last + 1, iterations_step)

    # 1. wrt number of inputs
    # -> scipy
    filename = "spearman.gsa.scipy.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        spearman_gsa_scipy_inputs = read_pickle(filepath)
    else:
        spearman_gsa_scipy_inputs = {}
        y = np.random.rand(iterations_const)
        for k in inputs:
            x = np.random.rand(iterations_const, k)
            t1 = time.time()
            s, _ = spearmanr(x, y)
            s = s[:-1, -1]
            t2 = time.time()
            print(s.shape)
            spearman_gsa_scipy_inputs[k] = t2 - t1
        write_pickle(spearman_gsa_scipy_inputs, filepath)
    # -> gsa_framework
    filename = "spearman.gsa.gsafr.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        spearman_gsa_gsafr_inputs = read_pickle(filepath)
    else:
        spearman_gsa_gsafr_inputs = {}
        y = np.random.rand(iterations_const)
        for k in inputs:
            x = np.random.rand(iterations_const, k)
            t1 = time.time()
            s = corrcoef_parallel_stability_spearman(y, x)["spearman"]
            t2 = time.time()
            print(s.shape)
            spearman_gsa_gsafr_inputs[k] = t2 - t1
        write_pickle(spearman_gsa_gsafr_inputs, filepath)

    # 2. wrt number of iterations
    # -> salib
    filename = "spearman.gsa.scipy.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        spearman_gsa_scipy_iterations = read_pickle(filepath)
    else:
        spearman_gsa_scipy_iterations = {}
        problem = {"num_vars": inputs_const, "bounds": [[0, 1] * inputs_const]}
        for N in iterations:
            y = np.random.rand(N)
            x = np.random.rand(N, inputs_const)
            t1 = time.time()
            s, _ = spearmanr(x, y)
            s = s[:-1, -1]
            t2 = time.time()
            print(s.shape)
            spearman_gsa_scipy_iterations[N] = t2 - t1
        write_pickle(spearman_gsa_scipy_iterations, filepath)
    # -> gsa_framework
    filename = "spearman.gsa.gsafr.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        spearman_gsa_gsafr_iterations = read_pickle(filepath)
    else:
        spearman_gsa_gsafr_iterations = {}
        for N in iterations:
            y = np.random.rand(N)
            x = np.random.rand(N, inputs_const)
            t1 = time.time()
            s = corrcoef_parallel_stability_spearman(y, x)["spearman"]
            t2 = time.time()
            print(s.shape)
            spearman_gsa_gsafr_iterations[N] = t2 - t1
        write_pickle(spearman_gsa_gsafr_iterations, filepath)

    ### D. GSA: delta indices
    #########################
    input_step = 100
    input_last = 1000
    inputs = np.arange(input_step, input_last + 1, input_step)
    iterations_const = 20000

    inputs_const = 1000
    iterations_last = iterations_const
    iterations_step = iterations_last // 10
    iterations = np.arange(iterations_step, iterations_last + 1, iterations_step)

    num_resamples = 1

    # 1. wrt number of inputs
    # -> scipy
    filename = "delta.gsa.salib.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        delta_gsa_salib_inputs = read_pickle(filepath)
    else:
        delta_gsa_salib_inputs = {}
        y = np.random.rand(iterations_const)
        for k in inputs:
            x = np.random.rand(iterations_const, k)
            problem = {"num_vars": k, "names": np.arange(k)}
            t1 = time.time()
            s = salib_delta(problem, x, y, num_resamples=num_resamples)["delta"]
            t2 = time.time()
            print(s.shape)
            delta_gsa_salib_inputs[k] = t2 - t1
        write_pickle(delta_gsa_salib_inputs, filepath)
    # -> gsa_framework
    filename = "delta.gsa.gsafr.k{}.kstep{}.N{}.pickle".format(
        input_last, input_step, iterations_const
    )
    filepath = path_base / filename
    if filepath.exists():
        delta_gsa_gsafr_inputs = read_pickle(filepath)
    else:
        delta_gsa_gsafr_inputs = {}
        y = np.random.rand(iterations_const)
        for k in inputs:
            x = np.random.rand(iterations_const, k)
            t1 = time.time()
            s = delta_indices_parallel_stability(y, x, num_resamples=num_resamples)[
                "delta"
            ]
            t2 = time.time()
            print(s.shape)
            delta_gsa_gsafr_inputs[k] = t2 - t1
        write_pickle(delta_gsa_gsafr_inputs, filepath)

    # 2. wrt number of iterations
    # -> salib
    filename = "delta.gsa.salib.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        delta_gsa_salib_iterations = read_pickle(filepath)
    else:
        delta_gsa_salib_iterations = {}
        problem = {"num_vars": inputs_const, "names": np.arange(inputs_const)}
        for N in iterations:
            y = np.random.rand(N)
            x = np.random.rand(N, inputs_const)
            t1 = time.time()
            s = salib_delta(problem, x, y, num_resamples=num_resamples)["delta"]
            t2 = time.time()
            print(s.shape)
            delta_gsa_salib_iterations[N] = t2 - t1
        write_pickle(delta_gsa_salib_iterations, filepath)
    # -> gsa_framework
    filename = "spearman.gsa.gsafr.N{}.Nstep{}.k{}.pickle".format(
        iterations_last, iterations_step, inputs_const
    )
    filepath = path_base / filename
    if filepath.exists():
        delta_gsa_gsafr_iterations = read_pickle(filepath)
    else:
        delta_gsa_gsafr_iterations = {}
        for N in iterations:
            y = np.random.rand(N)
            x = np.random.rand(N, inputs_const)
            t1 = time.time()
            s = delta_indices_parallel_stability(y, x, num_resamples=num_resamples)[
                "delta"
            ]
            t2 = time.time()
            print(s.shape)
            delta_gsa_gsafr_iterations[N] = t2 - t1
        write_pickle(delta_gsa_gsafr_iterations, filepath)

    plotting_dict = {
        "saltelli.sampling.vary_inputs": {
            "name": r"$\text{Saltelli sampling}$",
            "color": color_blue_rgb,
            "dash": "solid",
            "symbol": "circle",
        },
        "saltelli.sampling.vary_iterations": {
            "name": r"$\text{Saltelli sampling}$",
            "color": color_blue_rgb,
            "dash": "solid",
            "symbol": "circle",
        },
        "latin.sampling.vary_inputs": {
            "name": r"$\text{Latin hypercube sampling}$",
            "color": color_blue_rgb,
            "dash": "dash",
            "symbol": "diamond-tall",
        },
        "latin.sampling.vary_iterations": {
            "name": r"$\text{Latin hypercube sampling}$",
            "color": color_blue_rgb,
            "dash": "dash",
            "symbol": "diamond-tall",
        },
        "spearman.gsa.vary_inputs": {
            "name": r"$\text{Spearman correlations}$",
            "color": color_purple_rgb,
            "dash": "solid",
            "symbol": "circle",
        },
        "spearman.gsa.vary_iterations": {
            "name": r"$\text{Spearman correlations}$",
            "color": color_purple_rgb,
            "dash": "solid",
            "symbol": "circle",
        },
        "delta.gsa.vary_inputs": {
            "name": r"$\text{Delta indices}$",
            "color": color_blue_rgb,
            "dash": "dot",
            "symbol": "x",
        },
        "delta.gsa.vary_iterations": {
            "name": r"$\text{Delta indices}$",
            "color": color_blue_rgb,
            "dash": "dot",
            "symbol": "x",
        },
    }

    data = {
        "saltelli.sampling.vary_inputs": {
            "salib": saltelli_sampling_salib_inputs,
            "gsafr": saltelli_sampling_gsafr_inputs,
        },
        "saltelli.sampling.vary_iterations": {
            "salib": saltelli_sampling_salib_iterations,
            "gsafr": saltelli_sampling_gsafr_iterations,
        },
        "latin.sampling.vary_inputs": {
            "salib": latin_sampling_salib_inputs,
            "gsafr": latin_sampling_gsafr_inputs,
        },
        "latin.sampling.vary_iterations": {
            "salib": latin_sampling_salib_iterations,
            "gsafr": latin_sampling_gsafr_iterations,
        },
        # "spearman.gsa.vary_inputs": {
        #     'salib': spearman_gsa_scipy_inputs,
        #     'gsafr': spearman_gsa_gsafr_inputs,
        # },
        # "spearman.gsa.vary_iterations": {
        #     'salib': spearman_gsa_scipy_iterations,
        #     'gsafr': spearman_gsa_gsafr_iterations,
        # },
        "delta.gsa.vary_inputs": {
            "salib": delta_gsa_salib_inputs,
            "gsafr": delta_gsa_gsafr_inputs,
        },
        "delta.gsa.vary_iterations": {
            "salib": delta_gsa_salib_iterations,
            "gsafr": delta_gsa_gsafr_iterations,
        },
    }

    nrows = 2
    ncols = 2
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=False,
        subplot_titles=["Vary inputs", "Vary iterations"],
    )

    for k, v in data.items():
        if "vary_inputs" in k:
            col = 1
        else:
            col = 2
        if ".sampling." in k:
            row = 1
        else:
            row = 2
        for implementation, dict_ in v.items():
            x = list(dict_.keys())
            y = list(dict_.values())
            if implementation == "salib":
                symbol = "circle"
                dash = "solid"
                if col == 1:
                    showlegend = True
                else:
                    showlegend = False
            else:
                symbol = "x"
                dash = "dot"
                showlegend = False
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(color=plotting_dict[k]["color"], symbol=symbol),
                    line=dict(
                        dash=dash,
                    ),
                    showlegend=showlegend,
                    name=plotting_dict[k]["name"],
                ),
                row=row,
                col=col,
            )

    fig.update_yaxes(title=r"$\text{Time, [s]}$")
    fig.update_xaxes(title=r"$\text{Number of inputs}$", col=1)
    fig.update_xaxes(title=r"$\text{Number of iterations}$", col=2)

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
        width=800,
        height=nrows * 400,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            x=0.5,
            y=-0.22,
            xanchor="center",
            font_size=14,
            orientation="h",
            itemsizing="constant",
        ),
    )
    # fig.show()

    # Speed up gains
    nrows = 1
    ncols = 2
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        # subplot_titles=[r"$\text{Varying number of model inputs}$", r"$\text{Varying number of iterations}$"],
        shared_yaxes=True,
    )
    showlegend = True
    for k, v in data.items():
        if ".vary_inputs" in k:
            col = 1
            xx1 = np.array(list(v["salib"].keys()))
            showlegend = True
        else:
            col = 2
            xx2 = np.array(list(v["salib"].keys()))
            showlegend = False
        x1 = np.array(list(v["salib"].keys()))
        x2 = np.array(list(v["gsafr"].keys()))
        y1 = np.array(list(v["salib"].values()))
        y2 = np.array(list(v["gsafr"].values()))
        ratio = y1 / y2
        assert np.all(x1 == x2)
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=ratio,
                name=plotting_dict[k]["name"],
                mode="markers + lines",
                marker=dict(
                    color=plotting_dict[k]["color"],
                    symbol=plotting_dict[k]["symbol"],
                ),
                legendgroup=plotting_dict[k]["name"],
                showlegend=showlegend,
                line=dict(
                    dash=plotting_dict[k]["dash"],
                ),
            ),
            row=1,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=xx1[[0, -1]],
            y=[1, 1],
            name=plotting_dict[k]["name"],
            mode="lines",
            marker=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xx2[[0, -1]],
            y=[1, 1],
            name=plotting_dict[k]["name"],
            mode="lines",
            marker=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(type="log")
    fig.update_yaxes(title=r"$\text{Performance gain}$", col=1)
    fig.update_xaxes(title=r"$\text{Number of model inputs}$", col=1)
    fig.update_xaxes(title=r"$\text{Number of iterations}$", col=2)

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
        width=ncols * 300,
        height=nrows * 300,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            x=0.5,
            y=-0.28,
            xanchor="center",
            font_size=14,
            orientation="h",
            itemsizing="constant",
        ),
    )
    fig.show()
    save_fig(fig, "performance_comparison_gsafr_salib", fig_format, write_dir_fig)
