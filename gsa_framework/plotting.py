import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_histogram_Y(
    Y,
    default_Y=None,
    bin_min=None,
    bin_max=None,
    num_bins=60,
    trace_name="Y",
    trace_name_default="Default value",
    color="#636EFA",
    color_default_Y="red",
    opacity=0.65,
    xaxes_title_text="Values",
):
    if bin_min is None:
        bin_min = min(Y)
    if bin_max is None:
        bin_max = max(Y)
    bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    freq, bins = np.histogram(Y, bins=bins_)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bins,
            y=freq,
            name=trace_name,
            opacity=opacity,
            marker=dict(color=color),
            showlegend=True,
        ),
    )
    if default_Y is not None:
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
    fig.update_layout(
        width=410,
        height=220,
        # width=600,
        # height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(x=0.6, y=0.96),
        # legend=dict(x=1.0, y=1),
    )
    fig.update_yaxes(title_text="Frequency")
    fig.update_xaxes(title_text=xaxes_title_text)
    return fig


def plot_histogram_Y1_Y2(
    Y1,
    Y2,
    default_Y=None,
    bin_min=None,
    bin_max=None,
    num_bins=60,
    trace_name1="Y1",
    trace_name2="Y2",
    color1="#636EFA",
    color2="#EF553B",
    color_default_Y="red",
    opacity=0.65,
    xaxes_title_text="Values",
    showlegend=True,
):
    if bin_min is None:
        bin_min = min(np.hstack([Y1, Y2]))
    if bin_max is None:
        bin_max = max(np.hstack([Y1, Y2]))
    bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    freq1, bins1 = np.histogram(Y1, bins=bins_)
    freq2, bins2 = np.histogram(Y2, bins=bins_)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bins1,
            y=freq1,
            name=trace_name1,
            opacity=opacity,
            marker=dict(color=color1),
            showlegend=showlegend,
        ),
    )
    fig.add_trace(
        go.Bar(
            x=bins2,
            y=freq2,
            name=trace_name2,
            opacity=opacity,
            marker=dict(color=color2),
            showlegend=showlegend,
        ),
    )

    if default_Y is not None:
        fig.add_trace(
            go.Scatter(
                x=[default_Y],
                y=[0],
                mode="markers",
                name="Default value",
                opacity=opacity,
                marker=dict(
                    color=color_default_Y,
                    size=20,
                    symbol="x",
                ),
                showlegend=True,
            ),
        )

    fig.update_layout(
        barmode="overlay",
        width=410,
        height=220,
        # width=600,
        # height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(x=0.6, y=0.96),
        # legend=dict(x=1.0, y=1),
    )
    fig.update_yaxes(title_text="Frequency")
    fig.update_xaxes(title_text=xaxes_title_text)
    return fig


def plot_correlation_Y1_Y2(
    Y1,
    Y2,
    start=0,
    end=50,
    trace_name1="Y1",
    trace_name2="Y2",
    trace_name3="Scatter plot",
    color1="#636EFA",
    color2="#EF553B",
    color3="#A95C9A",
    xaxes1_title_text=None,
    yaxes1_title_text="Values",
    xaxes2_title_text="Values",
    yaxes2_title_text="Values",
):
    x = np.arange(start, end)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Y1[start:end],
            name=trace_name1,
            mode="lines+markers",
            marker=dict(color=color1),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Y2[start:end],
            name=trace_name2,
            mode="lines+markers",
            marker=dict(color=color2),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Y1,
            y=Y2,
            name=trace_name3,
            mode="markers",
            marker=dict(
                color=color3,
                line=dict(
                    width=1,
                    color="#782e69",
                ),
            ),
            showlegend=False,
            opacity=0.65,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        width=800,
        height=220,
        legend=dict(x=0.4, y=1.0),  # on top
        xaxis1=dict(domain=[0.0, 0.63]),
        xaxis2=dict(domain=[0.78, 1.0]),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    if xaxes1_title_text is None:
        xaxes1_title_text = "Subset of {0}/{1} datapoints".format(
            end - start, Y1.shape[0]
        )
    fig.update_xaxes(
        title_text=xaxes1_title_text,
        row=1,
        col=1,
    )
    Ymin = min(np.hstack([Y1, Y2]))
    Ymax = max(np.hstack([Y1, Y2]))
    fig.update_yaxes(range=[Ymin, Ymax], title_text=yaxes1_title_text, row=1, col=1)

    fig.update_xaxes(
        range=[Ymin, Ymax],
        title_text=xaxes2_title_text,
        color=color1,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        range=[Ymin, Ymax],
        title_text=yaxes2_title_text,
        color=color2,
        row=1,
        col=2,
    )
    return fig


def plot_max_min_band_many(data_dicts):
    nrows = max([len(data_dict) for data_dict in data_dicts.values()])
    ncols = len(data_dicts)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    col = 1
    annotations = []
    for data_title, data_dict in data_dicts.items():
        fig = plot_max_min_band(data_dict, fig, col=col)
        annotations.append(
            dict(
                x=0,
                y=1.05,  # annotation point
                xref="x{}".format(col),
                yref="paper",
                text=data_title,
                showarrow=False,
                xanchor="left",
            ),
        )
        col += 1
    fig["layout"].update(annotations=annotations)
    fig.update_layout(
        width=400 * ncols,
        height=200 * nrows,
    )
    # fig.update_xaxes(type="log", range=[2, 6])
    # fig.update_yaxes(range=[-1, 1])
    fig.show()
    return fig


def plot_max_min_band(data_dict, fig=None, col=None):
    if fig is None:
        # Plotting
        nrows = len(data_dict)
        ncols = 1
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        col = 1
    else:
        nrows = len(fig._grid_ref)
        ncols = len(fig._grid_ref[0])
    opacity = 0.5
    row = 1
    for sa_name, data in data_dict.items():
        x = data["iterations"]
        width = data["width"]
        lower = -width / 2
        upper = +width / 2
        color = np.random.randint(low=50, high=255, size=3)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.zeros(len(x)),
                mode="lines",
                opacity=1,
                showlegend=False,
                marker=dict(
                    color="rgba({},{},{},{})".format(
                        color[0],
                        color[1],
                        color[2],
                        1,
                    ),
                ),
            ),
            row=row,
            col=col,
        )
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
            row=row,
            col=col,
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
            row=row,
            col=col,
        )

        # fig.add_trace(
        #     go.Scatter(
        #         x=[convergence_iterations[0], convergence_iterations[-1]],
        #         y=[CI_THRESHOLD/2, CI_THRESHOLD/2],
        #         mode="lines",
        #         opacity=1,
        #         showlegend=False,
        #         marker = dict(color='red'),
        #     ),
        #     row=row + 1,
        #     col=1,
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=[convergence_iterations[0], convergence_iterations[-1]],
        #         y=[-CI_THRESHOLD / 2, -CI_THRESHOLD / 2],
        #         mode="lines",
        #         opacity=1,
        #         showlegend=False,
        #         marker = dict(color='red'),
        #     ),
        #     row=row + 1,
        #     col=1,
        # )
        fig.update_yaxes(title_text=sa_name, row=row, col=col)
        # fig.update_yaxes(title_text="$Stat_{indices}$ ", row=row, col=col)
        row += 1

    fig.update_layout(
        width=400 * ncols,
        height=400 * nrows,
    )
    fig.update_xaxes(title_text="iterations", row=row - 1, col=col)
    return fig


def plot_ranking_convergence_many(
    data_dicts, y_name="mean", lower_name="q05", upper_name="q95"
):
    nrows = len(list(list(list(data_dicts.values())[0].values())[0].keys()))
    ncols = len(data_dicts)
    rho_names = list(list(data_dicts.values())[0].keys())
    colors = {}
    colors_list = [
        (27, 158, 119),
        (217, 95, 2),
        (117, 112, 179),
        (231, 41, 138),
        (102, 166, 30),
        (230, 171, 2),
    ]
    for i, rho_name in enumerate(rho_names):
        colors[rho_name] = colors_list[i]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=False,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    col = 1
    annotations = []
    for data_title, data_dict in data_dicts.items():
        fig = plot_ranking_convergence(
            data_dict,
            colors,
            fig,
            col=col,
            y_name=y_name,
            lower_name=lower_name,
            upper_name=upper_name,
        )
        annotations.append(
            dict(
                x=0,
                y=1.1,  # annotation point
                xref="x{}".format(col),
                yref="paper".format(col),
                text=data_title,
                showarrow=False,
                xanchor="left",
            ),
        )
        col += 1
    fig["layout"].update(annotations=annotations)
    # fig.update_layout(
    #     width=400 * ncols,
    #     height=400 * nrows,
    # )
    fig.show()
    return fig


def plot_ranking_convergence(
    data_dict,
    colors,
    fig=None,
    col=None,
    y_name="mean",
    lower_name="q05",
    upper_name="q95",
):
    if fig is None:
        # Plotting
        nrows = len(data_dict)  # TODO??
        ncols = 1
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_yaxes=True,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
        col = 1
    else:
        nrows = len(fig._grid_ref)
        ncols = len(fig._grid_ref[0])
    opacity = 0.5
    sa_names = list(list(data_dict.values())[0].keys())
    for rho_name, dict_ in data_dict.items():
        color = colors[rho_name]
        for row, sa_name in enumerate(sa_names):
            data = dict_[sa_name]
            x = data["iterations"]
            y = data[y_name]
            lower = data.get(lower_name, None)
            upper = data.get(upper_name, None)
            # color = np.random.randint(low=50, high=255, size=3)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    opacity=1,
                    showlegend=True,
                    legendgroup=rho_name,
                    name=rho_name,
                    marker=dict(
                        color="rgba({},{},{},{})".format(
                            color[0],
                            color[1],
                            color[2],
                            1,
                        ),
                    ),
                ),
                row=row + 1,
                col=col,
            )
            if lower is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=lower,
                        mode="lines",
                        opacity=opacity,
                        showlegend=False,
                        legendgroup=rho_name,
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
                    row=row + 1,
                    col=col,
                )
            if upper is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=upper,
                        showlegend=False,
                        legendgroup=rho_name,
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
                    row=row + 1,
                    col=col,
                )
            fig.update_yaxes(title_text=sa_name, row=row + 1, col=col)

    # fig.update_layout(
    #     width=400 * ncols,
    #     height=400 * nrows,
    # )
    fig.update_xaxes(title_text="iterations", row=row + 1, col=col)
    return fig


def plot_S(data_dict):
    nrows = len(list(data_dict.values())[0])
    ncols = len(data_dict)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=False,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=list(data_dict.keys()),
    )
    col = 1
    for model_name, S_dicts in data_dict.items():
        row = 1
        for sa_name, S_array in S_dicts.items():
            l = len(S_array)
            use = int(0.1 * l)
            x = np.arange(l)[:use]
            y = S_array[:use]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=4, color="#636EFA"),
                ),
                row=row,
                col=col,
            )
            if col == 1:
                fig.update_yaxes(title_text=sa_name.lower(), row=row, col=col)
            row += 1
        fig.update_xaxes(title_text="model inputs", row=row - 1, col=col)
        col += 1
    fig.update_layout(
        width=800 * ncols,
        height=200 * nrows,
    )
    fig.show()
    return fig
