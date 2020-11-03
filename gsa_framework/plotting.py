import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram_Y(
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
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(x=0.43, y=0.96),
    )
    fig.update_yaxes(title_text="Frequency", range=[-10, 420])
    fig.update_xaxes(title_text=xaxes_title_text)
    fig.show()
    return fig


def histogram_Y1_Y2(
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
            showlegend=True,
        ),
    )
    fig.add_trace(
        go.Bar(
            x=bins2,
            y=freq2,
            name=trace_name2,
            opacity=opacity,
            marker=dict(color=color2),
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
        width=400,
        height=220,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(x=0.43, y=0.96),
    )
    fig.update_yaxes(title_text="Frequency", range=[0, 420])
    fig.update_xaxes(title_text=xaxes_title_text)
    fig.show()
    return fig


def correlation_Y1_Y2(
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
        legend=dict(x=0.03, y=1.0),  # on top
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
    fig.show()
    return fig
