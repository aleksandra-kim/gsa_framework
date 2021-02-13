from pathlib import Path
import brightway2 as bw
from gsa_framework.plotting import (
    plot_histogram_Y,
    plot_histogram_Y1_Y2,
    plot_correlation_Y1_Y2,
)
from gsa_framework.utils import read_hdf5_array
import numpy as np


if __name__ == "__main__":

    path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/")
    write_dir = path_base / "lca_model_food_10000"
    write_dir_arrays = write_dir / "arrays"
    write_dir_figs = write_dir / "figures"

    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a")

    lca = bw.LCA(demand, method)
    lca.lci()
    lca.lcia()
    print("LCA score is {}".format(lca.score))

    filepath_Yall = write_dir_arrays / "validation.Y.all.2000.66666.hdf5"
    filepath_Y10k = write_dir_arrays / "validation.Y.10000inf.2000.66666.LocalSA.hdf5"
    filepath_Y60 = write_dir_arrays / "validation.Y.60inf.2000.23467.SpearmanIndex.hdf5"
    Yall = read_hdf5_array(filepath_Yall).flatten()
    Y10k = read_hdf5_array(filepath_Y10k).flatten()
    Y60 = read_hdf5_array(filepath_Y60).flatten()
    bin_min = min(np.hstack([Yall, Y10k, Y60]))
    bin_max = max(np.hstack([Yall, Y10k, Y60, 300]))
    lca_score_text = "LCA score, [kg CO2-eq.]"

    # # Histogram of Y base
    # fig = histogram_Y(
    #     Yall,
    #     bin_min=bin_min,
    #     bin_max=bin_max,
    #     trace_name="All parameters vary  ",
    #     xaxes_title_text=lca_score_text,
    # )
    # filepath_fig = write_dir_figs / "base_Y.pdf"
    # fig.write_image(filepath_fig.as_posix())
    #
    # # 10k vary
    # fig = histogram_Y1_Y2(
    #     Yall,
    #     Y10k,
    #     bin_min=bin_min,
    #     bin_max=bin_max,
    #     trace_name1="All parameters vary  ",
    #     trace_name2="10k parameters vary",
    #     xaxes_title_text=lca_score_text,
    # )
    # filepath_fig = write_dir_figs / "base_influential_Y_histogram_10k.pdf"
    # fig.write_image(filepath_fig.as_posix())
    #
    # # 60 vary
    # fig = histogram_Y1_Y2(
    #     Yall,
    #     Y60,
    #     bin_min=bin_min,
    #     bin_max=bin_max,
    #     trace_name1="All parameters vary  ",
    #     trace_name2="60 parameters vary",
    #     xaxes_title_text=lca_score_text,
    # )
    # filepath_fig = write_dir_figs / "base_influential_Y_histogram_60.pdf"
    # fig.write_image(filepath_fig.as_posix())

    # Validation correlation plots
    filepath_Yall_23567 = write_dir_arrays / "validation.Y.all.2000.23467.hdf5"
    Yall = read_hdf5_array(filepath_Yall_23567).flatten()
    fig = plot_correlation_Y1_Y2(
        Yall,
        Yall,
        color2="#636EFA",
        start=0,
        end=100,
        trace_name1="All parameters vary  ",
        trace_name2="60 parameters vary",
        xaxes1_title_text=None,
        yaxes1_title_text=lca_score_text,
        xaxes2_title_text=lca_score_text,
        yaxes2_title_text=lca_score_text,
    )
    filepath_fig = write_dir_figs / "base_influential_Y_correlation_60_only_blue.pdf"
    fig.write_image(filepath_fig.as_posix())
