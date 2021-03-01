from pathlib import Path
from gsa_framework.utils import *
from gsa_framework.plotting import plot_S

# 1. Choose which stability dictionaries to include
path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files")

# 1. Models
# models = ["1000_morris4", "5000_morris4", "10000_morris4"]
models = ["1000_morris"]
exclude_S = ["pearson", "delta_conf", "stat.r2", "stat.explained_variance"]

data_dict = {}
for model in models:
    if "morris" in model:
        write_dir = path_base / model
    elif "lca" in model:
        write_dir = path_base / "lca_model_food_10000"
    write_arr = write_dir / "arrays"
    files = [
        x
        for x in write_arr.iterdir()
        if x.is_file() and "S." in x.name and "stability" not in x.name
    ]
    files = sorted(files)
    S_dict = {}
    for file in files:
        S_dict.update(read_pickle(file))
    for exc in exclude_S:
        try:
            S_dict.pop(exc)
        except:
            pass
    data_dict[model] = S_dict

fig1 = plot_S(data_dict)

# # 1. All Morris results
# filename_fig = "gsa_all_morris.html"
# filepath_fig = path_base / "figures" / filename_fig
# fig1.write_html(filepath_fig.as_posix())
# filename_fig = "gsa_all_morris.pdf"
# filepath_fig = path_base / "figures" / filename_fig
# fig1.write_image(filepath_fig.as_posix())

# # 2. All Morris zoomed in
# filename_fig = "gsa_all_morris_zoomed_in.html"
# filepath_fig = path_base / "figures" / filename_fig
# fig1.write_html(filepath_fig.as_posix())
# filename_fig = "gsa_all_morris_zoomed_in.pdf"
# filepath_fig = path_base / "figures" / filename_fig
# fig1.write_image(filepath_fig.as_posix())

# 2. All LCA
filename_fig = "gsa_all_lca.html"
filepath_fig = path_base / "figures" / filename_fig
fig1.write_html(filepath_fig.as_posix())
filename_fig = "gsa_all_lca.pdf"
filepath_fig = path_base / "figures" / filename_fig
fig1.write_image(filepath_fig.as_posix())
