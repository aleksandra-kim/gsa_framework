import plotly.offline as py

color_blue_rgb = "rgb(29,105,150)"
color_blue_tuple = (29, 105, 150)
color_purple_tuple = (95, 70, 144)
color_purple_rgb = "rgb(95, 70, 144)"
color_green_rgb = "rgb(36, 121, 108)"
color_green_tuple = (36, 121, 108)
color_pink_rgb = "rgb(148, 52, 110)"
color_pink_tuple = (148, 52, 110)
color_yellow_rgb = "rgb(237, 173, 8)"
color_yellow_tuple = (237, 173, 8)
color_gray_hex = "#b2bcc0"
color_gray_tuple = (178, 188, 192)
color_lightgray_hex = "#eef1fc"
color_darkgray_hex = "#485063"
color_darkgray_rgb = "rgb(72, 80, 99)"
color_darkgray_tuple = (72, 80, 99)
color_black_hex = "#212931"
color_black_tuple = (33, 41, 49)
color_orange_rgb = "rgb(217,95,2)"
color_orange_tuple = (217, 95, 2)
color_blue_orange_av_rgb = "rgb(123,100,76)"
color_blue_orange_av_tuple = (123, 100, 76)
color_lightblue_hex = "#98a5c0"  # "#598baf"


def save_fig(
    fig,
    fig_filename,
    fig_format=[
        "jpeg",
    ],
    write_dir_fig=None,
):
    _ = py.iplot(fig, filename="latex")
    for f in fig_format:
        filepath_fig = write_dir_fig / "{}.{}".format(fig_filename, f)
        fig.write_image(filepath_fig.as_posix())


sa_plot = {
    "corr": {
        "name": r"$\text{Spearman correlation coefficients}$",
        # "name": r"$\text{Spearman}$",
        "notation": r"$\hat{\rho}$",
        "stat": r"$Stat(\hat{\rho})$",
        "stat_indices": r"$Stat_{indices}(\hat{\rho})$",
        "stat_ranking": r"$Stat_{ranking}(\hat{\rho})$",
        "underlined_name": r"$\underline{\text{Spearman correlation coefficients}}$",
    },
    "salt": {
        "name": r"$\text{Sobol total order indices}$",
        # "name": r"$\text{Sobol}$",
        "notation": r"$\hat{S}^T$",
        "stat": r"$Stat(\hat{S}^T)$",
        "stat_indices": r"$Stat_{indices}(\hat{S}^T)$",
        "stat_ranking": r"$Stat_{ranking}(\hat{S}^T)$",
        "underlined_name": r"$\underline{\text{Sobol total order indices}}$",
    },
    "delt": {
        "name": r"$\text{Delta indices}$",
        # "name": r"$\text{Delta}$",
        "notation": r"$\hat{\delta}$",
        "stat": r"$Stat(\hat{\delta})$",
        "stat_indices": r"$Stat_{indices}(\hat{\delta})$",
        "stat_ranking": r"$Stat_{ranking}(\hat{\delta})$",
        "underlined_name": r"$\underline{\text{Delta indices}}$",
    },
    "xgbo": {
        "name": r"$\text{XGBoost importances}$",
        # "name": r"$\text{XGBoost}$",
        "notation": r"$\hat{I}^2$",
        "stat": r"$Stat(\hat{I}^2)$",
        "stat_indices": r"$Stat_{indices}(\hat{I}^2)$",
        "stat_ranking": r"$Stat_{ranking}(\hat{I}^2)$",
        "underlined_name": r"$\underline{\text{XGBoost importances}}$",
    },
}
