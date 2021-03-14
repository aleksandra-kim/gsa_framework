import plotly.offline as py

color_blue_rgb = "rgb(29,105,150)"
color_blue_tuple = (29, 105, 150)
color_purple_tuple = (95, 70, 144)
color_purple_rgb = "rgb(95, 70, 144)"
color_green_rgb = "rgb(36, 121, 108)"
color_pink_rgb = "rgb(148, 52, 110)"
color_yellow_rgb = "rgb(237, 173, 8)"
color_gray_hex = "#b2bcc0"
color_black_hex = "#212931"
color_orange_rgb = "rgb(217,95,2)"
color_orange_tuple = (217, 95, 2)
color_blue_orange_av_rgb = "rgb(123,100,76)"


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
        "notation": r"$\hat{\rho}$",
        "stat_indices": r"$Stat_{indices}(\hat{\rho})$",
        "stat_ranking": r"$Stat_{ranking}(\hat{\rho})$",
    },
    "salt": {
        "name": r"$\text{Sobol total order indices}$",
        "notation": r"$\hat{S}^T$",
        "stat_indices": r"$Stat_{indices}(\hat{S}^T)$",
        "stat_ranking": r"$Stat_{ranking}(\hat{S}^T)$",
    },
    "delt": {
        "name": r"$\text{Delta indices}$",
        "notation": r"$\hat{\delta}$",
        "stat_indices": r"$Stat_{indices}(\hat{\delta})$",
        "stat_ranking": r"$Stat_{ranking}(\hat{\delta})$",
    },
    "xgbo": {
        "name": r"$\text{XGBoost importances}$",
        "notation": r"$\hat{I}^2$",
        "stat_indices": r"$Stat_{indices}(\hat{I}^2)$",
        "stat_ranking": r"$Stat_{ranking}(\hat{I}^2)$",
    },
}
