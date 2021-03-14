from gsa_framework.setups_paper_gwp import setup_xgbo_morris4, setup_morris4_model
from pathlib import Path

if __name__ == "__main__":

    path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/")

    # 1. Models
    num_params = 10000
    iter_xgbo = 4 * num_params
    gsa_xgbo = setup_xgbo_morris4(num_params, iter_xgbo, setup_morris4_model, path_base)
    gsa_xgbo.perform_gsa()
