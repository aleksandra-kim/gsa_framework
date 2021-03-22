from dev.setups_paper_gwp import *
from pathlib import Path

if __name__ == "__main__":

    path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/")

    # 1. Models
    num_params = 10000

    # iter_corr = 4 * num_params
    # gsa_corr = setup_corr(num_params, iter_corr, setup_morris4_model, path_base)
    # gsa_corr.perform_gsa()

    iter_salt = 100 * num_params
    gsa_salt = setup_salt(num_params, iter_salt, setup_morris4_model, path_base)
    t1 = time.time()
    gsa_salt.generate_gsa_indices()
    t2 = time.time()
    print(t2 - t1)
    # iter_delt = 8 * num_params
    # gsa_delt = setup_delt(num_params, iter_delt, setup_morris4_model, path_base)
    # gsa_delt.perform_gsa()

    # iter_xgbo = 4 * num_params
    # gsa_xgbo = setup_xgbo_morris4(num_params, iter_xgbo, setup_morris4_model, path_base)
    # gsa_xgbo.perform_gsa()
