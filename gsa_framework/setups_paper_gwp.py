from gsa_framework.lca import LCAModel
from gsa_framework.test_functions import Morris4
from gsa_framework.methods.correlations import CorrelationCoefficients
from gsa_framework.methods.saltelli_sobol import SaltelliSobol
from gsa_framework.methods.delta_moment import DeltaMoment
from gsa_framework.methods.gradient_boosting import GradientBoosting
from pathlib import Path
import brightway2 as bw
import time
import numpy as np
from gsa_framework.utils import (
    read_hdf5_array,
    read_pickle,
    write_hdf5_array,
    write_pickle,
)
import h5py
from gsa_framework.convergence import Convergence
import dask


def setup_lca_model_oases(
    path_base, num_params=None, write_dir_name=None, flag_generate_scores_dict=False
):
    # LCA model
    bw.projects.set_current("GSA for oases")
    co = bw.Database("CH consumption 1.0")
    demand_act = [act for act in co if "ch hh average consumption" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a")
    # num_params
    if num_params is None:
        lca = bw.LCA(demand, method)
        lca.lci()
        lca.lcia()
        print("LCA score is {}".format(lca.score))
        all_uncertain_params = lca.tech_params[lca.tech_params["uncertainty_type"] > 1]
        num_params = len(all_uncertain_params)
        print("Total number of uncertain exchanges is {}".format(num_params))
    # Define some variables
    if write_dir_name is None:
        write_dir_name = "oases_gsa_gwp_{}".format(num_params)
    write_dir = path_base / write_dir_name
    if flag_generate_scores_dict:
        model = LCAModel(demand, method, write_dir)  # generate scores_dict
        del model
    model = LCAModel(demand, method, write_dir, num_params=num_params)
    gsa_seed = 92374523
    return model, write_dir, gsa_seed


def setup_lca_model_paper(
    path_base, num_params=None, write_dir=None, flag_generate_scores_dict=False
):
    # LCA model
    bw.projects.set_current("GSA for paper")
    co = bw.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a")
    # num_params
    if num_params is None:
        lca = bw.LCA(demand, method)
        lca.lci()
        lca.lcia()
        print("LCA score is {}".format(lca.score))
        all_uncertain_params = lca.tech_params[lca.tech_params["uncertainty_type"] > 1]
        num_params = len(all_uncertain_params)
        print("Total number of uncertain exchanges is {}".format(num_params))
    # Define some variables
    if write_dir is None:
        # path_base = Path('/data/user/kim_a/paper_gsa')
        write_dir = path_base / "lca_model_food_{}".format(num_params)
    if flag_generate_scores_dict:
        model = LCAModel(demand, method, write_dir)  # generate scores_dict
        del model
    model = LCAModel(demand, method, write_dir, num_params=num_params)
    gsa_seed = 92374523
    return model, write_dir, gsa_seed


def setup_morris4_model(path_base, num_params):
    num_influential = num_params // 100
    write_dir = path_base / "{}_morris4".format(num_params)
    model = Morris4(num_params=num_params, num_influential=num_influential)
    gsa_seed = 3407
    return model, write_dir, gsa_seed


def setup_corr(num_params, iterations, setup_model, path_base):
    model, write_dir, gsa_seed = setup_model(path_base, num_params)
    # Setup GSA
    gsa = CorrelationCoefficients(
        iterations=iterations,
        model=model,
        write_dir=write_dir,
        seed=gsa_seed,
    )
    return gsa


def setup_salt(num_params, iterations, setup_model, path_base):
    model, write_dir, gsa_seed = setup_model(path_base, num_params)
    gsa = SaltelliSobol(iterations=iterations, model=model, write_dir=write_dir)
    return gsa


def setup_delt(num_params, iterations, setup_model, path_base):
    model, write_dir, gsa_seed = setup_model(path_base, num_params)
    num_resamples = 0
    gsa = DeltaMoment(
        iterations=iterations,
        model=model,
        write_dir=write_dir,
        num_resamples=num_resamples,
        seed=gsa_seed,
    )
    return gsa


def setup_xgbo_lca(num_params, iterations, setup_lca_model, path_base):
    model, write_dir, gsa_seed = setup_lca_model(path_base, num_params)
    xgb_model = None
    test_size = 0.2
    tuning_parameters = dict(
        learning_rate=0.15,
        gamma=0,
        min_child_weight=300,
        max_depth=4,
        reg_lambda=0,
        reg_alpha=0,
        n_estimators=600,
        subsample=0.3,
        colsample_bytree=0.2,
    )
    gsa = GradientBoosting(
        iterations=iterations,
        model=model,
        write_dir=write_dir,
        seed=gsa_seed,
        tuning_parameters=tuning_parameters,
        test_size=test_size,
        xgb_model=xgb_model,
    )
    return gsa


def setup_xgbo_morris4(num_params, iterations, setup_morris4_model, path_base):
    model, write_dir, gsa_seed = setup_morris4_model(path_base, num_params)
    xgb_model = None
    test_size = 0.2
    if num_params == 1000:
        tuning_parameters = dict(
            learning_rate=0.1,
            gamma=0,
            min_child_weight=30,
            max_depth=2,
            reg_lambda=10,
            reg_alpha=0,
            n_estimators=500,
            subsample=0.6,
            colsample_bytree=0.3,
        )
    elif num_params == 5000:
        tuning_parameters = dict(
            learning_rate=0.2,
            gamma=0,
            min_child_weight=300,
            max_depth=2,
            reg_lambda=0,
            reg_alpha=0,
            n_estimators=800,
            subsample=0.3,
            colsample_bytree=0.3,
        )
    elif num_params == 10000:
        tuning_parameters = dict(
            learning_rate=0.2,
            gamma=0,
            min_child_weight=600,
            max_depth=2,
            reg_lambda=0,
            reg_alpha=0,
            n_estimators=1500,
            subsample=0.2,
            colsample_bytree=0.2,
        )
    gsa = GradientBoosting(
        iterations=iterations,
        model=model,
        write_dir=write_dir,
        seed=gsa_seed,
        tuning_parameters=tuning_parameters,
        test_size=test_size,
        xgb_model=xgb_model,
    )
    return gsa


def write_X_chunks(gsa, n_workers):
    X = gsa.generate_unitcube_samples_based_on_method(gsa.iterations)
    gsa.create_model_output_dir()
    print(X.shape, gsa.dirpath_Y)
    iter_chunk = gsa.iterations // n_workers
    for i in range(n_workers):
        start = iter_chunk * i
        end = iter_chunk * (i + 1)
        print(i, start, end)
        X_chunk = X[start:end, :]
        filepath_X_chunk = gsa.dirpath_Y / "X.unitcube.{}.{}.pickle".format(
            i, n_workers
        )
        write_pickle(X_chunk, filepath_X_chunk)


def compute_scores_per_worker(
    option, num_params, iterations, i_worker, n_workers, setup_lca_model, path_base
):
    if option == "corr":
        gsa = setup_corr(num_params, iterations, setup_lca_model, path_base)
    elif option == "salt":
        gsa = setup_salt(num_params, iterations, setup_lca_model, path_base)
    elif option == "delt":
        gsa = setup_delt(num_params, iterations, setup_lca_model, path_base)
    elif option == "xgbo":
        gsa = setup_xgbo(num_params, iterations, setup_lca_model, path_base)
    gsa.dirpath_Y.mkdir(parents=True, exist_ok=True)
    filepath_X_chunk = gsa.dirpath_Y / "X.unitcube.{}.{}.pickle".format(
        i_worker, n_workers
    )
    X_chunk_unitcube = read_pickle(filepath_X_chunk)
    X_chunk_rescaled = gsa.model.rescale(X_chunk_unitcube)
    del X_chunk_unitcube
    scores = gsa.model(X_chunk_rescaled)
    Y_filename = "{}.{}.pickle".format(i_worker, n_workers)
    filepath = gsa.dirpath_Y / Y_filename
    write_pickle(scores, filepath)
    return scores


def generate_model_output_from_chunks(gsa, n_workers):
    Y = np.zeros(shape=(0,))
    for i in range(n_workers):
        filepath_Y_chunk = gsa.dirpath_Y / "{}.{}.pickle".format(i, n_workers)
        Y_chunk = read_pickle(filepath_Y_chunk)
        Y = np.hstack([Y, Y_chunk])  # TODO change to vstack for multidimensional output
    write_hdf5_array(Y, gsa.filepath_Y)
    return Y
