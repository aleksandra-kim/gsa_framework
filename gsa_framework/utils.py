import numpy as np
import h5py, pickle
from scipy.stats import norm

get_z_alpha_2 = lambda confidence_level: norm.ppf(0.5 + confidence_level / 2)


def write_hdf5_array(array, filepath):
    """Write ``array`` to a file with .hdf5 extension"""
    try:
        n_rows, n_cols = array.shape[0], array.shape[1]
    except IndexError:
        n_rows, n_cols = 1, array.shape[0]

    with h5py.File(filepath, "w") as f:
        d = f.create_dataset(
            "dataset", (n_rows, n_cols), maxshape=(n_rows, n_cols), dtype=array.dtype
        )
        d[:] = array


def read_hdf5_array(filepath):
    """Read ``array`` from a file with .hdf5 extension"""
    with h5py.File(filepath, "r") as f:
        array = np.array(f["dataset"][:])
    return array


def write_pickle(data, filepath):
    """Write ``data`` to a file with .pickle extension"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def read_pickle(filepath):
    """Read ``data`` from a file with .pickle extension"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def uniform_rescale(X, inputs):
    """Rescale samples from standard [0,1] uniform distribution to samples from any uniform.

    Parameters
    ----------
    X : np.array
        Array of size [iterations, num_params] with samples from standard uniform distribution.
    inputs : dict
        Parameter dictionary, where keys are parameter names and values are their uniform distribution ranges.

    Returns
    -------
    X_rescaled : np.array
        Array of size [iterations, num_params] with rescaled samples.

    """
    left_rescale = np.array(list(inputs.values()))[:, 0]
    right_rescale = np.array(list(inputs.values()))[:, 1]
    X_rescaled = (right_rescale - left_rescale) * X + left_rescale
    return X_rescaled


def all_exc_same(tech_params):
    flag=True # means that all exchanges are exactly the same, including scale, loc, amount
    for p in tech_params[1:]:
        if p[["amount","loc", "scale"]] != tech_params[0][["amount","loc", "scale"]]:
            flag=False
            break
    return flag
