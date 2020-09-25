# constants needed to compute Sobol sequences, see https://web.maths.unsw.edu.au/~fkuo/sobol/
# Papers:
# S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom sequence generator, 2003
# S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, 2008

import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.resolve() / "data"


class Directions:
    """Class that implements .

    Parameters
    ----------
    func_unit : dict
        Dictionary of the form {bw_demand_activity: amount}.
    method : tuple
        Tuple with an impact assessment method.
    write_dir : str
        Directory where intermediate results will be stored.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Returns LCIA scores when technosphere exchanges are sampled from their respective distributions.

    """

    def __init__(self, filepath=None):
        # TODO Chris: add `directions` to __init__
        filepath = str(filepath or DATA_DIR / "directions.npy")
        self.data = np.load(filepath, mmap_mode="r")

    def __getitem__(self, index):
        row = self.data[int(index), :]
        return row[row != -1]
