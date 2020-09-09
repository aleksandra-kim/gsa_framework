# constants needed to compute Sobol sequences, see https://web.maths.unsw.edu.au/~fkuo/sobol/
# Papers:
# S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom sequence generator, 2003
# S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, 2008

import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.resolve() / "data"


class Directions:
    def __init__(self, filepath=None):
        # TODO: add `directions` to __init__
        filepath = str(filepath or DATA_DIR / "directions.npy")
        self.data = np.load(filepath, mmap_mode="r")

    def __getitem__(self, index):
        row = self.data[int(index), :]
        return row[row != -1]
