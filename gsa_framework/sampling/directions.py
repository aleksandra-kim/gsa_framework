# Constants taken from SALib: https://github.com/SALib/
# The MIT License (MIT)

# Copyright (c) 2013-2017 Jon Herman, Will Usher, and others.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# constants needed to compute Sobol sequences, see https://web.maths.unsw.edu.au/~fkuo/sobol/
# Papers:
# S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom sequence generator, 2003
# S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, 2008

import numpy as np
from pathlib import Path
from os.path import abspath, dirname


DATA_DIR = Path(abspath(dirname(__file__)))


class IndexedMemmappedArray:
    def __init__(self, filepath=None):
        filepath = str(filepath or DATA_DIR / "data" / "directions.npy")
        self.data = np.load(filepath, mmap_mode="r")

    def __getitem__(self, index):
        # TODO: Test algorithms for speed
        row = self.data[int(index), :]
        return row[row != -1]


# TODO: add `directions` to __init__
directions = IndexedMemmappedArray()
