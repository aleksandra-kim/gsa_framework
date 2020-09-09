# Code on Sobol sequences generation adapted from SALib: https://github.com/SALib/
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

# Module that generates quasi-random Sobol sequences. They are needed for a more uniform space coverage.
# see https://web.maths.unsw.edu.au/~fkuo/sobol/
# Papers:
# S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom sequence generator, 2003
# S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional projections, 2008

from .directions import Directions
import math
import numpy as np


class SobolSample:
    """Generate Sobol quasi-random sequences.

    The samples can be either of size ``num_params`` and generated one at a time using method ``next()`` or
    all samples computed simultaneously as a matrix of size ``[iterations, num_params]``.

    Parameters
    ----------
    iterations : int
        Number of samples to generate.
    num_params : int
        Number of parameters (dimensions) in the Sobol sequence.
    scale : int

    References
    ----------
    Papers
        Remark on Algorithm 659: Implementing Sobol's quasi-random sequence generator.
        Stephen Joe, Frances Y. Kuo
        https://doi.org/10.1145/641876.641879

        Constructing Sobol sequences with better two-dimensional projections.
        Stephen Joe, Frances Y. Kuo
        https://doi.org/10.1137/070709359

    """

    def __init__(self, iterations, num_params, scale=31):
        self.directions = Directions()
        if num_params > len(self.directions.data) + 1:
            raise ValueError("Error in Sobol sequence: not enough dimensions")
        L = int(math.ceil(math.log(iterations) / math.log(2)))
        if L > scale:
            raise ValueError("Error in Sobol sequence: not enough bits")
        self.n_samples, self.n_dimensions, self.scale, self.L = (
            iterations,
            num_params,
            scale,
            L,
        )
        self.current = 0
        self.Y = np.array([int(0)] * self.n_dimensions)
        self.V = self.generate_V()

    def index_of_least_significant_zero_bit(self, value):
        """Generate index of the least significant zero bit of a value."""

        index = 1
        while (value & 1) != 0:
            value >>= 1
            index += 1
        return index

    def generate_V(self):
        """Compute matrix V that is needed for Sobol quasi-random sequences generation."""

        n_samples, n_dimensions, L = self.n_samples, self.n_dimensions, self.L
        V = np.zeros([L + 1, n_dimensions], dtype=int)
        V[1:, 0] = [1 << (self.scale - j) for j in range(1, L + 1)]
        for i in range(n_dimensions - 1):
            # Sasha - is this conversion necessary?
            m = np.array(self.directions[i], dtype=int)
            s = len(m) - 1
            # The following code discards the first row of the ``m`` array
            # Because it has floating point errors, e.g. values of 2.24e-314
            if L <= s:
                V[1:, i + 1] = [m[j] << (self.scale - j) for j in range(1, L + 1)]
            else:
                V[1 : s + 1, i + 1] = [
                    m[j] << (self.scale - j) for j in range(1, s + 1)
                ]
                for j in range(s + 1, L + 1):
                    V[j, i + 1] = V[j - s, i + 1] ^ (V[j - s, i + 1] >> s)
                    for k in range(1, s):
                        V[j, i + 1] ^= ((m[0] >> (s - 1 - k)) & 1) * V[j - k][i + 1]
        return V

    def generate_sample(self):
        """Generate an array of size ``num_params`` that contains one sample of Sobol sequence."""

        sample_one = np.zeros(self.n_dimensions)

        for i in range(self.n_dimensions):
            self.Y[i] ^= self.V[
                self.index_of_least_significant_zero_bit(self.current - 1), i
            ]
            sample_one[i] = float(self.Y[i] / math.pow(2, self.scale))
        self.current += 1
        return sample_one

    def __iter__(self):
        """S: Is that needed here?"""

        # Yes, see https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator

        return self

    def __next__(self):
        if self.current > self.n_samples - 1:
            raise StopIteration
        elif self.current == 0:
            self.current = 1
            return self.Y
        else:
            return self.generate_sample()

    def generate_all_samples(self):
        """Generate a matrix that contains samples for all ``iterations`` and ``num_params``."""

        n_samples, n_dimensions, V = self.n_samples, self.n_dimensions, self.V
        sample_all = np.zeros([n_samples, n_dimensions])
        X = int(0)
        for j in range(1, n_samples):
            X ^= V[self.index_of_least_significant_zero_bit(j - 1)]
            sample_all[j][:] = [float(x / math.pow(2, self.scale)) for x in X]
        return sample_all
