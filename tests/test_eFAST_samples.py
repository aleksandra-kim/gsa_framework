from gsa_framework.sampling.get_samples import eFAST_samples
import numpy as np
import math


def sample_salib(problem, N, M=4, seed=None):
    """Generate model inputs for the Fourier Amplitude Sensitivity Test (FAST).

    Returns a NumPy matrix containing the model inputs required by the Fourier
    Amplitude sensitivity test.  The resulting matrix contains N * D rows and D
    columns, where D is the number of parameters.  The samples generated are
    intended to be used by :func:`SALib.analyze.fast.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    M : int
        The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition (default 4)
    seed : int
        Random seed.

    Notes
    -----
    This code has been copied from the SALib python library. The only thing that is changed
    is ``phi`` that is now created before the two ``for loops`` as an array. This is need to compare
    sampling results with our parallelized implementation.

    """

    if seed:
        np.random.seed(seed)

    if N <= 4 * M ** 2:
        raise ValueError(
            """
        Sample size N > 4M^2 is required. M=4 by default."""
        )

    D = problem["num_vars"]

    omega = np.zeros([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    if m >= (D - 1):
        omega[1:] = np.floor(np.linspace(1, m, D - 1))
    else:
        omega[1:] = np.arange(D - 1) % m + 1

    # Discretization of the frequency space, s
    s = (2 * math.pi / N) * np.arange(N)

    # Transformation to get points in the X space
    X = np.zeros([N * D, D])
    omega2 = np.zeros([D])
    phi = 2 * math.pi * np.random.rand(D)

    for i in range(D):
        omega2[i] = omega[0]
        idx = list(range(i)) + list(range(i + 1, D))
        omega2[idx] = omega[1:]
        l = range(i * N, (i + 1) * N)

        # random phase shift on [0, 2pi) following Saltelli et al.
        # Technometrics 1999

        for j in range(D):
            g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[j] * s + phi[i]))
            X[l, j] = g

    #     scale_samples(X, problem['bounds'])
    return X


def test_output():

    for i in range(10):
        print(i)
        seed = np.random.randint(1, 10000)
        cpus = 4
        M = np.random.randint(1, 20)
        low = 4 * M ** 2 + 1
        high = low + 100
        num_params = np.random.randint(1, 200)
        iterations_per_input = np.random.randint(low, high)
        iterations = iterations_per_input * num_params

        # ground truth from SALib
        problem = {
            "num_vars": num_params,
        }
        samples_salib = sample_salib(problem, iterations_per_input, M=M, seed=seed)

        # our implementation
        samples_gsa = eFAST_samples(num_params, iterations, M=M, seed=seed, cpus=cpus)

        assert np.allclose(samples_gsa, samples_salib)


#

# To test manually
# from tests.test_eFAST_samples import test_output
# if __name__ == "__main__":
#     test_output()
