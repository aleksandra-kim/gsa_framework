import numpy as np
import multiprocessing

OPTIMAL_CHUNK_SIZE_EFAST = 50
get_chunk_size_eFAST = lambda num_params: min(OPTIMAL_CHUNK_SIZE_EFAST, num_params)


def random_samples(iterations, num_params, seed=None):
    """Random standard uniform sampling for all iterations and parameters with an option of fixing random seed.

    Parameters
    ----------
    iterations : int
        Number of iterations.
    num_params : int
        Number of model inputs.
    seed : int
        Random seed.

    Returns
    -------
    samples : array
        Randomly generated samples of size ``iterations x num_params``.

    """
    np.random.seed(seed)
    samples = np.random.rand(iterations, num_params)
    return samples


def custom_unitcube_samples(X_unitcube):
    """Wrapper function to return custom sampling matrix if it is specified by the user, values are in [0,1] range."""
    return X_unitcube


def custom_rescaled_samples(X_rescaled):
    """Wrapper function to return custom sampling matrix if it is specified by the user, values are in rescaled range."""
    return X_rescaled


def latin_hypercube_samples(iterations, num_params, seed=None):
    """Latin hypercube samples in [0,1] range.

    Parameters
    ----------
    iterations : int
        Number of iterations.
    num_params : int
        Number of model inputs.
    seed : int
        Random seed.

    Returns
    -------
    samples : array
        Randomly generated latin hypercube samples of size ``iterations x num_params``.

    """
    np.random.seed(seed)
    step = 1 / iterations
    samples = np.random.uniform(low=0, high=step, size=(num_params, iterations))
    interval_start = np.linspace(start=0, stop=1, num=iterations, endpoint=False)
    for sample in samples:
        np.random.shuffle(interval_start)
        sample += interval_start
    return samples.T


def sobol_samples(iterations, num_params, skip_iterations=1000):
    """Quasi-random Sobol sequence in [0,1] range that skips first ``skip_iterations`` samples to avoid boundary values.

    Parameters
    ----------
    iterations : int
        Number of iterations.
    num_params : int
        Number of model inputs.
    skip_iterations : int
        Number of first Sobol sequence samples to skip.

    Returns
    -------
    samples : array
        Sobol samples of size ``iterations x num_params``.

    """
    from .sobol_sequence import SobolSample

    sobol = SobolSample(iterations + skip_iterations, num_params, scale=31)
    samples = sobol.generate_all_samples()
    return samples[skip_iterations:]


def saltelli_samples(iterations, num_params, skip_iterations=1000):
    """Saltelli samples in [0,1] range based on Sobol sequences and radial sampling.

    Parameters
    ----------
    iterations : int
        Number of iterations.
    num_params : int
        Number of model inputs.
    skip_iterations : int
        Number of first Sobol sequence samples to skip.

    Returns
    -------
    samples : array
        Saltelli samples of size ``iterations_per_parameter (num_params + 2) x num_params``,
        where ``iterations_per_parameter = iterations // (num_params + 2)``.

    References
    ----------
    Paper:
        Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index
        Saltelli A., Annoni P., Azzini I., Campolongo F., Ratto M., Tarantola S., 2010
        https://doi.org/10.1016/j.cpc.2009.09.018
    Link to the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/sample/saltelli.py

    """

    # Use Sobol samples as base
    from .sobol_sequence import SobolSample

    iterations_per_parameter = iterations // (num_params + 2)
    # generate base Sobol sequence samples
    sobol = SobolSample(
        iterations_per_parameter + skip_iterations, num_params * 2, scale=31
    )
    base_samples = sobol.generate_all_samples()
    base_samples = base_samples[skip_iterations:]
    # create saltelli samples with radial basis design
    samples = np.tile(base_samples[:, :num_params], (1, num_params + 2)).reshape(
        iterations_per_parameter * (num_params + 2), -1
    )
    samples[num_params + 1 :: num_params + 2, :] = base_samples[:, num_params:]
    # use a boolean mask for cross sampling of elements
    mask_ = np.full((num_params + 2) * num_params, False)
    mask_[num_params :: num_params + 1] = 1
    mask = np.tile(mask_, iterations_per_parameter).reshape(
        iterations_per_parameter * (num_params + 2), -1
    )
    samples[mask] = base_samples[:, num_params:].flatten()
    return samples


def eFAST_omega(iterations, num_params, M):
    """Compute omega parameter for the extended FAST sampling."""
    omega = np.zeros([num_params])
    omega[0] = np.floor((iterations - 1) / (2 * M))
    m = np.floor(omega[0] / (2 * M))
    if m >= (num_params - 1):
        omega[1:] = np.floor(np.linspace(1, m, num_params - 1))
    else:
        omega[1:] = np.arange(num_params - 1) % m + 1

    return omega


def eFAST_samples_one_chunk(i, iterations, num_params, M=4, seed=None):
    np.random.seed(seed)
    iterations_per_parameter = iterations // num_params
    # Determine current chunk
    chunk_size = get_chunk_size_eFAST(num_params)
    num_chunks = int(np.ceil(num_params / chunk_size))
    last_chunk = num_params % chunk_size

    if i < num_chunks - 1 or last_chunk == 0:
        num_params_curr = chunk_size
    elif i == num_chunks - 1:
        num_params_curr = last_chunk
    # Minimum number of iterations is chosen based on the Nyquist criterion, ``N`` in the paper
    N = max(4 * M ** 2 + 1, iterations_per_parameter)
    # Set of frequencies that would be assigned to each input factor
    omega = eFAST_omega(N, num_params, M)
    # Discretization of the frequency space
    s = (2 * np.pi / N) * np.arange(N)
    # Random phase-shift
    phi = 2 * np.pi * np.random.rand(num_params)

    mask_partial = np.ones([chunk_size, chunk_size], dtype=bool)
    np.fill_diagonal(mask_partial, False)

    chunk_before = i * chunk_size
    chunk_after = num_params - (i + 1) * chunk_size

    if i < num_chunks - 1 or last_chunk == 0:
        mask = np.hstack(
            [
                np.ones([chunk_size, chunk_before], dtype=bool),
                mask_partial,
                np.ones([chunk_size, chunk_after], dtype=bool),
            ]
        )
        omega_temp = np.zeros([chunk_size, num_params])
        omega_temp[mask] = np.tile(omega[1:], chunk_size)
        omega_temp[~mask] = omega[0]
    elif i == num_chunks - 1:
        mask = np.hstack(
            [
                np.ones([last_chunk, chunk_before], dtype=bool),
                mask_partial[:last_chunk, :last_chunk],
            ]
        )
        omega_temp = np.zeros([last_chunk, num_params])
        omega_temp[mask] = np.tile(omega[1:], last_chunk)
        omega_temp[~mask] = omega[0]

    start = i * chunk_size
    end = (i + 1) * chunk_size

    phi_chunk = phi[start:end]
    phi_chunk = np.tile(phi_chunk, [num_params]).reshape(num_params, num_params_curr).T
    phi_chunk = np.tile(phi_chunk, N).reshape(num_params_curr, num_params, N)
    omega2_kron = np.kron(omega_temp, s).reshape(num_params_curr, num_params, N)
    g = 0.5 + (1 / np.pi) * np.arcsin(np.sin(omega2_kron + phi_chunk))
    current_samples = np.transpose(g, (0, 2, 1)).reshape(
        N * num_params_curr, num_params
    )
    return current_samples


def eFAST_samples_many_chunks(
    icpu, iterations, num_params, num_params_per_cpu, M=4, seed=None
):
    chunk_size = get_chunk_size_eFAST(num_params_per_cpu)
    num_chunks = int(np.ceil(num_params_per_cpu / chunk_size))
    samples = np.zeros(shape=(0, num_params))
    for ichunk in range(num_chunks):
        i = icpu * num_chunks + ichunk
        samples = np.vstack(
            [samples, eFAST_samples_one_chunk(i, iterations, num_params, M, seed)]
        )
    return samples


def eFAST_samples(iterations, num_params, M=4, seed=None, cpus=None):
    """Extended FAST samples in [0,1] range.

    Parameters
    ----------
    iterations : int
        Number of iterations.
    num_params : int
        Number of model inputs.
    M : int
        Interference factor, usually 4 or higher.
    seed : int
        Random seed.
    cpus : int
        Number of cpus for parallel computation of eFAST samples with ``multiprocessing`` library.

    Returns
    -------
    samples : array
        eFASTT samples of size ``iterations x num_params``.

    References
    ----------
    Paper:
        A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output.
        Saltelli A., Tarantola S., Chan K. P.-S.
        https://doi.org/10.1080/00401706.1999.10485594
    Link to the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/sample/fast_sampler.py

    """

    chunk_size = get_chunk_size_eFAST(num_params)
    num_jobs = int(np.ceil(np.ceil(num_params / chunk_size) / cpus))
    params_range_per_cpu = np.hstack(
        [np.arange(0, num_params, num_jobs * chunk_size), num_params]
    )
    num_params_per_cpu = params_range_per_cpu[1:] - params_range_per_cpu[:-1]
    cpus_needed = len(num_params_per_cpu)

    with multiprocessing.Pool(processes=cpus_needed) as pool:
        samples_temp = pool.starmap(
            eFAST_samples_many_chunks,
            [
                (icpu, iterations, num_params, num_params_per_cpu[icpu], M, seed)
                for icpu in range(cpus_needed)
            ],
        )
    samples = np.zeros(shape=(0, num_params))
    for res in samples_temp:
        samples = np.vstack([samples, res])
    return samples
