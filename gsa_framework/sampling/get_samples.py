import numpy as np

def random_samples(dict_):
    """Random standard uniform sampling for all iterations and parameters with or without fixing random seed."""
    np.random.seed(dict_.get('seed'))
    return np.random.rand(
        dict_.get('iterations'),
        dict_.get('num_params')
    )


def custom_samples(dict_):
    """Wrapper function to return custom sampling matrix if it is specified by the user, values are in [0,1] range."""
    return dict_.get('X')


def sobol_samples(dict_):
    """Quasi-random Sobol sequence in [0,1] range that skips first ``skip_samples`` samples to avoid boundary values."""
    from .sobol_sequence import SobolSample
    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    skip_samples = dict_.get('skip_samples', 1000)
    sobol = SobolSample(iterations+skip_samples, num_params, scale=31)
    samples = sobol.generate_all_samples()
    return samples[skip_samples:]


def saltelli_samples(dict_):
    """Saltelli samples in [0,1] range based on Sobol sequences and radial sampling.

    Notes
    -----
        Speed up over SALib: for 2000 iterations and 1000 parameters, SALib needs 10 min, our implementation - 50 s.

    References
    ----------
    Paper:
        Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index
        Saltelli A., Annoni P., Azzini I., Campolongo F., Ratto M., Tarantola S., 2010
        https://doi.org/10.1016/j.cpc.2009.09.018
    Link with the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/sample/saltelli.py

    """

    # Use Sobol samples as base
    from .sobol_sequence import SobolSample
    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    skip_samples = dict_.get('skip_samples', 1000)
    iterations_per_parameter = iterations // (num_params+2)
    # generate base Sobol sequence samples
    sobol = SobolSample(iterations_per_parameter + skip_samples, num_params * 2, scale=31)
    base_samples = sobol.generate_all_samples()
    base_samples = base_samples[skip_samples:]
    # create saltelli samples with radial basis design
    samples = np.tile(base_samples[:, :num_params], (1, num_params+2)).reshape(iterations_per_parameter*(num_params+2), -1)
    samples[num_params+1::num_params+2, :] = base_samples[:, num_params:]
    # use a boolean mask for cross sampling of elements
    mask_ = np.full((num_params+2) * num_params, False)
    mask_[num_params::num_params+1] = 1
    mask = np.tile(mask_, iterations_per_parameter).reshape(iterations_per_parameter*(num_params+2), -1)
    samples[mask] = base_samples[:, num_params:].flatten()
    return samples


# Would be nice to have consistent function name parameters

def get_omega_eFAST(num_params, iterations, M):
    """Compute omega parameter for the extended FAST sampling."""
    omega = np.zeros([num_params])
    omega[0] = np.floor((iterations - 1) / (2 * M))
    m = np.floor(omega[0] / (2 * M))
    if m >= (num_params - 1):
        omega[1:] = np.floor(np.linspace(1, m, num_params - 1))
    else:
        omega[1:] = np.arange(num_params - 1) % m + 1

    return omega


def eFAST_samples(dict_, M=4):
    """Extended FAST samples in [0,1] range.

    Notes
    -----
        Code optimized from the SALib implementation, with a ~6 times speed up.

    References
    ----------
    Paper:
        A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output.
        Saltelli A., Tarantola S., Chan K. P.-S.
        https://doi.org/10.1080/00401706.1999.10485594
    Link with the original implementation:
        https://github.com/SALib/SALib/blob/master/src/SALib/sample/fast_sampler.py

    TODO what to do with M? should be consistent in both sampling and FAST indices computation

    """

    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    omega = get_omega_eFAST(num_params, iterations, M)
    if iterations <= 4 * M ** 2:
        print(iterations, M)
        raise ValueError("""Sample size N > 4M^2 is required. M=4 by default.""")

    # Discretization of the frequency space, s
    s = (2 * np.pi /  iterations) * np.arange( iterations)

    # Transformation to get points in the X space
    idx = np.ones([num_params, num_params], dtype=bool)
    np.fill_diagonal(idx, False)
    omega2 = np.zeros([num_params, num_params])
    omega2[idx] = np.tile(omega[1:], num_params)
    np.fill_diagonal(omega2, omega[0])

    np.random.seed(dict_.get('seed'))
    phi = 2 * np.pi * np.random.rand(num_params)
    phi = np.tile(phi, [num_params]).reshape(num_params, num_params).T
    phi = np.tile(phi,  iterations).reshape(num_params, num_params,  iterations)

    omega2_kron = np.kron(omega2, s).reshape(num_params, num_params,  iterations)
    g = 0.5 + (1 / np.pi) * np.arcsin(np.sin(omega2_kron + phi))

    samples = np.transpose(g, (0, 2, 1)).reshape( iterations * num_params, num_params)
    return samples


def dissimilarity_samples(dict_):
    """Sampling for dissimilarity measures.

    First base sampling is created where all parameters vary, then for each parameter base sampling is copied
    but for the current parameter base values are replaced with new values.

    """

    base_sampler_fnc = dict_.get('base_sampler_fnc')
    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    iterations_per_parameter = iterations // (num_params+1)
    dict_base = {
        'iterations': iterations_per_parameter,
        'num_params': num_params,
    }
    base_samples = base_sampler_fnc(dict_base)
    samples = np.tile(base_samples, (num_params+1, 1))
    for i in range(num_params):
        samples[(i+1)*iterations_per_parameter : (i+2)*iterations_per_parameter, i] = \
            np.mean(base_samples[:,i])
    return samples

# def dissimilarity_samples(dict_):
#     '''kth parameter is fixed'''
#     base_sampler_fnc = dict_.get('base_sampler_fnc')
#     iterations = dict_.get('iterations')
#     num_params = dict_.get('num_params')
#     iterations_per_parameter = iterations // (num_params+1)
#     dict_adjusted = {
#         'iterations': iterations_per_parameter*(num_params+1),
#         'num_params': num_params,
#     }
#     samples = base_sampler_fnc(dict_adjusted)
#     base_samples = samples[:iterations_per_parameter,:]
#     print(base_samples.shape)
#     for i in range(num_params):
#         samples[(i+1)*iterations_per_parameter : (i+2)*iterations_per_parameter, i] = base_samples[:,i]
#
#     return samples
