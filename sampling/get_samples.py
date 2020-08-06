import numpy as np


def random_samples(dict_):
    np.random.seed(dict_.get('seed'))
    return np.random.rand(
        dict_.get('iterations'),
        dict_.get('num_params')
    )


def custom_samples(dict_):
    return dict_.get('X')


def sobol_samples(dict_):
    from .sobol_sequence import SobolSample
    iterations = dict_.get('iterations')
    num_params = dict_.get('num_params')
    skip_samples = dict_.get('skip_samples', 1000)
    sobol = SobolSample(iterations+skip_samples, num_params, scale=31)
    samples = sobol.generate_all_samples()
    return samples[skip_samples:]


def saltelli_samples(dict_):
    '''
    Comment:
        significant speed up over salib, eg for 2000 iterations and 1000 parameters,
        salib generated samples in 10 min, our implementation - in 50 seconds
    '''
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


def dissimilarity_samples(dict_):
    '''only kth parameter is varied'''
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