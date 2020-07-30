import numpy as np

def random_samples(sampler_dict):
    np.random.seed(sampler_dict.get('seed'))
    return np.random.rand(
        sampler_dict.get('iterations'),
        sampler_dict.get('num_params')
    )


def custom_samples(sampler_dict):
    return sampler_dict.get('X')


def sobol_samples(sampler_dict):
    from .sobol_sequence import SobolSample
    iterations = sampler_dict.get('iterations')
    num_params = sampler_dict.get('num_params')
    sobol = SobolSample(iterations, num_params, scale=31)
    return sobol.generate_all_samples()

# def saltelli_samples(sampler_dict):
#
#     return