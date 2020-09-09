from gsa_framework.sampling.get_samples import eFAST_samples
from SALib.sample import fast_sampler
import numpy as np

def test_output():

    for i in range(10):
        seed = np.random.randint(1,10000)
        M = np.random.randint(1, 200)
        low = 4*M**2+1
        high = low+100
        iterations = np.random.randint(low, high)
        num_params = np.random.randint(1, 200)

        # ground truth
        problem = {
            'num_vars': num_params,
            'names': np.arange(num_params),
            'bounds': [[0, 1] * num_params]
        }
        samples_salib = fast_sampler.sample(problem, iterations, M=M, seed=seed)

        # our implementation
        sampling_dict = {
            'iterations': iterations,
            'num_params': num_params,
            'seed': seed,
        }
        samples_gsa = eFAST_samples(sampling_dict, M=M)

        assert np.allclose(samples_gsa, samples_salib)

# SALib is funny in that they check whether the seed is given by users like this: `if seed`,
# which means that seed=0 would be regarded as no seed
