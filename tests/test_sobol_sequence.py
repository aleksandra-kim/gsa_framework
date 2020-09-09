from gsa_framework.sampling.get_samples import sobol_samples
from SALib.sample import sobol_sequence
import numpy as np

def test_output():

    for i in range(10):
        iterations = np.random.randint(1,1000)
        num_params = np.random.randint(1,100)

        # ground truth
        samples_salib = sobol_sequence.sample(iterations,num_params)

        # our implementation
        sampling_dict = {
            'iterations': iterations,
            'num_params': num_params,
            'skip_samples': 0,
        }
        samples_gsa = sobol_samples(sampling_dict)

        assert np.allclose(samples_gsa,samples_salib)
