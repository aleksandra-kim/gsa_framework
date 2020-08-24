from ..sampling.get_samples import saltelli_samples
from SALib.sample import saltelli
import numpy as np
import random as rd

def test_output():

    for i in range(10):
        iterations = rd.randint(1,1000)
        num_params = rd.randint(1,100)

        #ground truth
        problem = {
            'num_vars': num_params,
            'bounds': [[0, 1] * num_params]
        }
        samples_salib = saltelli.sample(problem, iterations, calc_second_order=False)

        #our implementation
        sampling_dict = {
            'iterations': iterations,
            'num_params': num_params,
        }
        samples_gsa = saltelli_samples(sampling_dict)

        assert np.allclose(samples_salib, samples_gsa)