from pygsa.sampling.sobol_sequence import SobolSample
from SALib.sample import sobol_sequence
import numpy as np
import random as rd

def test_output():

    for i in range(10):
        n_runs = rd.randint(1,1000)
        n_dimensions = rd.randint(1,100)

        #ground truth
        samples_salib = sobol_sequence.sample(n_runs,n_dimensions)

        #our implementation
        Sampler = SobolSample(n_runs,n_dimensions)
        samples = Sampler.generate_all_samples()

        assert np.allclose(samples,samples_salib)
