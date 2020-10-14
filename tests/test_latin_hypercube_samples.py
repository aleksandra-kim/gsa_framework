from gsa_framework.sampling.get_samples import latin_hypercube_samples
import numpy as np


def test_output():

    for i in range(10):
        iterations = np.random.randint(1, 10000)
        num_params = np.random.randint(1, 1000)
        sampling_dict = {
            "iterations": iterations,
            "num_params": num_params,
        }
        samples = latin_hypercube_samples(sampling_dict)
        # Make sure that samples size is correct
        assert samples.shape == (iterations, num_params)
        # Make sure there's only one value in each interval
        interval_start = np.linspace(start=0, stop=1, num=iterations, endpoint=False)
        bins = np.hstack([interval_start, 1])
        for j in range(num_params):
            count, _ = np.histogram(
                samples[:, j],
                bins=bins,
                range=None,
                normed=None,
                weights=None,
                density=None,
            )
            assert np.all(count == 1)
