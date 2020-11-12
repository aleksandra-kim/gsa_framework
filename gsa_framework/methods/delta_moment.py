# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import latin_hypercube_samples
from ..sensitivity_analysis.delta_moment import delta_moment


class DeltaMoment(SAM):
    sampling_label = "latinSampling"
    gsa_label = "deltaGsa"

    def __init__(self, num_resamples=10, **kwargs):
        super().__init__(**kwargs)
        self.num_resamples = num_resamples
        self.gsa_label = self.gsa_label + "Nr{}".format(self.num_resamples)

    def generate_unitcube_samples_based_on_method(self, iterations):
        X = latin_hypercube_samples(iterations, self.num_params, seed=self.seed)
        return X

    def generate_gsa_indices_based_on_method(self, **kwargs):
        flag_convergence = kwargs.get("flag_convergence", False)
        if not flag_convergence:
            S_dict = delta_moment(
                self.filepath_Y,
                self.filepath_X_rescaled,
                self.iterations,
                self.num_params,
                seed=self.seed,
            )
        else:
            iterations = kwargs.get("iterations", self.iterations)
            S_dict = delta_moment(
                self.filepath_Y,
                self.filepath_X_rescaled,
                iterations,
                self.num_params,
                seed=self.seed,
            )
        return S_dict
