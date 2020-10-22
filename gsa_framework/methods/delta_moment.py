# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import latin_hypercube_samples
from ..sensitivity_analysis.delta_moment import delta_moment


class eFAST(SAM):
    sampling_label = "sampling_latin"
    gsa_label = "gsa_delta"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_unitcube_samples_based_on_method(self):
        X = latin_hypercube_samples(self.iterations, self.num_params, seed=self.seed)
        return X

    def generate_gsa_indices_based_on_method(self, selected_iterations=None):
        S_dict = delta_moment()
        return S_dict
