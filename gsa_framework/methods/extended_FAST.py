import numpy as np

# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import eFAST_samples
from ..sensitivity_analysis.extended_FAST import eFAST_indices


class eFAST(SAM):
    label = "extended_FAST"

    def __init__(self, M=4, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.iterations = self.calculate_efast_iterations()

    def calculate_efast_iterations(self):
        return max(
            (4 * self.M ** 2 + 1) * self.num_params, self.iterations
        )  # Sample size N > 4M^2 is required. M=4 by default.

    def generate_unitcube_samples_based_on_method(self):
        X = eFAST_samples(
            self.num_params, self.iterations, M=self.M, seed=self.seed, cpus=self.cpus
        )
        return X

    def generate_gsa_indices_based_on_method(self):
        S_dict = eFAST_indices(
            self.filepath_Y,
            self.iterations,
            self.num_params,
            M=4,
            selected_iterations=None,
        )
        return S_dict
