# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import saltelli_samples
from ..sensitivity_analysis.saltelli_sobol import sobol_indices


class SaltelliSobol(SAM):
    sampling_label = "saltelliSampling"
    gsa_label = "saltelliGsa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterations = self.calculate_iterations()

    def calculate_iterations(self):
        if self.iterations:
            iterations_per_param = max(self.iterations // (self.num_params + 2), 1)
        else:
            iterations_per_param = self.num_params
        return iterations_per_param * (self.num_params + 2)

    def generate_unitcube_samples_based_on_method(self, iterations):
        return saltelli_samples(iterations, self.num_params)

    def generate_gsa_indices_based_on_method(self, **kwargs):
        selected_iterations = kwargs.get("selected_iterations")
        S_dict = sobol_indices(self.filepath_Y, self.num_params, selected_iterations)
        return S_dict
