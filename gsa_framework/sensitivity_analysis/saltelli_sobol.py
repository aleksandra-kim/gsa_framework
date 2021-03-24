# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import saltelli_samples
from ..sensitivity_methods.saltelli_sobol import sobol_indices


class SaltelliSobol(SAM):
    """Global sensitivity analysis with Sobol indices estimated by Saltelli and Jansen estimators.

    References
    ----------
    Paper:
        :cite:ts:`saltelli2010variance`

    """

    sampling_label = "saltelliSampling"
    gsa_label = "saltelliGsa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterations = self.calculate_iterations()

    def calculate_iterations(self):
        """Compute number of iterations closest to ``iterations`` (if given) and consistent with Saltelli sampling."""
        if self.iterations:
            iterations_per_param = max(self.iterations // (self.num_params + 2), 1)
        else:
            iterations_per_param = self.num_params
        return iterations_per_param * (self.num_params + 2)

    def generate_unitcube_samples_based_on_method(self, iterations):
        """Generate samples in [0,1] range based on Saltelli block sampling design."""
        return saltelli_samples(iterations, self.num_params)

    def generate_gsa_indices_based_on_method(self, **kwargs):
        """Uses Saltelli samples to compute first and total order Sobol indices."""
        selected_iterations = kwargs.get("selected_iterations")
        S_dict = sobol_indices(self.filepath_Y, self.num_params, selected_iterations)
        return S_dict
