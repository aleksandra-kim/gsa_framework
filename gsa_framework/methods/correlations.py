from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_analysis.correlations import *


# TODO add possibility to pass X or filepath_X or filepath_X_rescaled as an argumemt


class CorrelationCoefficients(SAM):
    gsa_label = "gsa_correlations"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterations = self.calculate_iterations()

    def calculate_iterations(self):
        corrcoeff_constants = get_corrcoef_num_iterations()
        iterations = max(
            corrcoeff_constants["pearson"]["iterations"],
            corrcoeff_constants["spearman"]["iterations"],
        )
        return max(iterations, self.iterations)

    def generate_gsa_indices_based_on_method(self, selected_iterations=None):
        S_dict = correlation_coefficients(
            self.filepath_Y, self.filepath_X_rescaled, self.cpus, selected_iterations
        )
        return S_dict
