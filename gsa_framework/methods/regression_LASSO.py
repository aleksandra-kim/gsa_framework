# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_analysis.regression_LASSO import regression_lasso


class RegressionLASSO(SAM):
    gsa_label = "lassoGsa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_gsa_indices_based_on_method(self, selected_iterations=None):
        S_dict = regression_lasso(
            self.filepath_Y,
            self.filepath_X_rescaled,
            self.iterations,
            self.num_params,
            write_dir=self.write_dir,
        )
        return S_dict
