# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_analysis.gradient_boosting import xgboost_scores


class GradientBoosting(SAM):
    gsa_label = "gsa_xgboost"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_gsa_indices_based_on_method(self, selected_iterations=None):
        S_dict, r2, explained_var = xgboost_scores(
            self.filepath_Y,
            self.filepath_X_rescaled,
            self.iterations,
            self.num_params,
            write_dir=self.write_dir,
        )
        return S_dict
