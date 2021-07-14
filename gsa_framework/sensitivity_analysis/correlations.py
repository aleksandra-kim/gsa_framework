from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_methods.correlations import *
from ..utils import write_pickle, read_pickle


# TODO add possibility to pass X or filepath_X or filepath_X_rescaled as an argument


class Correlations(SAM):
    """Global sensitivity analysis with correlation coefficients."""

    gsa_label = "correlationsGsa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterations = self.calculate_iterations()
        self.write_dir_convergence = (
            self.write_dir / "convergence_intermediate_{}".format(self.gsa_label)
        )  # TODO
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)

    def calculate_iterations(self, interval_width=0.1):
        """Computes minimum number of iterations to obtain confidence intervals of width ``interval_width``."""
        iterations_dict = get_corrcoef_num_iterations(interval_width=interval_width)
        iterations = max(
            iterations_dict["pearson"],
            iterations_dict["spearman"],
        )
        return max(iterations, self.iterations)

    def create_S_convergence_filepath(self, iterations_step, iterations):
        filename = "S.{}.{}.{}Step{}.{}.pickle".format(
            self.gsa_label,
            self.sampling_label,
            iterations,
            iterations_step,
            self.seed,
        )
        filepath = self.write_dir_convergence / filename
        return filepath

    def generate_gsa_indices_based_on_method(self, **kwargs):
        """Uses random samples to compute correlation coefficients."""
        flag_convergence = kwargs.get("flag_convergence", False)
        if not flag_convergence:
            S_dict = correlation_coefficients(
                self.filepath_Y, self.filepath_X_rescaled, self.cpus
            )
        else:
            selected_iterations = kwargs.get("selected_iterations")
            iterations = kwargs.get("iterations", self.iterations)
            iterations_step = kwargs.get("iterations_step", self.iterations)
            filepath_S = self.create_S_convergence_filepath(iterations_step, iterations)
            if not filepath_S.exists():
                S_dict = correlation_coefficients(
                    self.filepath_Y,
                    self.filepath_X_rescaled,
                    self.cpus,
                    selected_iterations,
                )
                write_pickle(S_dict, filepath_S)
            else:
                S_dict = read_pickle(filepath_S)
        return S_dict
