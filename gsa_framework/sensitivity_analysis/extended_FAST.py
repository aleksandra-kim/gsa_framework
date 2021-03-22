# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import eFAST_samples
from ..sensitivity_methods.extended_FAST import eFAST_indices
from ..utils import write_hdf5_array, read_hdf5_array
import numpy as np


class eFAST(SAM):
    sampling_label = "eFastSampling"
    gsa_label = "eFastGsa"

    def __init__(self, M=4, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.iterations = self.calculate_iterations(self.iterations)
        self.sampling_label = self.sampling_label + "M{}".format(
            M
        )  # TODO where should this be?
        self.write_dir_convergence = (
            self.write_dir / "convergence_intermediate_{}".format(self.sampling_label)
        )
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)

    def calculate_iterations(self, iterations):
        return max(
            (4 * self.M ** 2 + 1) * self.num_params, iterations
        )  # Sample size N > 4M^2 is required. M=4 by default.

    def generate_unitcube_samples_based_on_method(self, iterations):
        X = eFAST_samples(
            iterations, self.num_params, M=self.M, seed=self.seed, cpus=self.cpus
        )
        return X

    def create_Y_convergence_filepath(self, iterations_step, iterations):
        filename = "Y.{}.{}Step{}.{}.hdf5".format(
            self.sampling_label,
            iterations,
            iterations_step,
            self.seed,
        )
        filepath = self.write_dir_convergence / filename
        return filepath

    def generate_gsa_indices_based_on_method(self, **kwargs):
        flag_convergence = kwargs.get("flag_convergence", False)
        if not flag_convergence:
            S_dict = eFAST_indices(
                self.filepath_Y,
                self.iterations,
                self.num_params,
                self.M,
            )
        else:
            iterations = kwargs.get("iterations", self.iterations)
            iterations_step = kwargs.get("iterations_step", self.iterations)
            filepath_Y = self.create_Y_convergence_filepath(iterations_step, iterations)
            if not filepath_Y.exists():
                X = self.generate_unitcube_samples_based_on_method(iterations)
                X_rescaled = self.model.rescale(X)
                Y = self.model(X_rescaled)
                write_hdf5_array(Y, filepath_Y)
            S_dict = eFAST_indices(
                filepath_Y,
                iterations,
                self.num_params,
                self.M,
            )
        return S_dict
