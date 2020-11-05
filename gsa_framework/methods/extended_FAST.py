# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import eFAST_samples
from ..sensitivity_analysis.extended_FAST import eFAST_indices
from ..utils import write_hdf5_array
import time
import numpy as np


class eFAST(SAM):
    sampling_label = "sampling_e_fast"
    gsa_label = "gsa_e_fast"

    def __init__(self, M=4, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.iterations = self.calculate_iterations(self.iterations)
        self.sampling_label = self.sampling_label + ".M{}".format(
            M
        )  # TODO where should this be?
        self.write_dir_convergence = self.write_dir / "convergence"
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)

    def calculate_iterations(self, iterations):
        return max(
            (4 * self.M ** 2 + 1) * self.num_params, iterations
        )  # Sample size N > 4M^2 is required. M=4 by default.

    def generate_unitcube_samples_based_on_method(self, iterations):
        X = eFAST_samples(
            self.num_params, iterations, M=self.M, seed=self.seed, cpus=self.cpus
        )
        return X

    def create_Y_convergence_filepath(self, block, iterations):
        filename = "Y.{}.block{}.{}.{}.{}.hdf5".format(
            self.sampling_label,
            block,
            iterations,
            self.num_params,
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
            X = self.generate_unitcube_samples_based_on_method(iterations)
            X_rescaled = self.model.rescale(X)
            Y = self.model(X_rescaled)
            filepath_Y = self.create_Y_convergence_filepath(iterations_step, iterations)
            write_hdf5_array(Y, filepath_Y)
            S_dict = eFAST_indices(
                filepath_Y,
                iterations,
                self.num_params,
                self.M,
            )
        return S_dict

    # def convergence(self, iterations_end, num_steps):
    #     iterations_start = self.calculate_iterations(1)
    #     iterations_step = max((iterations_end-iterations_start)//num_steps, 1)
    #     iterations_block = range(iterations_start, iterations_end, iterations_step)
    #     sa_convergence_dict_temp = {}
    #     for iterations in iterations_block:
    #         t0 = time.time()
    #
    #         t1 = time.time()
    #         print("{0:8d} iterations -> {1:8.3f} s".format(iterations, t1 - t0))
    #         sa_convergence_dict_temp[iterations] = S_dict
    #     # Put all blocks together
    #     sa_convergence_dict = {
    #         key: np.zeros(shape=(0, self.num_params))
    #         for key in sa_convergence_dict_temp[iterations_start].keys()
    #     }
    #     for sa_dict in sa_convergence_dict_temp.values():
    #         for key, sa_array in sa_convergence_dict.items():
    #             new_sa_array = np.vstack([sa_array, sa_dict[key]])
    #             sa_convergence_dict.update({key: new_sa_array})
    #     return sa_convergence_dict
