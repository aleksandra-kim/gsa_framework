from .base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import eFAST_samples
from ..utils import read_hdf5_array, write_hdf5_array


class eFAST(SAM):
    label = "extended FAST"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = self.calculate_efast_iterations(self.iterations)

    def calculate_efast_iterations(self, iterations):
        M = 4
        return max(
            (4 * M ** 2 + 1) * self.num_params, iterations
        )  # Sample size N > 4M^2 is required. M=4 by default.

    def generate_normalized_samples(self, X=None):
        if not self.filename_X.exists():
            X = eFAST_samples()
            write_hdf5_array(X, self.filename_X)

        return X
