# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import latin_hypercube_samples
from ..sensitivity_analysis.delta_moment import delta_moment
from ..utils import write_pickle, read_pickle


class DeltaMoment(SAM):
    sampling_label = "latinSampling"
    gsa_label = "deltaGsa"

    def __init__(self, num_resamples=10, **kwargs):
        super().__init__(**kwargs)
        self.num_resamples = num_resamples
        self.gsa_label = self.gsa_label + "Nr{}".format(self.num_resamples)
        self.write_dir_convergence = (
            self.write_dir / "convergence_intermediate_{}".format(self.gsa_label)
        )  # TODO
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)

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

    def generate_unitcube_samples_based_on_method(self, iterations):
        X = latin_hypercube_samples(iterations, self.num_params, seed=self.seed)
        return X

    def generate_gsa_indices_based_on_method(self, **kwargs):
        flag_convergence = kwargs.get("flag_convergence", False)
        if not flag_convergence:
            S_dict = delta_moment(
                self.filepath_Y,
                self.filepath_X_rescaled,
                self.iterations,
                self.num_params,
                seed=self.seed,
                num_resamples=self.num_resamples,
            )
        else:
            iterations = kwargs.get("iterations", self.iterations)
            iterations_step = kwargs.get("iterations_step", self.iterations)
            filepath_S = self.create_S_convergence_filepath(iterations_step, iterations)
            if not filepath_S.exists():
                S_dict = delta_moment(
                    filepath_Y=self.filepath_Y,
                    filepath_X=self.filepath_X_rescaled,
                    iterations=iterations,
                    num_params=self.num_params,
                    seed=self.seed,
                    num_resamples=self.num_resamples,
                )
                write_pickle(S_dict, filepath_S)
            else:
                S_dict = read_pickle(filepath_S)
        return S_dict
