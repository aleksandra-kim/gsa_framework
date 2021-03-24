# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sampling.get_samples import latin_hypercube_samples
from ..sensitivity_methods.delta import delta_indices
from ..utils import write_pickle, read_pickle


class Delta(SAM):
    """Global sensitivity analysis with delta moment independent indices and latin hypercube sampling.

    References
    ----------
    Paper:
        :cite:ts:`borgonovo2007new`

    """

    sampling_label = "latinSampling"
    gsa_label = "deltaGsa"

    def __init__(self, num_resamples=1, **kwargs):
        super().__init__(**kwargs)
        self.num_resamples = num_resamples
        self.conf_level = kwargs.get("conf_level", 0.95)
        self.gsa_label = self.gsa_label + "Nr{}".format(self.num_resamples)
        self.write_dir_convergence = (
            self.write_dir / "convergence_intermediate_{}".format(self.gsa_label)
        )  # TODO
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)
        self.write_dir_stability = self.write_dir / "stability_intermediate_{}".format(
            self.gsa_label
        )  # TODO
        self.write_dir_stability.mkdir(parents=True, exist_ok=True)

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
        """Generate samples in [0,1] range based on latin hypercube sampling design."""
        X = latin_hypercube_samples(iterations, self.num_params, seed=self.seed)
        return X

    def generate_gsa_indices_based_on_method(self, **kwargs):
        """Uses latin hypercube samples to compute Borgonovo delta indices."""
        flag_convergence = kwargs.get("flag_convergence", False)
        if not flag_convergence:
            S_dict = delta_indices(
                self.filepath_Y,
                self.filepath_X_rescaled,
                self.num_resamples,
                self.conf_level,
                self.seed,
                self.cpus,
            )
        # else:
        #     iterations = kwargs.get("iterations", self.iterations)
        #     iterations_step = kwargs.get("iterations_step", self.iterations)
        #     filepath_S = self.create_S_convergence_filepath(iterations_step, iterations)
        #     if not filepath_S.exists():
        #         S_dict = delta_moment(
        #             filepath_Y=self.filepath_Y,
        #             filepath_X=self.filepath_X_rescaled,
        #             iterations=iterations,
        #             num_params=self.num_params,
        #             seed=self.seed,
        #             num_resamples=self.num_resamples,
        #         )
        #         write_pickle(S_dict, filepath_S)
        #     else:
        #         S_dict = read_pickle(filepath_S)
        return S_dict
