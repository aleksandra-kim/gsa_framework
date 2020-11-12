# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_analysis.gradient_boosting import xgboost_scores
from ..utils import write_pickle, read_pickle


class GradientBoosting(SAM):
    gsa_label = "xgboostGsa"

    def __init__(
        self, tuning_parameters=None, num_boost_round=10, xgb_model=None, **kwargs
    ):
        super().__init__(**kwargs)
        if tuning_parameters is None:
            tuning_parameters = {}
        tuning_parameters.update({"random_state": self.seed})
        self.tuning_parameters = tuning_parameters
        self.num_boost_round = num_boost_round
        self.gsa_label = self.gsa_label + "N{}D{}E{}S{}".format(
            self.num_boost_round,
            self.tuning_parameters.get("max_depth"),
            int(self.tuning_parameters.get("eta") * 100),
            int(self.tuning_parameters.get("subsample") * 100),
        )
        self.xgb_model = xgb_model

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
        flag_convergence = kwargs.get("flag_convergence", False)
        return_stats = kwargs.get("return_stats", False)
        if not flag_convergence:
            S_dict, r2, explained_var = xgboost_scores(
                filepath_Y=self.filepath_Y,
                filepath_X=self.filepath_X_rescaled,
                iterations=self.iterations,
                tuning_parameters=self.tuning_parameters,
                num_boost_round=self.num_boost_round,
                xgb_model=self.xgb_model,
            )
            # print("XGBoost training results: \n "
            #       "  r2={0:4.3f}, explained_variance={1:4.3f} \n".format(r2, explained_var))
        else:
            write_dir_convergence = (
                self.write_dir / "convergence_intermediate_{}".format(self.gsa_label)
            )  # TODO
            write_dir_convergence.mkdir(parents=True, exist_ok=True)
            iterations = kwargs.get("iterations", self.iterations)
            iterations_step = kwargs.get("iterations_step", self.iterations)
            filepath_S = self.create_S_convergence_filepath(iterations_step, iterations)
            if not filepath_S.exists():
                S_dict, r2, explained_var = xgboost_scores(
                    filepath_Y=self.filepath_Y,
                    filepath_X=self.filepath_X_rescaled,
                    iterations=iterations,
                    tuning_parameters=self.tuning_parameters,
                    num_boost_round=self.num_boost_round,
                    xgb_model=self.xgb_model,
                )
                write_pickle(S_dict, filepath_S)
            else:
                S_dict = read_pickle(filepath_S)
        if return_stats:
            return S_dict, r2, explained_var
        else:
            return S_dict
