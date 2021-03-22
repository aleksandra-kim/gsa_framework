# Local files
from .method_base import SensitivityAnalysisMethod as SAM
from ..sensitivity_methods.gradient_boosting import xgboost_indices
from ..utils import write_pickle, read_pickle


class GradientBoosting(SAM):
    gsa_label = "xgboostGsa"

    def __init__(self, tuning_parameters=None, test_size=0.2, xgb_model=None, **kwargs):
        super().__init__(**kwargs)
        if tuning_parameters is None:
            tuning_parameters = {}
        tuning_parameters.update({"random_state": self.seed})
        self.tuning_parameters = tuning_parameters
        self.test_size = test_size
        self.xgb_model = xgb_model
        self.gsa_label = self.create_gsa_label()
        self.write_dir_convergence = (
            self.write_dir / "convergence_intermediate_{}".format(self.gsa_label)
        )  # TODO
        self.write_dir_convergence.mkdir(parents=True, exist_ok=True)
        self.write_dir_stability = self.write_dir / "stability_intermediate_{}".format(
            self.gsa_label
        )  # TODO
        self.write_dir_stability.mkdir(parents=True, exist_ok=True)

    # def create_S_convergence_filepath(self, iterations_step, iterations):
    #     filename = "S.{}.{}.{}Step{}.{}.pickle".format(
    #         self.gsa_label,
    #         self.sampling_label,
    #         iterations,
    #         iterations_step,
    #         self.seed,
    #     )
    #     filepath = self.write_dir_convergence / filename
    #     return filepath

    def create_gsa_label(self):
        gsa_label = self.gsa_label + "_Lr{}G{}Mcw{}Md{}RegL{}RegA{}Ne{}Ss{}Cbt{}_".format(  # TODO change to include more info in the filename
            self.tuning_parameters.get("learning_rate", 0.3),
            self.tuning_parameters.get("gamma", 0),
            self.tuning_parameters.get("min_child_weight", 1),
            self.tuning_parameters.get("max_depth", 6),
            self.tuning_parameters.get("reg_lambda", 0),
            self.tuning_parameters.get("reg_alpha", 0),
            self.tuning_parameters.get("n_estimators", 10),
            self.tuning_parameters.get("subsample", 1),
            self.tuning_parameters.get("colsample_bytree", 1),
        )
        return gsa_label

    def generate_gsa_indices_based_on_method(self, **kwargs):
        # flag_convergence = kwargs.get("flag_convergence", False)
        # if not flag_convergence:
        flag_return_xgb_model = kwargs.get("flag_return_xgb_model", True)
        S_dict = xgboost_indices(
            filepath_Y=self.filepath_Y,
            filepath_X=self.filepath_X_rescaled,
            tuning_parameters=self.tuning_parameters,
            test_size=self.test_size,
            xgb_model=self.xgb_model,
            flag_return_xgb_model=flag_return_xgb_model,
        )
        # else:
        #     iterations = kwargs.get("iterations", self.iterations)
        #     iterations_step = kwargs.get("iterations_step", self.iterations)
        #     filepath_S = self.create_S_convergence_filepath(iterations_step, iterations)
        #     if not filepath_S.exists():
        #         S_dict, r2, explained_var = xgboost_scores(
        #             filepath_Y=self.filepath_Y,
        #             filepath_X=self.filepath_X_rescaled,
        #             tuning_parameters=self.tuning_parameters,
        #             num_boost_round=self.num_boost_round,
        #             xgb_model=self.xgb_model,
        #         )
        #         write_pickle(S_dict, filepath_S)
        #     else:
        #         S_dict = read_pickle(filepath_S)
        return S_dict
