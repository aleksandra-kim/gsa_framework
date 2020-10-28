import brightway2 as bw
import numpy as np
import os
import pickle
from copy import deepcopy
from stats_arrays import uncertainty_choices, MCRandomNumberGenerator

# Local imports
from ..model_base import ModelBase
from ..utils import read_hdf5_array
from ..utils_setac_lca import get_amounts_shift, get_score_shift

# ###############
# ## Glossary ###
# ###############
#
#     LSA     Local Sensitivity Analysis
#     GSA     Global Sensitivity Analysis
#     LCA     Life Cycle Assessment
#     LCIA    Life Cycle Impact Assessment
#
# #########################################


class LCAModel(ModelBase):
    """Class that implements basic LCA model which uses uncertainty in the background database.

    Parameters
    ----------
    func_unit : dict
        Dictionary of the form {bw_demand_activity: amount}.
    method : tuple
        Tuple with an impact assessment method.
    write_dir : str
        Directory where intermediate results will be stored.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Returns LCIA scores when technosphere exchanges are sampled from their respective distributions.

    """

    def __init__(
        self,
        func_unit,
        method,
        write_dir,
        var_threshold=0,
    ):
        self.func_unit = func_unit
        self.method = method
        self.lca = bw.LCA(self.func_unit, self.method)
        self.lca.lci()
        self.lca.lcia()
        self.write_dir = write_dir
        self.make_dirs()
        self.var_threshold = var_threshold
        self.uncertain_tech_params_where = np.where(
            self.lca.tech_params["uncertainty_type"] > 1
        )[0]
        self.uncertain_tech_params = self.lca.tech_params[
            self.uncertain_tech_params_where
        ]
        self.default_uncertain_amounts = get_amounts_shift(
            self.uncertain_tech_params, shift_median=False
        )
        self.static_output = get_score_shift(
            self.default_uncertain_amounts, self.uncertain_tech_params_where, self.lca
        )
        method_unit = bw.Method(self.method).metadata["unit"]
        self.output_name = "LCIA scores, [{}]".format(method_unit)

        # self.uncertain_tech_params_where = self.get_LSA_params(
        #     self.var_threshold
        # )  # TODO change the threshold
        # self.uncertain_tech_params = self.lca.tech_params[
        #     self.uncertain_tech_params_where
        # ]

        self.num_params = self.__len__()
        self.influential_params = []

        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        directories = {"LSA_scores": os.path.join(self.write_dir, "LSA_scores")}
        for dir in directories.values():
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.directories = directories

    def get_lsa_scores_pickle(self, path):
        """Get LCIA scores stored in the ``path``, where each parameter was sampled only few (eg 3-10) times.

        Parameters
        ----------
        path : str
            Path that contains pickle files with LCIA scores, where each uncertain exchange in the technosphere
            is varied independently of all other exchanges but only few times. Needed to determine which exchanges
            do NOT change LCIA score of the given functional unit and method, and hence, can be disregarded in the GSA.
            Pickle files explicitly contain ``(database, code)`` tuples for input and output activities, in order to
            most certainly define exchanges.

        Returns
        -------
        scores : dict
            Keys are indices of the exchanges as they appear in the lca.tech_params, values are LCIA scores.

        """
        filepath_all = os.path.join(path, "LSA_scores.pickle")
        if os.path.isfile(filepath_all):
            with open(filepath_all, "rb") as f:
                scores_ = pickle.load(f)
            scores = {int(k): v for k, v in scores_.items()}
        else:
            files = [
                f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and "LSA_scores_" in f
            ]
            starts = [int(f.split("_")[2]) for f in files]
            ind_sort = np.argsort(starts)
            files_sorted = [files[i] for i in ind_sort]

            scores = {}
            for file in files_sorted:
                filepath = os.path.join(path, file)
                with open(filepath, "rb") as f:
                    temp = pickle.load(f)
                temp_int = {int(k): v["scores"] for k, v in temp.items()}
                scores.update(temp_int)
        return scores

    def get_nonzero_params(self, scores_dict, var_threshold):
        """Given a dictionary of LSA scores, finds parameters that have variance below and above the threshold.

        Variance of the LSA scores that is smaller than eg 1e-20 is due to numerical imprecision. That means,
        ``params_no`` can be disregarded in the subsequent GSA.

        Parameters
        ----------
        scores_dict : dict
            Keys are indices of the exchanges as they appear in the lca.tech_params, values are LCIA scores.
        var_threshold : float
            Threshold for the variance of LSA scores.

        Returns
        -------
        params_no : np.array
            Indices of exchanges (as in lca.tech_params), where the variance of LSA scores is BELOW ``var_threshold``.
        params_yes : np.array
            Indices of exchanges (as in lca.tech_params), where the variance of LSA scores is ABOVE ``var_threshold``.

        """
        keys = np.array(list(scores_dict.keys()))
        vals = np.array(list(scores_dict.values()))

        # Variance of LSA scores for each input / parameter
        var = np.var(vals, axis=1)
        where = np.where(var > var_threshold)[0]

        params_yes = keys[where]
        params_no = np.setdiff1d(keys, params_yes)
        params_yes.sort(), params_no.sort()

        return params_no, params_yes

    def get_LSA_params(self, var_threshold):
        """Get ``params_yes`` with a specific threshold.

        If the given threshold is being used for the first time, ``params_yes`` will be computed using the function
        ``get_nonzero_params`` and saved to the LSA_scores directory, otherwise it is loaded which saves time.

        """
        params_yes_filename = os.path.join(
            self.directories["LSA_scores"],
            "params_yes_" + str(var_threshold) + ".pickle",
        )
        if not os.path.exists(params_yes_filename):
            scores_dict = self.get_lsa_scores_pickle(self.directories["LSA_scores"])
            _, params_yes = self.get_nonzero_params(
                scores_dict, var_threshold=var_threshold
            )
            with open(params_yes_filename, "wb") as f:
                pickle.dump(params_yes, f)
        else:
            with open(params_yes_filename, "rb") as f:
                params_yes = pickle.load(f)

        return params_yes

    def __len__(self):
        # self.uncertain_bio_params  = self.lca.bio_params[self.lca.bio_params['uncertainty_type'] > 1]
        # self.uncertain_cf_params   = self.lca.cf_params[self.lca.cf_params['uncertainty_type'] > 1]
        return len(
            self.uncertain_tech_params
        )  # + len(self.uncertain_bio_params) + len(self.lca.cf_params)

    def rescale(self, X):
        iterations, num_params = X.shape[0], X.shape[1]
        assert num_params == self.uncertain_tech_params.shape[0]

        X_reordered = X[:, self.mc.ordering]
        X_rescaled = np.zeros((iterations, num_params))
        X_rescaled[:] = np.nan

        offset = 0
        for uncertainty_type in self.choices:
            num_uncertain_params = self.mc.positions[uncertainty_type]
            if not num_uncertain_params:
                continue
            random_data = uncertainty_type.ppf(
                params=self.mc.params[offset : num_uncertain_params + offset],
                percentages=X_reordered[:, offset : num_uncertain_params + offset].T,
            )
            X_rescaled[:, offset : num_uncertain_params + offset] = random_data.T
            offset += num_uncertain_params

        X_rescaled = X_rescaled[:, np.argsort(self.mc.ordering)]
        return X_rescaled

    def __call__(self, X):
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i, x in enumerate(X):
            amounts = deepcopy(self.lca.tech_params["amount"])
            amounts[self.uncertain_tech_params_where] = x
            self.lca.rebuild_technosphere_matrix(amounts)
            self.lca.redo_lci()
            self.lca.redo_lcia()
            scores[i] = self.lca.score
        return scores


# class ParameterizedLCAModel:
#     """A model which does LCA of parameterized models.
#
#     Builds on the ``presamples`` library. For each Monte Carlo iteration, the model is evaluated,
#     and the evaluation result is saved into an ``overrides`` processed array.
#     This array is then used in building the LCA matrices.
#
#     """
#     pass


class LCAModelSetac(ModelBase):
    """Class that implements basic LCA model which uses uncertainty in the background database.

    Parameters
    ----------
    func_unit : dict
        Dictionary of the form {bw_demand_activity: amount}.
    method : tuple
        Tuple with an impact assessment method.
    write_dir : str
        Directory where intermediate results will be stored.

    Returns
    -------
    y : np.array of size [iterations, 1]
        Returns LCIA scores when technosphere exchanges are sampled from their respective distributions.

    """

    def __init__(
        self,
        func_unit,
        method,
        tech_params,
    ):
        self.func_unit = func_unit
        self.method = method
        self.lca = bw.LCA(self.func_unit, self.method)
        self.lca.lci()
        self.lca.lcia()
        self.tech_params = tech_params
        self.uncertain_tech_params_where = np.where(
            self.tech_params["uncertainty_type"] > 1
        )[0]
        self.uncertain_tech_params = self.tech_params[self.uncertain_tech_params_where]
        self.num_params = self.__len__()
        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)

    def __len__(self):
        return len(self.uncertain_tech_params)

    def rescale(self, X, filepath_base_X_rescaled=None):
        if filepath_base_X_rescaled.exists():
            X_rescaled = read_hdf5_array(filepath_base_X_rescaled)
        else:
            iterations, num_params = X.shape[0], X.shape[1]
            assert num_params == self.uncertain_tech_params.shape[0]

            X_reordered = X[:, self.mc.ordering]
            X_rescaled = np.zeros((iterations, num_params))
            X_rescaled[:] = np.nan

            offset = 0
            for uncertainty_type in self.choices:
                num_uncertain_params = self.mc.positions[uncertainty_type]
                if not num_uncertain_params:
                    continue
                random_data = uncertainty_type.ppf(
                    params=self.mc.params[offset : num_uncertain_params + offset],
                    percentages=X_reordered[
                        :, offset : num_uncertain_params + offset
                    ].T,
                )
                X_rescaled[:, offset : num_uncertain_params + offset] = random_data.T
                offset += num_uncertain_params

            X_rescaled = X_rescaled[:, np.argsort(self.mc.ordering)]
        return X_rescaled

    def __call__(self, X):
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i, x in enumerate(X):
            amounts = deepcopy(self.tech_params["amount"])
            amounts[self.uncertain_tech_params_where] = x
            self.lca.rebuild_technosphere_matrix(amounts)
            self.lca.redo_lci()
            self.lca.redo_lcia()
            scores[i] = self.lca.score
        return scores
