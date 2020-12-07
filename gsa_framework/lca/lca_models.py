import brightway2 as bw
import numpy as np
import os
from pathlib import Path
import pickle
from copy import deepcopy
from stats_arrays import uncertainty_choices, MCRandomNumberGenerator

# Local imports
from ..model_base import ModelBase
from ..utils import read_pickle, write_pickle, all_exc_same
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
        # var_threshold=0, #TODO should be either var_threshold or num_params
        num_params=None,
        lca=None,
    ):
        self.func_unit = func_unit
        self.method = method
        if lca is None:
            self.lca = bw.LCA(self.func_unit, self.method)
            self.lca.lci()
            self.lca.lcia()
        else:
            self.lca = deepcopy(lca)
        print(self.lca.score)
        self.write_dir = Path(write_dir)
        self.make_dirs()
        if num_params is None:
            self.uncertain_tech_params_where = np.where(
                self.lca.tech_params["uncertainty_type"] > 1
            )[0]
            self.uncertain_tech_params = self.lca.tech_params[
                self.uncertain_tech_params_where
            ]
            self.num_params = len(self.uncertain_tech_params)
        else:
            self.num_params = num_params
            self.scores_dict = self.get_lsa_scores_pickle(self.write_dir / "LSA_scores")
            (
                self.uncertain_tech_params_where,
                _,
            ) = self.get_nonzero_params_from_num_params(
                self.scores_dict, self.num_params
            )
            self.uncertain_tech_params = self.lca.tech_params[
                self.uncertain_tech_params_where
            ]
        self.default_uncertain_amounts = get_amounts_shift(
            self.uncertain_tech_params, shift_median=False
        )
        self.static_output = get_score_shift(
            self.default_uncertain_amounts, self.uncertain_tech_params_where, self.lca
        )
        self.adjusted_score = self.static_output - self.lca.score #2675.372419737564
        method_unit = bw.Method(self.method).metadata["unit"]
        self.output_name = "LCIA scores, [{}]".format(method_unit)
        self.influential_params = []
        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)

    def make_dirs(self):
        """Create subdirectories where intermediate results will be stored."""
        dirs_list = [
            "arrays",
            "figures",
            "LSA_scores",
        ]  # TODO maybe add logging later on
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

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
        path = Path(path)
        filepath_scores_dict = self.write_dir / "scores_dict.pickle"
        if filepath_scores_dict.exists():
            scores_dict = read_pickle(filepath_scores_dict)
        else:
            files = [
                filepath.name
                for filepath in path.iterdir()
                if "LSA_scores_" in filepath.name and filepath.is_file()
            ]
            starts = [int(filepath.split("_")[2]) for filepath in files]
            ind_sort = np.argsort(starts)
            files_sorted = [files[i] for i in ind_sort]

            scores, inputs, outputs = [], [], []
            for file in files_sorted:
                filepath = path / file
                temp = read_pickle(filepath)
                inputs += [vals["input"] for vals in temp.values()]
                outputs += [vals["output"] for vals in temp.values()]
                scores += [vals["scores"] for vals in temp.values()]
            num_exchanges = len(inputs)

            input_row_dict = {}
            for input_ in list(set(inputs)):
                input_row_dict[input_] = self.lca.activity_dict[input_]
            output_col_dict = {}
            for output_ in list(set(outputs)):
                output_col_dict[output_] = self.lca.activity_dict[output_]

            scores_dict = {}
            for i in range(num_exchanges):
                row = input_row_dict[inputs[i]]
                col = output_col_dict[outputs[i]]
                where_temp = np.where(
                    np.logical_and(
                        np.logical_and(
                            self.lca.tech_params["row"] == row,
                            self.lca.tech_params["col"] == col,
                        ),
                        self.lca.tech_params["uncertainty_type"] > 1,
                    )
                )[0]
                if len(where_temp) == 1:
                    scores_dict[where_temp[0]] = scores[i]
                elif len(where_temp) > 1:
                    temp_tech_params = self.lca.tech_params[where_temp]
                    flag_all_excs_same = all_exc_same(temp_tech_params)
                    if flag_all_excs_same: 
                        scores_dict[where_temp[0]] = scores[i] # so we can take any exchange
                    else:
                        ind = np.where(temp_tech_params['scale']==max(temp_tech_params['scale']))[0] # take max scale
                        if len(ind)==1: # if only one exc with max scale, then just take it
                            scores_dict[where_temp[ind[0]]] = scores[i]
                        elif len(ind)>1 and all_exc_same(temp_tech_params[ind]): # if multiple, but they're all the same, take any
                            scores_dict[where_temp[ind[0]]] = scores[i]
                        else:
                            ind = np.where(temp_tech_params['loc']==max(temp_tech_params['loc']))[0] # take max loc
                            if len(ind)==1: # if only one exc with max loc, then just take it
                                scores_dict[where_temp[ind[0]]] = scores[i]
                            elif len(ind)>1 and all_exc_same(temp_tech_params[ind]): # if multiple, but all the same, take any
                                scores_dict[where_temp[ind[0]]] = scores[i]
                            else:
                                print(temp_tech_params)
                else:
                    print("{} row and {} column exchange was not found".format(row,col))
            write_pickle(scores_dict, filepath_scores_dict)
        return scores_dict

    def get_nonzero_params_from_num_params(self, scores_dict, num_params):
        keys = np.array(list(scores_dict.keys()))
        vals = np.array(list(scores_dict.values()))
        vals = np.hstack([vals, np.tile(self.lca.score, (len(vals), 1))])
        # Variance of LSA scores for each input / parameter
        var = np.var(vals, axis=1)
        where_high_var = np.argsort(var)[::-1][:num_params]
        assert np.all(var[where_high_var] > 0)
        params_yes = keys[where_high_var]
        params_no = np.setdiff1d(keys, params_yes)
        params_yes.sort(), params_no.sort()
        return params_yes, params_no

    def get_nonzero_params_from_var_threshold(self, scores_dict, var_threshold):
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

        return params_yes, params_no  # TODO changed order here!!

    def get_LSA_params(self, var_threshold):
        """Get ``params_yes`` with a specific threshold.

        If the given threshold is being used for the first time, ``params_yes`` will be computed using the function
        ``get_nonzero_params`` and saved to the LSA_scores directory, otherwise it is loaded which saves time.

        """
        params_yes_filename = (
            self.write_dir / "LSA_scores" / "params_yes_{}.pickle".format(var_threshold)
        )
        if not params_yes_filename.exists():
            scores_dict = self.get_lsa_scores_pickle(self.write_dir / "LSA_scores")
            params_yes, _ = self.get_nonzero_params_from_var_threshold(
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
        lca = deepcopy(self.lca)
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i, x in enumerate(X):
            amounts = deepcopy(lca.tech_params["amount"])
            amounts[self.uncertain_tech_params_where] = x
            lca.rebuild_technosphere_matrix(amounts)
            lca.redo_lci()
            lca.redo_lcia()
            scores[i] = lca.score
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
        self.tech_params = tech_params
        self.uncertain_tech_params_where = np.where(
            self.tech_params["uncertainty_type"] > 1
        )[0]
        self.uncertain_tech_params = self.tech_params[self.uncertain_tech_params_where]
        self.num_params = self.__len__()
        self.choices = uncertainty_choices
        self.mc = MCRandomNumberGenerator(self.uncertain_tech_params)
        method_unit = bw.Method(self.method).metadata["unit"]
        self.output_name = "LCIA scores, [{}]".format(method_unit)

    def __len__(self):
        return len(self.uncertain_tech_params)

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
        lca = bw.LCA(self.func_unit, self.method)
        lca.lci()
        lca.lcia()
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i, x in enumerate(X):
            amounts = deepcopy(self.tech_params["amount"])
            amounts[self.uncertain_tech_params_where] = x
            lca.rebuild_technosphere_matrix(amounts)
            lca.redo_lci()
            lca.redo_lcia()
            scores[i] = lca.score
        return scores
