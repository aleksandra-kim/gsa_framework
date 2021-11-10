import numpy as np
from pathlib import Path
import pickle
from copy import deepcopy
from stats_arrays import uncertainty_choices, MCRandomNumberGenerator
import bw2calc as bc
import bw2data as bd

# Local imports
from gsa_framework.models.model_base import ModelBase
from ..utils import read_pickle, write_pickle, all_exc_same
from .utils_lca import get_amounts_shift, get_lca_score_shift

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


class LCAModelBase(ModelBase):
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
        uncertain_params,
        uncertain_params_selected_where_dict=None,
    ):
        self.func_unit = func_unit
        self.method = method
        self.lca = bc.LCA(self.func_unit, self.method)
        self.lca.lci()
        self.lca.lcia()
        self.uncertain_params = uncertain_params
        self.uncertain_exchanges_types = list(self.uncertain_params.keys())
        if uncertain_params_selected_where_dict is None:
            self.uncertain_params_selected_where_dict = {}
            for uncertain_exchange_type in self.uncertain_exchanges_types:
                params = self.get_params(uncertain_exchange_type)
                params_temp = []
                for p in self.uncertain_params[uncertain_exchange_type]:
                    where = np.where(
                        np.logical_and(
                            params["amount"] == p["amount"],
                            np.logical_and(
                                params["col"] == p["col"],
                                params["row"] == p["row"],
                            ),
                        )
                    )
                    assert len(where[0]) == 1
                    params_temp.append(where[0][0])
                self.uncertain_params_selected_where_dict[
                    uncertain_exchange_type
                ] = params_temp
        else:
            self.uncertain_params_selected_where_dict = (
                uncertain_params_selected_where_dict
            )
        self.num_params, self.uncertain_exchange_lengths = self.initialize(
            self.uncertain_params_selected_where_dict,
        )
        self.choices = uncertainty_choices
        method_unit = bd.Method(self.method).metadata["unit"]
        self.output_name = "LCIA scores, [{}]".format(method_unit)

    def initialize(self, uncertain_params_selected_where_dict):
        num_params = len(self)
        uncertain_exchange_lengths = {
            k: len(v) for k, v in uncertain_params_selected_where_dict.items()
        }
        return num_params, uncertain_exchange_lengths

    # def get_params_from_params_dict(self, exchange_type):
    #     return self.params_dict[exchange_type]

    def get_params(self, exchange_type):
        if exchange_type == "tech":
            return self.lca.tech_params
        elif exchange_type == "bio":
            return self.lca.bio_params
        elif exchange_type == "cf":
            return self.lca.cf_params

    def __len__(self):
        return sum([len(v) for v in self.uncertain_params_selected_where_dict.values()])

    def rescale(self, X):
        iterations, num_params = X.shape[0], X.shape[1]
        assert num_params == len(self)
        params_offset = 0
        X_rescaled_all = np.zeros((iterations, 0))
        for exchange_type in self.uncertain_exchanges_types:
            mc = MCRandomNumberGenerator(self.uncertain_params[exchange_type])
            X_reordered = X[:, mc.ordering + params_offset]

            X_rescaled = np.zeros(
                (iterations, self.uncertain_exchange_lengths[exchange_type])
            )
            X_rescaled[:] = np.nan

            offset = 0
            for uncertainty_type in self.choices:
                num_uncertain_params = mc.positions[uncertainty_type]
                if not num_uncertain_params:
                    continue
                random_data = uncertainty_type.ppf(
                    params=mc.params[offset : num_uncertain_params + offset],
                    percentages=X_reordered[
                        :, offset : num_uncertain_params + offset
                    ].T,
                )
                X_rescaled[:, offset : num_uncertain_params + offset] = random_data.T
                offset += num_uncertain_params

            X_rescaled_all = np.hstack(
                [X_rescaled_all, X_rescaled[:, np.argsort(mc.ordering)]]
            )
            params_offset += self.uncertain_exchange_lengths[exchange_type]
        return X_rescaled_all

    def __call__(self, X):
        lca = deepcopy(self.lca)
        scores = np.zeros(X.shape[0])
        scores[:] = np.nan
        for i, x in enumerate(X):
            params_offset = 0
            for exchange_type in self.uncertain_exchanges_types:
                amounts = deepcopy(self.get_params(exchange_type)["amount"])
                params_offset_next = (
                    params_offset + self.uncertain_exchange_lengths[exchange_type]
                )
                amounts[self.uncertain_params_selected_where_dict[exchange_type]] = x[
                    params_offset:params_offset_next
                ]
                params_offset = params_offset_next
                if exchange_type == "tech":
                    lca.rebuild_technosphere_matrix(amounts)
                elif exchange_type == "bio":
                    lca.rebuild_biosphere_matrix(amounts)
                elif exchange_type == "cf":
                    lca.rebuild_characterization_matrix(amounts)
            lca.redo_lci()
            lca.redo_lcia()
            scores[i] = lca.score
        return scores


class LCAModel(LCAModelBase):
    """Class that implements basic LCA model which uses uncertainty in the background database.

    Parameters
    ----------
    func_unit : dict
        Dictionary of the form {bw_demand_activity: amount}.
    method : tuple
        Tuple with an impact assessment method.
    write_dir : str
        Directory where intermediate results will be stored.
    num_params : int
        Number of parameters that can be influential, the rest are removed based on local sensitivity analysis.
    lca : bw.LCA object
        Instance of an LCA class from brightway library.

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
        num_params=None,
        uncertain_exchanges_types=("tech", "bio", "cf"),
    ):
        self.lca = bc.LCA(func_unit, method)
        self.lca.lci()
        self.lca.lcia()
        # for uncertain_exchange_type in uncertain_exchanges_types:
        #     uncertain_params[uncertain_exchange_type] = self.get_uncertain_params_all(uncertain_exchange_type)

        self.write_dir = Path(write_dir)
        self.make_dirs()
        if num_params is not None:
            self.scores_dict_raw = self.get_lsa_scores_pickle(
                self.write_dir / "LSA_scores", uncertain_exchanges_types
            )
            self.scores_dict = {}
            for exchanges_type in uncertain_exchanges_types:
                self.scores_dict[exchanges_type] = self.scores_dict_raw[exchanges_type]
            self.uncertain_params_selected_where_dict = (
                self.get_nonzero_params_from_num_params(self.scores_dict, num_params)
            )
        else:
            self.uncertain_params_selected_where_dict = {}
            for exchanges_type in uncertain_exchanges_types:
                self.uncertain_params_selected_where_dict[exchanges_type] = np.where(
                    self.get_params(exchanges_type)["uncertainty_type"] > 1
                )[0]
        uncertain_params = {}
        for uncertain_exchange_type in uncertain_exchanges_types:
            uncertain_params[uncertain_exchange_type] = self.get_params(
                uncertain_exchange_type
            )[self.uncertain_params_selected_where_dict[uncertain_exchange_type]]

        super().__init__(
            func_unit,
            method,
            uncertain_params,
            self.uncertain_params_selected_where_dict,
        )
        self.num_params, self.uncertain_exchange_lengths = self.initialize(
            self.uncertain_params_selected_where_dict,
        )
        self.default_uncertain_amounts = get_amounts_shift(
            self.uncertain_params, shift_median=False
        )
        self.static_output = get_lca_score_shift(
            self.default_uncertain_amounts,
            self.uncertain_params_selected_where_dict,
            self.lca,
        )
        self.adjusted_score = self.static_output - self.lca.score  # 2675.372419737564

    def make_dirs(self):
        """Create subdirectories where intermediate results can be stored."""
        dirs_list = [
            "arrays",
            "figures",
            "LSA_scores",
        ]  # TODO maybe add logging later on
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    #
    # def get_params(self, exchange_type):
    #     if exchange_type == "tech":
    #         return self.lca.tech_params
    #     elif exchange_type == "bio":
    #         return self.lca.bio_params
    #     elif exchange_type == "cf":
    #         return self.lca.cf_params

    def get_scores_dict_from_params(self, exchanges_type, scores, inputs, outputs=None):
        """Get scores_dict where keys are indices of exchanges in tech_params/bio_params/cf_params, and values are LSA scores."""

        assert len(inputs) == len(scores)

        if exchanges_type == "tech":
            inputs_dict = self.lca.activity_dict
            outputs_dict = self.lca.activity_dict
            input_params = self.lca.tech_params
        elif exchanges_type == "bio":
            inputs_dict = self.lca.biosphere_dict
            outputs_dict = self.lca.activity_dict
            input_params = self.lca.bio_params
        elif exchanges_type == "cf":
            inputs_dict = self.lca.biosphere_dict
            input_params = self.lca.cf_params

        num_exchanges = len(inputs)
        input_row_dict = {}
        for input_ in list(set(inputs)):
            input_row_dict[input_] = inputs_dict[input_]
        if exchanges_type != "cf":
            output_col_dict = {}
            for output_ in list(set(outputs)):
                output_col_dict[output_] = outputs_dict[output_]

        scores_dict = {}
        for i in range(num_exchanges):
            row = input_row_dict[inputs[i]]
            if exchanges_type != "cf":
                col = output_col_dict[outputs[i]]
                where_temp = np.where(
                    np.logical_and(
                        np.logical_and(
                            input_params["row"] == row,
                            input_params["col"] == col,
                        ),
                        input_params["uncertainty_type"] > 1,
                    )
                )[0]
            else:
                where_temp = np.where(
                    np.logical_and(
                        input_params["row"] == row,
                        input_params["uncertainty_type"] > 1,
                    )
                )[0]
            if len(where_temp) == 1:
                scores_dict[where_temp[0]] = scores[i]
            elif len(where_temp) > 1:
                temp_params = input_params[where_temp]
                flag_all_excs_same = all_exc_same(temp_params)
                if flag_all_excs_same:
                    scores_dict[where_temp[0]] = scores[
                        i
                    ]  # so we can take any exchange
                else:
                    ind = np.where(temp_params["scale"] == max(temp_params["scale"]))[
                        0
                    ]  # take max scale
                    if (
                        len(ind) == 1
                    ):  # if only one exc with max scale, then just take it
                        scores_dict[where_temp[ind[0]]] = scores[i]
                    elif len(ind) > 1 and all_exc_same(
                        temp_params[ind]
                    ):  # if multiple, but they're all the same, take any
                        scores_dict[where_temp[ind[0]]] = scores[i]
                    else:
                        ind = np.where(temp_params["loc"] == max(temp_params["loc"]))[
                            0
                        ]  # take max loc
                        if (
                            len(ind) == 1
                        ):  # if only one exc with max loc, then just take it
                            scores_dict[where_temp[ind[0]]] = scores[i]
                        elif len(ind) > 1 and all_exc_same(
                            temp_params[ind]
                        ):  # if multiple, but all the same, take any
                            scores_dict[where_temp[ind[0]]] = scores[i]
                        else:
                            print(temp_params)
            else:
                print("exchange was not found")

        return scores_dict

    def get_inputs_outputs_scores_from_files(self, path, files_sorted):
        """Get inputs, outputs and scores from multiple LSA_scores files."""
        scores, inputs, outputs = [], [], []
        for file in files_sorted:
            filepath = path / file
            temp = read_pickle(filepath)
            inputs += [vals["input"] for vals in temp.values()]
            outputs += [vals.get("output") for vals in temp.values()]
            scores += [vals["scores"] for vals in temp.values()]
        return scores, inputs, outputs

    def get_lsa_scores_tech_files(self, path):
        """Function that finds all `LSA_scores*` files"""
        files = [
            filepath.name
            for filepath in path.iterdir()
            if "LSA_scores_" in filepath.name and filepath.is_file()
        ]
        starts = [int(filepath.split("_")[2]) for filepath in files]
        ind_sort = np.argsort(starts)
        files_sorted = [files[i] for i in ind_sort]
        return files_sorted

    def get_lsa_scores_dict(self, path, exchanges_type):
        """Retrieve scores_dict for tech, bio and cf exchanges."""
        filepath_scores_dict = path / "scores_dict_{}.pickle".format(exchanges_type)
        if filepath_scores_dict.exists():
            scores_dict = read_pickle(filepath_scores_dict)
        else:
            if exchanges_type == "tech":
                path_tech = path / "tech"
                files = self.get_lsa_scores_tech_files(path_tech)
                scores, inputs, outputs = self.get_inputs_outputs_scores_from_files(
                    path_tech, files
                )
            else:
                files = ["LSA_scores_{}.pickle".format(exchanges_type)]
                scores, inputs, outputs = self.get_inputs_outputs_scores_from_files(
                    path, files
                )
            if exchanges_type == "cf":
                scores_dict = self.get_scores_dict_from_params(
                    exchanges_type, scores, inputs
                )
            else:
                scores_dict = self.get_scores_dict_from_params(
                    exchanges_type, scores, inputs, outputs
                )
            write_pickle(scores_dict, filepath_scores_dict)
        return scores_dict

    def get_lsa_scores_pickle(self, path, uncertain_exchanges_types):
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
            Keys are indices of the exchanges as they appear in the models.tech_params, values are LCIA scores.

        """
        path = Path(path)
        exchanges_types = "_".join(uncertain_exchanges_types)
        filepath_scores_dict = self.write_dir / "scores_dict_{}.pickle".format(
            exchanges_types
        )
        if filepath_scores_dict.exists():
            scores_dict = read_pickle(filepath_scores_dict)
        else:
            scores_dict = {}
            for uncertain_exchange_type in uncertain_exchanges_types:
                scores_dict[uncertain_exchange_type] = self.get_lsa_scores_dict(
                    path, uncertain_exchange_type
                )
            write_pickle(scores_dict, filepath_scores_dict)
        return scores_dict

    def get_where_high_var(self, scores_dict, num_params):
        vals = np.zeros([0, 3])
        for scores_dict_exchange_type in scores_dict.values():
            vals_temp = np.array(list(scores_dict_exchange_type.values()))
            vals_temp = np.hstack(
                [vals_temp, np.tile(self.lca.score, (len(vals_temp), 1))]
            )
            vals = np.vstack([vals, vals_temp])
        # Variance of LSA scores for each input / parameter
        var = np.var(vals, axis=1)
        where_high_var = np.argsort(var)[::-1][:num_params]
        assert np.all(var[where_high_var] > 0)
        where_high_var = np.sort(where_high_var)
        return where_high_var

    def get_nonzero_params_from_num_params(self, scores_dict, num_params):
        where_high_var = self.get_where_high_var(scores_dict, num_params)
        len_curr, len_next = 0, 0
        params_selected_where_dict = {}
        for exchange_type, scores_dict_exchange_type in scores_dict.items():
            params_inds = np.array(list(scores_dict_exchange_type.keys()))
            len_next += len(params_inds)
            where = (
                where_high_var[
                    np.logical_and(
                        where_high_var >= len_curr, where_high_var < len_next  # TODO
                    )
                ]
                - len_curr
            )
            len_curr = len_next
            params_selected_where = params_inds[where]
            params_selected_where_dict[exchange_type] = np.sort(params_selected_where)
        return params_selected_where_dict

    def get_nonzero_params_from_var_threshold(self, scores_dict, var_threshold):
        """Given a dictionary of LSA scores, finds parameters that have variance below and above the threshold.

        Variance of the LSA scores that is smaller than eg 1e-20 is due to numerical imprecision. That means,
        ``params_no`` can be disregarded in the subsequent GSA.

        Parameters
        ----------
        scores_dict : dict
            Keys are indices of the exchanges as they appear in the models.tech_params, values are LCIA scores.
        var_threshold : float
            Threshold for the variance of LSA scores.

        Returns
        -------
        params_no : np.array
            Indices of exchanges (as in models.tech_params), where the variance of LSA scores is BELOW ``var_threshold``.
        params_yes : np.array
            Indices of exchanges (as in models.tech_params), where the variance of LSA scores is ABOVE ``var_threshold``.

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


# class ParameterizedLCAModel:
#     """A model which does LCA of parameterized models.
#
#     Builds on the ``presamples`` library. For each Monte Carlo iteration, the model is evaluated,
#     and the evaluation result is saved into an ``overrides`` processed array.
#     This array is then used in building the LCA matrices.
#
#     """
#     pass
