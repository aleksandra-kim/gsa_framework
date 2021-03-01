from sklearn.metrics import r2_score, explained_variance_score
import numpy as np
import json
import os
import xgboost as xgb
from ..utils import read_hdf5_array
from sklearn.model_selection import train_test_split

##############################
### Tuning parameters info ###
##############################
# eta : 0.3
#      `learning rate`  - range=[0,1], 0.1 is more common than 0.8,
# gamma : 0.0
#      `min_split_loss` - range=[0,\inf], tree complexity parameter, higher gamma -> more pruning
# min_child_weight : 1
#      `cover` - range=[0,\inf], minimum number of residuals in each leaf
# max_depth : 6
#      `maximum depth of a tree` - range=[0,\inf]
# lambda : 1
#      `reg_lambda` - L2 regularization, higher value -> more conservative model
# alpha : 1,
#      `reg_alpha`  - L1 regularization, higher value -> more conservative model
# n_estimators : 10
#       used in XGBRegressor, same as `num_boost_rounds` in xgb.train

# subsample : 1
#      `subsample ratio` - range=(0,1], trees built on randomly selected partial data
# colsample_bytree : 1
#      subsample ratio of columns when constructing each tree. there are other colsample options
# tree_method : "auto"
#      `tree construction alg.` - choices include auto, exact, approx, hist, gpu_hist


def xgboost_scores(
    filepath_Y,
    filepath_X,
    tuning_parameters=None,
    test_size=0.2,
    xgb_model=None,
    importance_types=None,
    flag_return_xgb_model=True,
):
    X = read_hdf5_array(filepath_X)
    Y = read_hdf5_array(filepath_Y).flatten()
    S_dict = xgboost_scores_base(
        Y,
        X,
        tuning_parameters,
        test_size,
        xgb_model,
        importance_types,
        flag_return_xgb_model,
    )
    return S_dict


def xgboost_scores_base(
    Y,
    X,
    tuning_parameters=None,
    test_size=0.2,
    xgb_model=None,
    importance_types=None,  # TODO set default to empty list?
    flag_return_xgb_model=True,
):
    """Compute fscores obtained from the gradient boosting machines regression using XGBoost library.

    Parameters
    ----------
    dict_ : dict
        Dictionary that contains parameter sampling matrix ``X``, model outputs ``y``, number of iterations
        ``iterations`` and number of parameters ``num_params``. Optionally can also contain ``train_test_ratio`` that
        is a float between 0 and 1, specifying how much of the data is used for training and testing.

    Returns
    -------
    sa_dict : dict
        Dictionary that contains computed sensitivity indices.

    References
    ----------
    Paper:
        XGBoost: A Scalable Tree Boosting System.
        Tianqi Chen, Carlos Guestrin.
        http://dx.doi.org/10.1145/2939672.2939785

    Link to XGBoost library:
        https://xgboost.readthedocs.io/en/latest/index.html

    """

    # 1. Preparations
    num_params = X.shape[1]
    if tuning_parameters is None:
        tuning_parameters = {}
    tuning_parameters["base_score"] = np.mean(Y)
    random_state = tuning_parameters.get("random_state", None)
    num_boost_round = tuning_parameters.get("n_estimators")
    tuning_parameters.pop("n_estimators")
    # 3. Prepare training and testing sets for  gradient boosting trees
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state,
    )

    dtrain = xgb.DMatrix(X_train, Y_train)
    X_dtest = xgb.DMatrix(X_test)

    # 4. Train the model
    xgb_model_current = xgb.train(
        tuning_parameters,
        dtrain,
        num_boost_round=num_boost_round,
        xgb_model=xgb_model,
    )

    # 5. make predictions and compute prediction score
    y_pred = xgb_model_current.predict(X_dtest)
    r2 = r2_score(Y_test, y_pred)
    explained_variance = explained_variance_score(Y_test, y_pred)

    S_dict = {
        "stat.r2": r2,
        "stat.explained_variance": explained_variance,
    }
    if flag_return_xgb_model:
        S_dict["stat.xgb_model"] = xgb_model_current

    # 6. Save importance scores
    if importance_types is None:
        importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    for importance_type in importance_types:
        importance_scores_ = xgb_model_current.get_score(
            importance_type=importance_type
        )
        importance_scores = {
            int(key[1:]): val for key, val in importance_scores_.items()
        }
        importance_scores_arr = np.array(
            [importance_scores.get(i, 0) for i in range(num_params)]
        )
        S_dict.update(
            {importance_type: importance_scores_arr / np.sum(importance_scores_arr)}
        )

    return S_dict


def xgboost_scores_stability(
    Y,
    X,
    tuning_parameters=None,
    test_size=0.2,
    xgb_model=None,
    importance_types=None,  # TODO set default to empty list?
    flag_return_xgb_model=False,
):
    S_dict = xgboost_scores_base(
        Y,
        X,
        tuning_parameters,
        test_size,
        xgb_model,
        importance_types,
        flag_return_xgb_model,
    )
    return S_dict
