import numpy as np
import json
import os
import xgboost as xgb

def xgboost_scores(dict_):
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
    X = dict_.get('X')
    y = dict_.get('y')
    num_params = dict_.get('num_params')
    iterations = dict_.get('iterations')
    train_test_ratio = dict_.get('train_test_ratio', 0.8)
    write_dir = dict_.get('write_dir')

    # 2. Read xgboost parameters
    filename = os.path.join(write_dir, 'xgboost_params.json')
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
    except:
        params = {}

    # 3. Prepare training and testing sets for  gradient boosting trees
    n_split = int(train_test_ratio * iterations)
    X_train, X_test = X[:n_split, :], X[n_split:iterations, :]
    y_train, y_test = y[:n_split], y[n_split:iterations]
    dtrain = xgb.DMatrix(X_train, y_train)
    X_dtest = xgb.DMatrix(X_test)

    # 4. Train the model
    model = xgb.train(params, dtrain)

    # 5. make predictions and compute prediction score
    from sklearn.metrics import r2_score, explained_variance_score
    y_pred = model.predict(X_dtest)
    r2 = r2_score(y_test, y_pred)
    exp_v_score = explained_variance_score(y_test, y_pred)

    # 6. Save importance scores
    fscores_inf = model.get_fscore()
    fscores_dict = {int(key[1:]): val for key, val in fscores_inf.items()}
    fscores_all = np.array([fscores_dict.get(i,0) for i in range(num_params)])

    sa_dict = {
        'fscores': fscores_all,
        # 'r2': r2,
        # 'explained_variance': exp_v_score,
    }
    # print('r2 = ' + str(r2))

    return sa_dict