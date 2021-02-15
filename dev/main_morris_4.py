from gsa_framework.test_functions import Morris4
from gsa_framework.methods.gradient_boosting import GradientBoosting
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import xgboost as xgb
from sklearn.metrics import explained_variance_score, r2_score

if __name__ == "__main__":

    #     path_base = Path('/data/user/kim_a/paper_gsa/')
    path_base = Path("/Users/akim/PycharmProjects/gsa_framework/dev/write_files/")

    # 1. Models
    num_params = 1000
    num_influential = max(num_params // 100, 10)
    write_dir = path_base / "{}_morris4".format(num_params)
    model = Morris4(num_params=num_params, num_influential=num_influential)
    gsa_seed = 3407
    validation_seed = 7043

    fig_format = []  # can have elements "pdf", "html", "pickle"

    iterations = 2 * num_params
    test_size = 0.2

    option = "no tuning"
    if "tuning" in option:
        # 1. Preparations
        np.random.seed(gsa_seed)
        X = np.random.rand(iterations, num_params)
        Y = model(X)
        # 2. Prepare training and testing sets for  gradient boosting trees
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=gsa_seed,
        )

        dtrain = xgb.DMatrix(X_train, Y_train)
        X_dtest = xgb.DMatrix(X_test)

        if option == "tuning":
            ### ROUND 1 ###
            # xgb.train uses parameter `num_boost_round`, while XGBRegressor needs `n_estimators`. These two are the same.
            param_grid = {
                "learning_rate": [0.1],
                "gamma": [0],
                "min_child_weight": [30, 100],
                "max_depth": [2],
                "reg_lambda": [0, 10],
                "reg_alpha": [0, 10],
                "n_estimators": [500],
                "subsample": [0.1, 0.3],
                "colsample_bytree": [0.1, 0.3],
            }

            optimal_params = GridSearchCV(
                estimator=xgb.XGBRegressor(
                    objective="reg:squarederror",
                    seed=gsa_seed,
                ),
                param_grid=param_grid,
                scoring="explained_variance",  # explained_variance takes into account mean squared error, r2 does not. former is unbiasede, so better than r2
                cv=3,
            )
            optimal_params.fit(
                X_train,
                Y_train,
                early_stopping_rounds=10,
                eval_set=[(X_test, Y_test)],
                verbose=False,
            )

            print(optimal_params.best_params_)

            import pickle

            filepath = write_dir / "arrays" / "optimal_params_round_1.pickle"
            if filepath.exists():
                filepath = write_dir / "arrays" / "optimal_params_round_2.pickle"
            with open(filepath, "wb") as f:
                pickle.dump(optimal_params, f)

        elif option == "no tuning":
            np.random.seed(None)
            reg = xgb.XGBRegressor(
                verbosity=1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
                objective="reg:squarederror",
                seed=gsa_seed,
                learning_rate=0.1,
                gamma=0,
                min_child_weight=30,
                max_depth=2,
                reg_lambda=10,
                reg_alpha=0,
                n_estimators=500,
                subsample=0.6,
                colsample_bytree=0.3,
            )
            reg.fit(X_train, Y_train)
            ev_train = explained_variance_score(reg.predict(X_train), Y_train)
            ev_test = explained_variance_score(reg.predict(X_test), Y_test)
            print(ev_train, ev_test)

    else:
        tuning_parameters = dict(
            learning_rate=0.1,
            gamma=0,
            min_child_weight=30,
            max_depth=2,
            reg_lambda=10,
            reg_alpha=0,
            n_estimators=500,
            subsample=0.6,
            colsample_bytree=0.3,
        )
        gsa = GradientBoosting(
            iterations=iterations,
            model=model,
            write_dir=write_dir,
            seed=gsa_seed,
            tuning_parameters=tuning_parameters,
            test_size=test_size,
            xgb_model=None,
        )

        S_dict = gsa.perform_gsa(flag_save_S_dict=True)
        print(S_dict["stat.r2"], S_dict["stat.explained_variance"])
        gsa.plot_sa_results(
            {"fscores": S_dict["fscores"]},
            fig_format=fig_format,
        )
