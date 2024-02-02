"""
Optuna search that optimizes a classifier configuration for Richters dataset
using XGBoost.

"""

import numpy as np
import optuna
import sklearn.metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def transform_data(X_train, y_train, X_valid, y_valid, preprocessor):
    opt_X_train = preprocessor.fit_transform(X_train, y_train['damage_grade'])
    opt_X_valid = preprocessor.transform(X_valid)
    opt_y_train = y_train['damage_grade'].values
    opt_y_valid = y_valid['damage_grade'].values

    return opt_X_train, opt_y_train, opt_X_valid, opt_y_valid
    


def do_study(X_train, y_train, X_valid, y_valid, preprocessor):

    opt_X_train, opt_y_train, opt_X_valid, opt_y_valid = transform_data(X_train, y_train, X_valid, y_valid, preprocessor)
    def objective(trial):
        # X_train, y_train, X_valid, y_valid = transform_data(X_train, y_train, X_valid, y_valid, preprocessor)

        """Define the objective function"""
        params = {
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "gamma" : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int("min_child_weight", 2, 10),
            # 'n_estimators': trial.suggest_int("n_estimators", 600, 1200, step=50),
            'eval_metric': 'auc',
            'use_label_encoder': False,
            # 'n_estimators': 1000,
            'seed': 42,
            'device': 'cuda'
        }

        # Fit the model
        optuna_model = XGBClassifier(**params)
        optuna_model.fit(opt_X_train, opt_y_train)

        # Make predictions
        y_pred = optuna_model.predict(opt_X_valid)

        # Evaluate predictions
        accuracy = f1_score(opt_y_valid, y_pred, average='micro')
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)#, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial.params