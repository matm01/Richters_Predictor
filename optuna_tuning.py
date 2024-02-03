"""
Optuna search that optimizes a classifier configuration for Richters dataset
using XGBoost.
"""

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

def transform_data(X_train, y_train, X_valid, y_valid, preprocessor):
    """Transforms the training and validation data using the provided preprocessor.

    Returns:
    tuple: Transformed training input samples, transformed training target values, transformed validation input samples 
    and transformed validation target values.
    """
    opt_X_train = preprocessor.fit_transform(X_train, y_train['damage_grade'])
    opt_X_valid = preprocessor.transform(X_valid)
    opt_y_train = y_train['damage_grade'].values
    opt_y_valid = y_valid['damage_grade'].values

    return opt_X_train, opt_y_train, opt_X_valid, opt_y_valid
    


def do_study(X_train, y_train, X_valid, y_valid, preprocessor):
    """This function does an automatic hyperparameter optimization.

    It takes training and validation Data and a preprocessor pipeline as input. The function transforms the data, 
    defines an objective function for hyperparameter optimization, creates an Optuna study, and then optimizes the 
    objective function over a specified number of trials.

    Returns: 
    dict: The best Parameters are returned as the output of the function.
    """
    trf_X_train, trf_y_train, trf_X_valid, trf_y_valid = transform_data(X_train, y_train, X_valid, y_valid, preprocessor)
    dtrain = xgb.DMatrix(trf_X_train, label=trf_y_train)
    dvalid = xgb.DMatrix(trf_X_valid, label=trf_y_valid)

    def objective(trial):
        """This function defines the objective to be optimized. 
        It takes a 'trial' object as input and returns the accuracy of the model. The 'trial' object is used to suggest hyperparameters 
        for the XGBoost model. The function sets various parameters such as learning rate, booster type, max depth, gamma, etc. for the XGBoost model. 
        It then trains the model on the suggested parameters and evaluates its accuracy using the F1 score. The accuracy value is returned 
        as the output of the function.
        """
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "gamma" : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int("min_child_weight", 2, 10),
            'n_estimators': trial.suggest_int("n_estimators", 600, 1200, step=50),
            'eval_metric': 'auc',
            'seed': 42,
            'device': 'cuda'
        }

        bst = xgb.train(params, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = f1_score(trf_y_valid, pred_labels, average='micro')
        
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial