# author: Mai Le
# date: 2020-11-27
"""Reads the data from the data clean-up script, performs some statistical or
machine learning analysis and summarizes the results as a figure(s) and
a table(s).

Usage: src/04_build_model.py --data_path=<data_path> --out_report_path=<out_report_path> [--random_state=<random_state>] [--tune_params=<tune_params>]

Options:
--data_path=<data_path>                The path containing train & test dataset
--out_report_path=<out_report_path>    The path to export model scores in figures and tables
--random_state=<random_state>          The random state that we want to use for splitting. [default: 2020]
--tune_params=<tune_params>            Whether we need to tune hyperparameters or not [default: True]
"""

# region import libraries
from sklearn.compose import (
    make_column_transformer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

import scipy
from scipy.stats import loguniform

import pickle

from sklearn.feature_selection import RFECV
from docopt import docopt

import pandas as pd
import numpy as np

import utils
import shopping_data_reader as sdr
# endregion

PARAM_DIST = "param_dist"


# region main function
def main(data_path, out_report_path, random_state, tune_params):
    # read the data files, split into X and y
    print("Start build_model script")

    random_state = int(random_state)
    tune = True if tune_params == 'True' else False

    if tune:
        print("We will tune the hyperparameters")
    else:
        print("We will use the predefined hyperamater values")

    print("Read the data files, split into X and y")
    X_train, y_train, X_test, y_test = sdr.read_data_as_xy(data_path)

    preprocessor = make_column_transformer(
        ("drop", sdr.drop_features),
        (StandardScaler(), sdr.numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), sdr.categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"),
            sdr.binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and
    # LogisticRegression
    print("Process models")
    hyperparams_best_model = tune_hyperparams(
        preprocessor, X_train, y_train, random_state, tune
    )

    # find the best model
    print("Finding the best model using cross validation")
    _, best_model, _ = find_best_model(
        hyperparams_best_model,
        X_train,
        y_train,
        random_state,
        data_path + "/model_selection_result.csv",
    )

    # get result plots
    print("Creating plots and classification report")
    plot, class_report = utils.plot_results(
        best_model, X_test, y_test, ["No-revenue", "Revenue"]
    )

    # save plots to report path
    print("Saving reports")
    utils.save_plots(
        out_report_path,
        plot,
        class_report,
        ("confusion_matrix", "classification_report"),
    )

    # save model to disk
    print("Saving the best model for later use")
    pickle.dump(best_model, open(data_path + "/best_model.sav", "wb"))

    # try feature selection
    print("Building the model with RFE")

    fs_model = make_pipeline(
        preprocessor,
        RFECV(Ridge(random_state=random_state), cv=10),
        RandomForestClassifier(max_depth=12, n_estimators=275),
    )
    fs_model.fit(X_train, y_train)

    plot, class_report = utils.plot_results(
        fs_model, X_test, y_test, ["No-revenue", "Revenue"]
    )

    # save plots to report path
    print("Saving reports")
    utils.save_plots(
        out_report_path,
        plot,
        class_report,
        (
            "confusion_matrix_feature_selection",
            "classification_report_feature_selection",
        ),
    )
    print("End build_model script")
    return


def create_logistic_regression_model(
    random_state, tune=True, class_balanced=True
):
    """Create a logistic regression model using best hyperparameters or tuning

    Parameters
    ----------
    tune : bool, optional
        tune the hyperparameter or using the best values, by default True

    Returns
    -------
    Logistic Regression Model
        The model we want to create
    """

    class_weight = "balanced"

    if not class_balanced:
        class_weight = None

    model = {
        "clf": LogisticRegression(
            class_weight=class_weight,
            random_state=random_state,
            max_iter=1000
        )
    }

    if tune:
        model[PARAM_DIST] = {
            "logisticregression__C": loguniform(1e-3, 1e3)
        }
    else:
        if class_balanced:
            model[PARAM_DIST] = {
                "logisticregression__C": [0.008713608033492446]
            }
        else:
            model[PARAM_DIST] = {
                "logisticregression__C": [0.008713608033492446]
        }
    return model


def create_random_forest_model(
    random_state, tune=True, class_balanced=True
):
    """Create a random forest model using best hyperparameters or tuning

    Parameters
    ----------
    tune : bool, optional
        tune the hyperparameter or using the best values, by default True

    Returns
    -------
    Random Forest Model
        The model we want to create
    """

    class_weight = "balanced"

    if not class_balanced:
        class_weight = None

    model = {
        "clf": RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state
        )
    }

    if tune:
        model[PARAM_DIST] = {
            "randomforestclassifier__n_estimators":
                scipy.stats.randint(low=10, high=300),
            "randomforestclassifier__max_depth":
                scipy.stats.randint(low=2, high=20)
        }
    else:
        if class_balanced:
            model[PARAM_DIST] = {
                "randomforestclassifier__n_estimators": [65],
                "randomforestclassifier__max_depth": [12],
            }
        else:
            model[PARAM_DIST] = {
            "randomforestclassifier__n_estimators": [170],
            "randomforestclassifier__max_depth": [18],
        }
    return model


def create_SVC_model(
    random_state, tune=True, class_balanced=True
):
    """Create a SVC model using best hyperparameters or tuning

    Parameters
    ----------
    tune : bool, optional
        tune the hyperparameter or using the best values, by default True

    Returns
    -------
    SVC Model
        The model we want to create
    """

    class_weight = "balanced"

    if not class_balanced:
        class_weight = None

    model = {
        "clf": SVC(
            class_weight=class_weight,
            random_state=random_state
        )
    }

    if tune:
        model[PARAM_DIST] = {
            "svc__gamma": [0.1, 1.0, 10, 100],
            "svc__C": [0.1, 1.0, 10, 100],
        }
    else:
        if class_balanced:
            model[PARAM_DIST] = {
                "svc__gamma": [0.1],
                "svc__C": [1.0]
            }
        else:
            model[PARAM_DIST] = {
                "svc__gamma": [0.1],
                "svc__C": [1.0]
            }
    return model


def tune_hyperparams(preprocessor, X, y, random_state, tune=True):
    """tuning hyperparameters for LogisticRegression, RandomForestClassifier
    and SVC with preprocessor on X, y with random_state using
    RandomizedSearchCV

    Args:
        preprocessor (Pipeline/ColumnTransformer): a Pipeline to transform X
        X (DataFrame): features
        y (DataFrame): target
        random_state (integer): random state

    Returns:
        dict: a dictionary with key=model's name,
        value={"best_model":best_model, "best_params":best_params}

    Examples:
        perparams_best_model = tune_hyperparams(preprocessor, X_train, y_train,
            2020)
    """
    classifiers = {
        "Logistic Regression Balanced":
            create_logistic_regression_model(random_state, tune),
        "Logistic Regression SMOTE":
            create_logistic_regression_model(random_state, tune, False),
        "Random Forest Balanced":
            create_random_forest_model(random_state, tune),
        "Random Forest SMOTE":
            create_random_forest_model(random_state, tune, False),
        "SVC Balanced":
            create_SVC_model(random_state, tune),
        "SVC SMOTE":
            create_SVC_model(random_state, tune, False)
    }

    hyperparams_best_model = {}

    n_iter = 10 if tune else 1

    # find the best hyperparameters of each model
    for name, model_dict in classifiers.items():
        print("Processing", name)

        if "SMOTE" in name:
            pipe = make_imb_pipeline(preprocessor, SMOTE(),  model_dict["clf"])
        else:
            pipe = make_pipeline(preprocessor, model_dict["clf"])

        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=model_dict[PARAM_DIST],
            n_iter=n_iter,
            verbose=1,
            n_jobs=-1,
            scoring="f1",
            cv=10,
            random_state=random_state,
        )
        random_search.fit(X, y)
        hyperparams_best_model[name] = {
            "best_model": random_search.best_estimator_,
            "best_params": random_search.best_params_,
        }

    # print(hyperparams_best_model)
    return hyperparams_best_model


def find_best_model(
    hyperparams_best_model,
    X,
    y,
    random_state,
    path="../data/processed/model_selection_result.csv"
):
    """find the best model among 3 models using cross validation

    Args:
        hyperparams_best_model (dict): a dictionary with key=model's name,
            value={"best_model":best_model, "best_params":best_params}
        X (DataFrame): features
        y (DataFrame): target
        random_state (int): random_state

    Returns:
        (string, Pipeline, list): a tuple consisting of best model's name,
            best model object and its best hyperparameters

    Examples:
        best_model_name, best_model, best_params =
            find_best_model(hyperparams_best_model, X_train, y_train, 2020)
    """
    scoring = ["accuracy", "recall", "f1"]
    clf_score_map = [
        "fit_time",
        "score_time",
        "test_accuracy",
        "test_f1",
        "test_recall",
    ]
    results = {}
    for name, model in hyperparams_best_model.items():
        cv_scores = cross_validate(
            model["best_model"], X, y, cv=10, n_jobs=-1, scoring=scoring
        )
        utils.store_cross_val_results(clf_score_map, name, cv_scores, results)
    results_df = pd.DataFrame(results).T
    print(results_df)
    results_df.reset_index().to_csv(path)
    best_model_name = results_df.iloc[np.argmax(results_df["test_f1"])].name
    best_model = hyperparams_best_model[best_model_name]["best_model"]
    best_params = hyperparams_best_model[best_model_name]["best_params"]
    return (best_model_name, best_model, best_params)


def retrieve_important_features(
    preprocessor,
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    binary_features,
    random_state
):
    """return a list of features that are important in predicting the target
        using RFE

    Args:
        preprocessor (ColumnTransformer): feature transformation
        X_train (DataFrame): train feature set
        y_train (DataFrame): train target
        random_state (int): random_state

    Returns:
        [list]: list of important features

    Examples:
        retrieve_important_features(preprocessor, X_train, y_train, 2020)
    """
    rfecv = make_pipeline(
        preprocessor, RFECV(Ridge(random_state=random_state), cv=10)
    )
    rfecv.fit(X_train, y_train)
    ohe_columns = list(
        rfecv.named_steps["columntransformer"]
        .transformers_[2][1]
        .get_feature_names(categorical_features)
    )
    bin_columns = list(
        rfecv.named_steps["columntransformer"]
        .transformers_[3][1]
        .get_feature_names(binary_features)
    )
    new_columns = numerical_features + ohe_columns + bin_columns
    new_columns = np.array(new_columns)
    return new_columns[rfecv.named_steps["rfecv"].support_]


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(
        opt["--data_path"],
        opt["--out_report_path"],
        opt["--random_state"],
        opt["--tune_params"],
    )
